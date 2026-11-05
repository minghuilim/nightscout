import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.nightscout_client import (
    health_check,
    get_server_status,
    get_entries,
    get_entries_by_range,
    get_treatments,
    get_treatments_by_range,
    add_treatment,
    delete_treatment,
    get_profiles,
    update_profile,
    get_device_status,
    get_aggregated_glucose_stats,
)

AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN")
EVENT_TIME_FUTURE_TOLERANCE = timedelta(minutes=5)
SAME_TIME_LOOKUP_WINDOW = timedelta(seconds=1)

mcp = FastMCP("nightscout")


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse

    api_ok = health_check()
    status = "ok" if api_ok else "degraded"
    return JSONResponse({"status": status, "nightscout_api": api_ok})


# =============================================================================
# Status & Settings tools
# =============================================================================


@mcp.tool()
def server_status() -> dict:
    """Get Nightscout server status including version, settings, enabled plugins,
    thresholds (urgent high, high, low, urgent low), units (mg/dL or mmol/L),
    and other server configuration."""
    return get_server_status()


@mcp.tool()
def get_current_profile() -> list[dict]:
    """Get all Nightscout profiles. Each profile contains:
    - Basal rates schedule (units/hour by time of day)
    - Insulin Sensitivity Factor (ISF) schedule
    - Insulin-to-Carb Ratio (IC/ICR) schedule
    - Target BG range schedule
    - DIA (Duration of Insulin Action)
    - Timezone and units

    These are the core settings that control insulin dosing calculations."""
    return get_profiles()


# =============================================================================
# CGM / Glucose reading tools
# =============================================================================


@mcp.tool()
def get_current_glucose() -> dict:
    """Get the most recent CGM glucose reading with value (sgv), direction/trend
    arrow, noise level, and timestamp. Returns a single entry."""
    entries = get_entries(count=1)
    if not entries:
        return {"error": "No glucose entries found"}
    entry = entries[0]
    return {
        "glucose_mgdl": entry.get("sgv"),
        "direction": entry.get("direction"),
        "trend": entry.get("trend"),
        "noise": entry.get("noise"),
        "timestamp": entry.get("dateString"),
        "device": entry.get("device"),
        "type": entry.get("type"),
    }


@mcp.tool()
def get_recent_glucose(count: int = 36) -> list[dict]:
    """Get recent CGM glucose readings. Default 36 entries (~3 hours at 5-min intervals).

    Args:
        count: Number of entries to return (default 36 = ~3 hours)
    """
    entries = get_entries(count=count)
    return [
        {
            "glucose_mgdl": e.get("sgv"),
            "direction": e.get("direction"),
            "timestamp": e.get("dateString"),
            "noise": e.get("noise"),
        }
        for e in entries
    ]


@mcp.tool()
def get_glucose_by_date_range(date_from: str, date_to: str, count: int = 1000) -> list[dict]:
    """Get raw CGM glucose readings for a short date range.

    For long-term ranges, prefer get_glucose_stats_batch() to avoid
    many repeated calls and client-side rate limits.

    Args:
        date_from: Start date in ISO format (e.g. '2024-01-15T00:00:00Z')
        date_to: End date in ISO format (e.g. '2024-01-15T23:59:59Z')
        count: Maximum number of entries (default 1000)
    """
    entries = get_entries_by_range(date_from, date_to, count)
    return [
        {
            "glucose_mgdl": e.get("sgv"),
            "direction": e.get("direction"),
            "timestamp": e.get("dateString"),
        }
        for e in entries
    ]


@mcp.tool()
def calculate_time_in_range(
    hours: int = 24,
    low: int = 70,
    high: int = 180,
) -> dict:
    """Calculate Time-In-Range (TIR) statistics for the last N hours.
    Provides percentage of time in range, below range, and above range.

    Args:
        hours: Number of hours to analyze (default 24)
        low: Low threshold in mg/dL (default 70)
        high: High threshold in mg/dL (default 180)
    """
    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(hours=hours)).isoformat()
    date_to = now.isoformat()

    entries = get_entries_by_range(date_from, date_to, count=2000)
    if not entries:
        return {"error": "No glucose data found for the specified period"}

    values = [e["sgv"] for e in entries if "sgv" in e]
    total = len(values)
    if total == 0:
        return {"error": "No valid glucose readings found"}

    below = sum(1 for v in values if v < low)
    above = sum(1 for v in values if v >= high)
    in_range = total - below - above
    very_low = sum(1 for v in values if v < 54)
    very_high = sum(1 for v in values if v > 250)

    avg_glucose = sum(values) / total
    # Estimated HbA1c (DCCT) = (avg_glucose + 46.7) / 28.7
    estimated_hba1c = round((avg_glucose + 46.7) / 28.7, 1)

    std_dev = (sum((v - avg_glucose) ** 2 for v in values) / total) ** 0.5
    cv = round((std_dev / avg_glucose) * 100, 1) if avg_glucose > 0 else 0

    return {
        "period_hours": hours,
        "total_readings": total,
        "average_glucose_mgdl": round(avg_glucose, 1),
        "std_dev": round(std_dev, 1),
        "coefficient_of_variation": cv,
        "estimated_hba1c": estimated_hba1c,
        "time_in_range_pct": round(in_range / total * 100, 1),
        "time_below_range_pct": round(below / total * 100, 1),
        "time_above_range_pct": round(above / total * 100, 1),
        "time_very_low_pct": round(very_low / total * 100, 1),
        "time_very_high_pct": round(very_high / total * 100, 1),
        "min_glucose": min(values),
        "max_glucose": max(values),
        "thresholds": {"low": low, "high": high, "very_low": 54, "very_high": 250},
    }


@mcp.tool()
def get_daily_glucose_stats(date: str) -> dict:
    """Get detailed glucose statistics for a specific day: average, median,
    standard deviation, min, max, time in range, and hourly averages.

    Args:
        date: Date in YYYY-MM-DD format (e.g. '2024-01-15')
    """
    date_from = f"{date}T00:00:00Z"
    date_to = f"{date}T23:59:59Z"

    entries = get_entries_by_range(date_from, date_to, count=2000)
    if not entries:
        return {"error": f"No glucose data found for {date}"}

    values = [e["sgv"] for e in entries if "sgv" in e]
    if not values:
        return {"error": "No valid glucose readings found"}

    total = len(values)
    avg = sum(values) / total
    sorted_vals = sorted(values)
    median = sorted_vals[total // 2]
    std_dev = (sum((v - avg) ** 2 for v in values) / total) ** 0.5

    # Hourly breakdown
    hourly: dict[int, list] = {}
    for e in entries:
        if "sgv" in e and "dateString" in e:
            try:
                ts = datetime.fromisoformat(e["dateString"].replace("Z", "+00:00"))
                hour = ts.hour
                hourly.setdefault(hour, []).append(e["sgv"])
            except (ValueError, KeyError):
                pass

    hourly_avg = {
        h: round(sum(vals) / len(vals), 1)
        for h, vals in sorted(hourly.items())
    }

    return {
        "date": date,
        "total_readings": total,
        "average": round(avg, 1),
        "median": median,
        "std_dev": round(std_dev, 1),
        "min": min(values),
        "max": max(values),
        "hourly_averages": hourly_avg,
    }


# =============================================================================
# Treatment tools (insulin, carbs, etc.)
# =============================================================================


@mcp.tool()
def get_recent_treatments(count: int = 20) -> list[dict]:
    """Get recent treatments (insulin boluses, carb entries,
    temp basals, notes, etc.).

    Args:
        count: Number of treatments to return (default 20)
    """
    return get_treatments(count=count)


@mcp.tool()
def get_treatments_by_date(date_from: str, date_to: str, count: int = 500) -> list[dict]:
    """Get treatments for a specific date range.

    Args:
        date_from: Start date in ISO format
            (e.g. '2024-01-15T00:00:00Z')
        date_to: End date in ISO format
            (e.g. '2024-01-15T23:59:59Z')
        count: Maximum number of treatments (default 500)
    """
    return get_treatments_by_range(date_from, date_to, count)


@mcp.tool()
def get_insulin_on_board() -> dict:
    """Estimate current Insulin on Board (IOB) from recent boluses and the latest
    device status. Returns the latest loop/pump device status which typically
    includes IOB calculation from the loop algorithm."""
    statuses = get_device_status(count=1)
    if not statuses:
        return {"error": "No device status found"}
    status = statuses[0]
    result = {
        "timestamp": status.get("created_at"),
        "device": status.get("device"),
    }
    # Extract IOB from loop or openaps status
    if "loop" in status:
        loop = status["loop"]
        if "iob" in loop:
            result["iob"] = loop["iob"]
    if "openaps" in status:
        openaps = status["openaps"]
        if "iob" in openaps:
            result["iob"] = openaps["iob"]
    if "pump" in status:
        result["pump"] = status["pump"]
    return result


class EventTimeError(ValueError):
    """An event_time that cannot be used for a backdated treatment."""


def _to_nightscout_utc(moment: datetime) -> str:
    """Format a time the way Nightscout stores created_at."""
    utc = moment.astimezone(timezone.utc)
    millis = utc.microsecond // 1000
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z"


def _parse_event_time(event_time: str) -> datetime:
    """Parse an ISO 8601 time that states its timezone offset."""
    try:
        moment = datetime.fromisoformat(event_time)
    except ValueError:
        raise EventTimeError(
            f"event_time '{event_time}' is not an ISO 8601 time, "
            "e.g. '2026-11-04T08:10:00+08:00'"
        )
    if moment.tzinfo is None:
        raise EventTimeError(
            f"event_time '{event_time}' has no timezone offset; "
            "add one, e.g. '+08:00' or 'Z'"
        )
    latest = datetime.now(timezone.utc) + EVENT_TIME_FUTURE_TOLERANCE
    if moment > latest:
        raise EventTimeError(
            f"event_time '{event_time}' is in the future; "
            "treatments can only be logged for now or the past"
        )
    return moment.replace(microsecond=0)


def _same_time_treatment_exists(
    event_type: str, moment: datetime
) -> bool:
    """True if a treatment of this type is already stored at this time.

    Nightscout can merge a new treatment into one with the same
    created_at and eventType, so a match must not be written over.
    """
    stored_at = _to_nightscout_utc(moment)
    nearby = get_treatments_by_range(
        _to_nightscout_utc(moment - SAME_TIME_LOOKUP_WINDOW),
        _to_nightscout_utc(moment + SAME_TIME_LOOKUP_WINDOW),
        count=50,
    )
    return any(
        t.get("eventType") == event_type
        and t.get("created_at") == stored_at
        for t in nearby
    )


def _backdated_created_at(event_type: str, event_time: str) -> str:
    """Validate event_time and return it as created_at for Nightscout."""
    moment = _parse_event_time(event_time)
    if _same_time_treatment_exists(event_type, moment):
        raise EventTimeError(
            f"A '{event_type}' treatment is already logged at "
            f"{moment.isoformat()}. Nothing was saved. Use a time a "
            "minute apart, or combine the two into one entry."
        )
    return moment.isoformat()


@mcp.tool()
def log_treatment(
    event_type: str,
    glucose: Optional[int] = None,
    glucose_type: Optional[str] = None,
    carbs: Optional[float] = None,
    insulin: Optional[float] = None,
    duration: Optional[int] = None,
    notes: Optional[str] = None,
    event_time: Optional[str] = None,
) -> any:
    """Log a new treatment entry to Nightscout.

    Args:
        event_type: Type of treatment. Common types:
            'Correction Bolus', 'Meal Bolus', 'Carb Correction',
            'Snack Bolus', 'BG Check', 'Note', 'Exercise',
            'Temp Basal', 'Profile Switch', 'Site Change',
            'Insulin Change', 'Sensor Start', 'Sensor Change'
        glucose: Optional blood glucose value in mg/dL
        glucose_type: Optional glucose measurement type ('Finger' or 'Sensor')
        carbs: Optional carbs in grams
        insulin: Optional insulin in units
        duration: Optional duration in minutes (for temp basals, exercise)
        notes: Optional text note
        event_time: Optional time the treatment actually happened, for
            backdating. ISO 8601 with a timezone offset, e.g.
            '2026-11-04T08:10:00+08:00'. Must not be in the future.
            Omit to use the current time. For a past glucose reading,
            use event_type 'BG Check' with glucose and glucose_type.
    """
    if event_time:
        try:
            created_at = _backdated_created_at(event_type, event_time)
        except EventTimeError as error:
            return {"error": str(error)}
    else:
        created_at = datetime.now(timezone.utc).isoformat()

    treatment: dict = {
        "eventType": event_type,
        "created_at": created_at,
    }
    if glucose is not None:
        treatment["glucose"] = glucose
    if glucose_type:
        treatment["glucoseType"] = glucose_type
    if carbs is not None:
        treatment["carbs"] = carbs
    if insulin is not None:
        treatment["insulin"] = insulin
    if duration is not None:
        treatment["duration"] = duration
    if notes:
        treatment["notes"] = notes
    return add_treatment(treatment)


@mcp.tool()
def remove_treatment(treatment_id: str) -> any:
    """Delete a treatment entry from Nightscout by its ID.

    Args:
        treatment_id: The _id of the treatment to delete
    """
    return delete_treatment(treatment_id)


# =============================================================================
# Device status tools
# =============================================================================


@mcp.tool()
def get_latest_device_status(count: int = 1) -> list[dict]:
    """Get the latest device status entries. Contains pump status, loop/openaps
    decisions, CGM status, uploader battery, and IOB/COB from the loop algorithm.

    Args:
        count: Number of device status entries (default 1)
    """
    return get_device_status(count=count)


# =============================================================================
# Profile management tools
# =============================================================================


@mcp.tool()
def update_nightscout_profile(profile: dict) -> any:
    """Update a Nightscout profile. The profile dict should contain the full
    profile document with fields like:
    - defaultProfile: name of the active profile
    - store: dict of profiles, each containing:
      - dia: Duration of Insulin Action (hours)
      - carbratio: list of {time, value} for IC ratios
      - sens: list of {time, value} for ISF
      - basal: list of {time, value} for basal rates
      - target_low: list of {time, value} for low target BG
      - target_high: list of {time, value} for high target BG
      - units: 'mg/dL' or 'mmol/L'
      - timezone: timezone string

    WARNING: This modifies active treatment settings. Verify values carefully.

    Args:
        profile: Full profile document to save
    """
    return update_profile(profile)


# =============================================================================
# Analysis tools
# =============================================================================


@mcp.tool()
def analyze_glucose_patterns(hours: int = 168) -> dict:
    """Analyze glucose patterns over a period to identify:
    - Recurring highs/lows by time of day
    - Average glucose by hour
    - Days with best/worst control
    - Patterns that may suggest basal or bolus ratio adjustments

    Args:
        hours: Number of hours to analyze (default 168 = 7 days)
    """
    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(hours=hours)).isoformat()
    date_to = now.isoformat()

    entries = get_entries_by_range(date_from, date_to, count=5000)
    if not entries:
        return {"error": "No glucose data found for the specified period"}

    # Build hourly and daily groupings
    hourly: dict[int, list] = {}
    daily: dict[str, list] = {}

    for e in entries:
        if "sgv" not in e or "dateString" not in e:
            continue
        try:
            ts = datetime.fromisoformat(e["dateString"].replace("Z", "+00:00"))
            hourly.setdefault(ts.hour, []).append(e["sgv"])
            day = ts.strftime("%Y-%m-%d")
            daily.setdefault(day, []).append(e["sgv"])
        except (ValueError, KeyError):
            pass

    hourly_stats = {}
    for h in range(24):
        vals = hourly.get(h, [])
        if vals:
            avg = sum(vals) / len(vals)
            below_70 = sum(1 for v in vals if v < 70)
            above_180 = sum(1 for v in vals if v > 180)
            hourly_stats[h] = {
                "avg": round(avg, 1),
                "min": min(vals),
                "max": max(vals),
                "readings": len(vals),
                "pct_below_70": round(below_70 / len(vals) * 100, 1),
                "pct_above_180": round(above_180 / len(vals) * 100, 1),
            }

    daily_stats = {}
    for day, vals in sorted(daily.items()):
        avg = sum(vals) / len(vals)
        in_range = sum(1 for v in vals if 70 <= v <= 180)
        daily_stats[day] = {
            "avg": round(avg, 1),
            "tir_pct": round(in_range / len(vals) * 100, 1),
            "readings": len(vals),
        }

    # Find problematic hours (consistently high or low)
    problem_hours = []
    for h, stats in hourly_stats.items():
        if stats["pct_below_70"] > 15:
            problem_hours.append({"hour": h, "issue": "frequent_lows", "pct_low": stats["pct_below_70"]})
        if stats["pct_above_180"] > 40:
            problem_hours.append({"hour": h, "issue": "frequent_highs", "pct_high": stats["pct_above_180"]})

    return {
        "period_hours": hours,
        "total_readings": len(entries),
        "hourly_stats": hourly_stats,
        "daily_stats": daily_stats,
        "problem_hours": problem_hours,
    }


@mcp.tool()
def get_glucose_stats_batch(date_from: str, date_to: str) -> dict:
    """Get aggregated glucose statistics for a long date range.

    Uses one Nightscout range query and aggregates server-side.

    Args:
        date_from: Start date in ISO format (e.g. '2026-01-03T00:00:00Z')
        date_to: End date in ISO format (e.g. '2026-04-03T23:59:59Z')
    """
    return get_aggregated_glucose_stats(date_from, date_to)


# =============================================================================
# Auth middleware (identical to mcp_duckdb)
# =============================================================================


class TokenPathAuthMiddleware(BaseHTTPMiddleware):
    """Auth via token in URL path: /mcp/<token> or Bearer header.

    Accepts:
      - POST /mcp/<token>  (for Claude.ai connectors)
      - POST /mcp with Authorization: Bearer <token>  (for Claude Code CLI)
      - GET /health  (always open, no auth)
    """

    async def dispatch(self, request, call_next):
        path = request.url.path

        # Health check — always open
        if path == "/health":
            return await call_next(request)

        if not AUTH_TOKEN:
            return await call_next(request)

        # Check token in path: /mcp/<token> -> rewrite to /mcp
        if path.startswith("/mcp/"):
            token = path[len("/mcp/"):]
            if token == AUTH_TOKEN:
                request.scope["path"] = "/mcp"
                return await call_next(request)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        # Check Bearer header for /mcp
        if path == "/mcp":
            auth_header = request.headers.get("authorization", "")
            if auth_header == f"Bearer {AUTH_TOKEN}":
                return await call_next(request)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return JSONResponse({"error": "Not Found"}, status_code=404)


app = mcp.http_app(stateless_http=True)
app.add_middleware(TokenPathAuthMiddleware)
