"""Unit tests for log_treatment, including backdated entries."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.server import log_treatment


def _log(stored=(), **kwargs):
    """Call log_treatment with Nightscout replaced by mocks."""
    with patch(
        "src.server.get_treatments_by_range", return_value=list(stored)
    ) as lookup, patch(
        "src.server.add_treatment", return_value=[{"ok": True}]
    ) as add:
        result = log_treatment(**kwargs)
    return result, add, lookup


def _sent(add):
    """The single treatment that was posted to Nightscout."""
    return add.call_args.args[0]


def test_treatment_without_event_time_is_stamped_now():
    # Given: no event_time
    before = datetime.now(timezone.utc)

    # When: logging an insulin dose
    _, add, _ = _log(event_type="Correction Bolus", insulin=2)

    # Then: created_at is the current time, as before the change
    created_at = datetime.fromisoformat(_sent(add)["created_at"])
    assert before <= created_at <= datetime.now(timezone.utc)


def test_treatment_without_event_time_skips_duplicate_lookup():
    # Given: no event_time

    # When: logging an insulin dose
    _, _, lookup = _log(event_type="Correction Bolus", insulin=2)

    # Then: Nightscout is not queried before saving
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "event_time,expected_created_at",
    [
        ("2026-11-04T08:10:00+08:00", "2026-11-04T08:10:00+08:00"),
        ("2026-11-04T00:10:00Z", "2026-11-04T00:10:00+00:00"),
        ("2026-11-04T08:10+08:00", "2026-11-04T08:10:00+08:00"),
        ("2026-11-04T08:10:00.123+08:00", "2026-11-04T08:10:00+08:00"),
    ],
)
def test_backdated_treatment_is_saved_at_the_given_time(
    event_time: str, expected_created_at: str
):
    # Given: a past time with a timezone offset

    # When: logging a meal bolus at that time
    _, add, _ = _log(event_type="Meal Bolus", insulin=9, event_time=event_time)

    # Then: created_at carries that time and its offset
    assert _sent(add)["created_at"] == expected_created_at


def test_backdated_treatment_keeps_all_other_fields():
    # Given: a past finger-prick reading

    # When: logging it as a BG Check
    _, add, _ = _log(
        event_type="BG Check",
        glucose=180,
        glucose_type="Finger",
        notes="before gym",
        event_time="2026-11-04T21:30:00+08:00",
    )

    # Then: every field is posted unchanged
    assert _sent(add) == {
        "eventType": "BG Check",
        "created_at": "2026-11-04T21:30:00+08:00",
        "glucose": 180,
        "glucoseType": "Finger",
        "notes": "before gym",
    }


def test_backdated_treatment_returns_nightscout_response():
    # Given: Nightscout accepts the treatment

    # When: logging a backdated note
    result, _, _ = _log(
        event_type="Note", notes="x", event_time="2026-11-04T08:10:00Z"
    )

    # Then: the response from Nightscout is passed back
    assert result == [{"ok": True}]


@pytest.mark.parametrize(
    "event_time,message_part",
    [
        ("2026-11-04T08:10:00", "no timezone offset"),
        ("4 Nov 8:10am", "not an ISO 8601 time"),
        ("2026-13-04T08:10:00+08:00", "not an ISO 8601 time"),
    ],
)
def test_unusable_event_time_is_rejected_without_saving(
    event_time: str, message_part: str
):
    # Given: an event_time that does not pin down one moment

    # When: logging a treatment with it
    result, add, _ = _log(
        event_type="Meal Bolus", insulin=9, event_time=event_time
    )

    # Then: an error explains why and nothing is posted
    assert message_part in result["error"]
    add.assert_not_called()


def test_event_time_in_the_future_is_rejected_without_saving():
    # Given: a time 10 minutes from now
    later = datetime.now(timezone.utc) + timedelta(minutes=10)

    # When: logging a treatment at that time
    result, add, _ = _log(
        event_type="Meal Bolus", insulin=9, event_time=later.isoformat()
    )

    # Then: it is refused
    assert "in the future" in result["error"]
    add.assert_not_called()


def test_event_time_within_clock_tolerance_is_accepted():
    # Given: a time 2 minutes ahead, within the 5-minute tolerance
    soon = datetime.now(timezone.utc) + timedelta(minutes=2)

    # When: logging a treatment at that time
    _, add, _ = _log(
        event_type="Meal Bolus", insulin=9, event_time=soon.isoformat()
    )

    # Then: it is saved
    add.assert_called_once()


def test_same_type_at_same_time_is_rejected_without_saving():
    # Given: a Note already stored at 00:10:00 UTC
    stored = [{"eventType": "Note", "created_at": "2026-11-04T00:10:00.000Z"}]

    # When: logging another Note at the same moment in Malaysia time
    result, add, _ = _log(
        stored=stored,
        event_type="Note",
        notes="Toujeo 16u",
        event_time="2026-11-04T08:10:00+08:00",
    )

    # Then: it is refused, so the stored Note is not overwritten
    assert "already logged" in result["error"]
    add.assert_not_called()


@pytest.mark.parametrize(
    "stored",
    [
        [
            {
                "eventType": "Meal Bolus",
                "created_at": "2026-11-04T00:10:00.000Z",
            }
        ],
        [{"eventType": "Note", "created_at": "2026-11-04T00:10:01.000Z"}],
        [],
    ],
)
def test_no_clash_at_that_time_is_saved(stored: list):
    # Given: nothing of the same type at exactly that time

    # When: logging a Note at 00:10:00 UTC
    _, add, _ = _log(
        stored=stored,
        event_type="Note",
        notes="Toujeo 16u",
        event_time="2026-11-04T00:10:00Z",
    )

    # Then: it is saved
    add.assert_called_once()


def test_duplicate_lookup_covers_one_second_either_side_in_utc():
    # Given: a backdated time in Malaysia time

    # When: logging a treatment at that time
    _, _, lookup = _log(
        event_type="Note", notes="x", event_time="2026-11-04T08:10:00+08:00"
    )

    # Then: Nightscout is asked for that second's neighbours in UTC
    assert lookup.call_args.args[:2] == (
        "2026-11-04T00:09:59.000Z",
        "2026-11-04T00:10:01.000Z",
    )
