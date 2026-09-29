from datetime import UTC, datetime

import pytest

from price_alert.formatting import beijing_time, describe_window, format_price


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (208.1305585, 2, "208.13"),
        (101.0, 2, "101.00"),
        (112345.63, 1, "112345.6"),
        (112345.6, None, "112345.6"),
        (112345.63333, None, "112345.63"),
        (0.000012345678, None, "0.000012345678"),
        (75800.0, None, "75800"),
        (0.5, None, "0.5"),
        (0.0, None, "0"),
    ],
)
def test_format_price(value, decimals, expected):
    assert format_price(value, decimals) == expected


@pytest.mark.parametrize(("seconds", "expected"), [(30, "30秒"), (90, "90秒"), (60, "1分钟"), (180, "3分钟")])
def test_describe_window(seconds, expected):
    assert describe_window(seconds) == expected


def test_beijing_time_converts_from_utc():
    assert beijing_time(datetime(2026, 9, 29, 1, 27, tzinfo=UTC)) == "2026-09-29 09:27:00"
