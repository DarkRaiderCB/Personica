from personica.time_context import build_time_system_message


def test_valid_timezone():
    msg = build_time_system_message("Asia/Kolkata")
    assert "Asia/Kolkata" in msg
    assert "Current local time" in msg


def test_invalid_timezone_falls_back_to_utc():
    msg = build_time_system_message("Not/AZone")
    assert "UTC" in msg
    assert "Current local time" in msg
