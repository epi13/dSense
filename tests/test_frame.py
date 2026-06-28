from dsense.frame import INT32_MAX, INT32_MIN, build_frame, parse_frame, verify_frame, FRAME_SIZE, frame_to_dict


def test_frame_size_round_trip():
    frame = build_frame(7, 123456789, 0b111, 0b010, 100, -5, 321)
    assert len(frame) == FRAME_SIZE
    parsed = parse_frame(frame)
    assert parsed.sequence == 7
    assert parsed.t_ns == 123456789
    assert parsed.dt_ns == 100
    assert parsed.sleep_drift_ns == -5
    assert parsed.process_ns_estimate == 321
    assert frame_to_dict(frame)["checksum_ok"] is True


def test_checksum_catches_mutation():
    data = bytearray(build_frame(1, 2, 3, 4, 5, 6, 7))
    assert verify_frame(bytes(data))
    data[30] ^= 0xFF
    assert not verify_frame(bytes(data))


def test_frame_raw_fields_clamp_int32_overflow():
    frame = build_frame(1, 2, 3, 4, INT32_MAX + 1000, INT32_MIN - 1000, 9_999_999_999)
    parsed = parse_frame(frame)

    assert parsed.dt_ns == INT32_MAX
    assert parsed.sleep_drift_ns == INT32_MIN
    assert parsed.process_ns_estimate == INT32_MAX
    assert verify_frame(frame)
