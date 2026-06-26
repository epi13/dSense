from dsense.frame import build_frame, FRAME_SIZE
from dsense.quality import summarize_frames


def test_quality_summary(tmp_path):
    p = tmp_path / "frames.ds64"
    frames = [build_frame(i, i * 10_000_000, 7, 0, 10_000_000 if i else 0, 0, 100) for i in range(5)]
    p.write_bytes(b"".join(frames))
    q = summarize_frames(p, expected_frames=5, target_interval_ns=10_000_000)
    assert q.actual_frames == 5
    assert q.frame_size_valid
    assert q.checksum_ok
    assert p.stat().st_size % FRAME_SIZE == 0
