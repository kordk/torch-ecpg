import pytest
import os
import torch
import logging

from tecpg.processing import qr_peak_bytes, check_chunk_headroom, _cuda_free_bytes


class MockLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, msg):
        self.info_messages.append(msg)

    def warning(self, msg):
        self.warning_messages.append(msg)


def test_qr_peak_bytes_matches_closed_form():
    mt_count = 135028
    nrows = 1185
    ncols = 13
    datum_bytes = 4
    expected = 18026778112
    assert qr_peak_bytes(mt_count, nrows, ncols, datum_bytes) == expected


def test_qr_peak_bytes_scales_linearly_in_mt():
    m = 1000
    nrows = 1185
    ncols = 13
    datum_bytes = 4
    val1 = qr_peak_bytes(m, nrows, ncols, datum_bytes)
    val2 = qr_peak_bytes(2 * m, nrows, ncols, datum_bytes)
    assert val2 == 2 * val1


def test_qr_peak_bytes_short_final_chunk():
    mt_count = 80493
    nrows = 1185
    ncols = 13
    datum_bytes = 4
    expected = 10746137472

    full_mt_count = 135028
    val1 = qr_peak_bytes(full_mt_count, nrows, ncols, datum_bytes)
    val2 = qr_peak_bytes(mt_count, nrows, ncols, datum_bytes)

    assert val2 == expected
    assert val2 < val1


def test_headroom_guard_raises_on_deficit():
    mt_count = 100
    nrows = 50
    ncols = 10
    datum_bytes = 4
    required = qr_peak_bytes(mt_count, nrows, ncols, datum_bytes)

    def mock_free_bytes(device):
        return required - 1

    device = type('MockDevice', (), {'type': 'cuda'})()
    logger = MockLogger()

    with pytest.raises(RuntimeError) as exc_info:
        check_chunk_headroom(
            mt_count, nrows, ncols, datum_bytes, device, 0, 1, logger, mock_free_bytes
        )

    msg = str(exc_info.value)
    assert msg.startswith("tecpg: insufficient GPU memory for methylation chunk")
    assert f"deficit=1" in msg
    assert f"mt_count={mt_count}" in msg


def test_headroom_guard_silent_when_ample():
    mt_count = 100
    nrows = 50
    ncols = 10
    datum_bytes = 4
    required = qr_peak_bytes(mt_count, nrows, ncols, datum_bytes)

    def mock_free_bytes(device):
        return 10 * required

    device = type('MockDevice', (), {'type': 'cuda'})()
    logger = MockLogger()

    # Should not raise
    check_chunk_headroom(
        mt_count, nrows, ncols, datum_bytes, device, 0, 1, logger, mock_free_bytes
    )


def test_headroom_guard_noop_on_non_cuda_device():
    mt_count = 100
    nrows = 50
    ncols = 10
    datum_bytes = 4

    device = type('MockDevice', (), {'type': 'cpu'})()
    logger = MockLogger()

    # _cuda_free_bytes returns None on cpu
    # The default free_bytes_fn is _cuda_free_bytes, so we pass it explicitly here for completeness
    # to mock free_bytes_fn returning None
    def mock_free_bytes(dev):
        return _cuda_free_bytes(dev)

    # Should not raise or log
    check_chunk_headroom(
        mt_count, nrows, ncols, datum_bytes, device, 0, 1, logger, mock_free_bytes
    )

    assert len(logger.info_messages) == 0
    assert len(logger.warning_messages) == 0


def test_headroom_guard_warn_only_env_bypasses_raise(monkeypatch):
    monkeypatch.setenv("TECPG_HEADROOM_WARN_ONLY", "1")

    mt_count = 100
    nrows = 50
    ncols = 10
    datum_bytes = 4
    required = qr_peak_bytes(mt_count, nrows, ncols, datum_bytes)

    def mock_free_bytes(device):
        return required - 1

    device = type('MockDevice', (), {'type': 'cuda'})()
    logger = MockLogger()

    # Should not raise because of TECPG_HEADROOM_WARN_ONLY=1
    check_chunk_headroom(
        mt_count, nrows, ncols, datum_bytes, device, 0, 1, logger, mock_free_bytes
    )

    assert len(logger.warning_messages) == 1
    assert "tecpg: insufficient GPU memory for methylation chunk" in logger.warning_messages[0]


def test_headroom_guard_logs_on_every_boundary():
    mt_count = 100
    nrows = 50
    ncols = 10
    datum_bytes = 4
    required = qr_peak_bytes(mt_count, nrows, ncols, datum_bytes)
    free = 10 * required

    def mock_free_bytes(device):
        return free

    device = type('MockDevice', (), {'type': 'cuda'})()
    logger = MockLogger()

    # Should not raise
    check_chunk_headroom(
        mt_count, nrows, ncols, datum_bytes, device, 0, 1, logger, mock_free_bytes
    )

    assert len(logger.info_messages) == 1

    msg = logger.info_messages[0]
    free_mb = free / (1024 * 1024)
    required_mb = required / (1024 * 1024)
    headroom_mb = free_mb - required_mb

    assert f"mt_count={mt_count}" in msg
    assert f"headroom={headroom_mb:.2f}MB" in msg
