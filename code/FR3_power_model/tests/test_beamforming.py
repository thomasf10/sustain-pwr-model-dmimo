"""Sanity checks for the NumPy-only beamforming helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr3_power.beamforming import (  # noqa: E402
    gen_codebook_1d,
    gen_codebook_2d,
    hermitian,
    subarray_beam_sel,
)


def test_codebook_1d_unitary():
    W = gen_codebook_1d(8)
    assert np.allclose(hermitian(W) @ W, np.eye(8))


def test_codebook_2d_shape_and_unit_norm():
    cb = gen_codebook_2d(4, 2)
    assert cb.shape == (8, 8)
    assert np.allclose(np.linalg.norm(cb, axis=1), 1.0)


def test_subarray_beam_sel_downlink_shape():
    np.random.seed(0)
    m_hor, m_ver, m_rf, K = 8, 8, 16, 4
    H = (np.random.randn(K, m_hor * m_ver)
         + 1j * np.random.randn(K, m_hor * m_ver))
    W_RF = subarray_beam_sel(m_hor, m_ver, m_rf, H, "downlink")
    assert W_RF.shape == (m_hor * m_ver, m_rf)
    # Each RF chain drives exactly one subarray (one block of nonzero entries).
    assert np.count_nonzero(W_RF) == m_hor * m_ver


def test_subarray_beam_sel_uplink_shape():
    np.random.seed(1)
    m_hor, m_ver, m_rf, K = 8, 8, 16, 4
    H = (np.random.randn(m_hor * m_ver, K)
         + 1j * np.random.randn(m_hor * m_ver, K))
    W_RF = subarray_beam_sel(m_hor, m_ver, m_rf, H, "uplink")
    assert W_RF.shape == (m_rf, m_hor * m_ver)


if __name__ == "__main__":
    test_codebook_1d_unitary()
    test_codebook_2d_shape_and_unit_norm()
    test_subarray_beam_sel_downlink_shape()
    test_subarray_beam_sel_uplink_shape()
    print("All beamforming checks passed.")
