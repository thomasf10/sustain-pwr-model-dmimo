"""DFT codebooks and subarray beam selection for hybrid beamforming.

Used by the rate model to build the analog beamformer of a hybrid
partially-connected architecture. Pure NumPy (no Sionna), so it is unit-testable
on its own.
"""

from __future__ import annotations

import numpy as np


def hermitian(A: np.ndarray) -> np.ndarray:
    """Conjugate (Hermitian) transpose of ``A``."""
    return np.conjugate(A.T)


def gen_codebook_1d(m: int) -> np.ndarray:
    """One-dimensional DFT codebook of ``m`` beams (columns are beams)."""
    m_range = np.arange(m)
    W = np.exp(-1j * 2 * np.pi * np.outer(m_range, m_range) / m)
    return W / np.sqrt(m)


def gen_codebook_2d(m_hor: int, m_ver: int) -> np.ndarray:
    """Two-dimensional DFT codebook for a uniform planar array.

    Returns an array of shape ``(m_hor * m_ver, m_hor * m_ver)`` whose rows are
    the Kronecker products of horizontal and vertical beams.
    """
    W_ver = gen_codebook_1d(m_ver)
    W_hor = gen_codebook_1d(m_hor)
    codebook = []
    for m in range(m_ver):
        for n in range(m_hor):
            codebook.append(np.kron(W_hor[:, n], W_ver[:, m]))
    return np.array(codebook)


def _subarray_shape(m_hor: int, m_ver: int, m_rf: int):
    """Horizontal/vertical size of each subarray for ``m_rf`` RF chains."""
    ratio = np.sqrt(m_hor * m_ver / m_rf)
    if ratio - np.round(ratio) == 0:
        size = int(ratio)
        return size, size
    if np.random.uniform(0, 1) > 0.5:
        m_s_hor = int(2 ** np.ceil(np.log2(ratio)))
        m_s_ver = int(np.round(m_hor * m_ver / m_rf) / m_s_hor)
    else:
        m_s_ver = int(2 ** np.ceil(np.log2(ratio)))
        m_s_hor = int(np.round(m_hor * m_ver / m_rf) / m_s_ver)
    return m_s_hor, m_s_ver


def subarray_beam_sel(m_hor: int, m_ver: int, m_rf: int, hmat: np.ndarray,
                      direction: str) -> np.ndarray:
    """Select the best DFT beam per subarray (hybrid partially-connected).

    Args:
        m_hor, m_ver: Horizontal/vertical antennas of the planar array.
        m_rf: Number of RF chains (= number of subarrays).
        hmat: Channel matrix, ``K x M`` in downlink or ``M x K`` in uplink.
        direction: ``"downlink"`` or ``"uplink"``.

    Returns:
        The analog beamforming matrix ``W_RF`` (``M x m_rf`` in downlink,
        ``m_rf x M`` in uplink).
    """
    m_s_hor, m_s_ver = _subarray_shape(m_hor, m_ver, m_rf)
    codebook = gen_codebook_2d(m_s_hor, m_s_ver)
    index_map = np.reshape(np.arange(m_hor * m_ver), [m_hor, m_ver]).T

    if direction == "downlink":
        W_RF = np.zeros((m_hor * m_ver, m_rf), dtype=complex)
    else:
        W_RF = np.zeros((m_rf, m_hor * m_ver), dtype=complex)

    chain = 0
    for i in range(int(m_hor / m_s_hor)):
        for j in range(int(m_ver / m_s_ver)):
            idx = index_map[j * m_s_ver:(j + 1) * m_s_ver,
                            i * m_s_hor:(i + 1) * m_s_hor].T.reshape(-1)
            if direction == "downlink":
                H_sub = hmat[:, idx]
                best = np.argmax([np.mean(np.abs(H_sub @ codebook[b, :]) ** 2)
                                  for b in range(codebook.shape[0])])
                W_RF[idx, chain] = codebook[best, :]
            else:
                H_sub = hmat[idx, :]
                best = np.argmax([np.mean(np.abs(codebook[b, :] @ H_sub) ** 2)
                                  for b in range(codebook.shape[0])])
                W_RF[chain, idx] = codebook[best, :]
            chain += 1
    return W_RF
