"""Operating-mode frame averaging with the load-independent / load-dependent split.

This is the abstraction shared by the digital, analog and PA components. In the
original scripts the same algebra was copy-pasted (and re-derived by hand) for
each component; here it lives in one place.

Within a frame a component is, for a fraction of the time, in one of four modes
(eqs 2.23 / 2.26 / 2.33):

    * data transmission/reception   (fraction xbar * tau * (1 - tau_sig))
    * reference signalling          (fraction      tau * tau_sig)
    * micro-sleep / DTX             (fraction (1 - xbar) * tau * (1 - tau_sig))
    * idle                          (fraction          (1 - tau))

The micro-sleep and idle modes consume a reduced power, scaled by the reduction
factors ``delta_micro`` and ``delta_idle`` relative to a base level.

Only the *data* term scales with the load ``xbar`` while the micro-sleep term
scales with ``(1 - xbar)``. Collecting the terms proportional to ``xbar`` gives
the load-dependent part; the remainder is load-independent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoadSplit:
    """A power value split into load-independent and load-dependent parts [W]."""

    load_ind: float
    load_dep: float

    @property
    def total(self) -> float:
        return self.load_ind + self.load_dep

    def __add__(self, other: "LoadSplit") -> "LoadSplit":
        return LoadSplit(self.load_ind + other.load_ind,
                         self.load_dep + other.load_dep)

    def scaled(self, factor: float) -> "LoadSplit":
        """Return a copy with both parts multiplied by ``factor``."""
        return LoadSplit(self.load_ind * factor, self.load_dep * factor)


def frame_average(tau, tau_sig, xbar, P_active, P_signaling, P_sleep_base,
                  delta_micro, delta_idle) -> LoadSplit:
    """Average a component over the operating modes of one link direction.

    Args:
        tau: Direction duration / frame duration (``tau_DL`` or ``tau_UL``).
        tau_sig: Signalling duration / direction duration.
        xbar: Average physical resource load of this direction, in [0, 1].
            Passed in as the *active-subcarrier ratio* (eq 1 of the paper), but
            used here as the *active-resource time ratio* N_a / N that weights
            the data mode. The two are equal only at full spatial load,
            K = K_max; using xbar as the data-mode weight is therefore exact for
            K = K_max and approximate otherwise. See ``OperatingPoint``.
        P_active: Consumption during data transmission/reception [W].
        P_signaling: Consumption during reference signalling [W]. Equals
            ``P_active`` for digital/analog; the PA uses the signalling
            transmit power instead.
        P_sleep_base: Base consumption that the reduction factors scale, i.e.
            the level reached in micro-sleep/idle before the factor is applied
            [W]. Equals ``P_active`` for digital/analog; the PA uses its
            zero-output consumption.
        delta_micro: Micro-sleep reduction factor in [0, 1].
        delta_idle: Idle reduction factor in [0, 1].

    Returns:
        The frame-averaged consumption split into load-independent and
        load-dependent parts.
    """
    data = xbar * tau * (1 - tau_sig) * P_active
    signalling = tau * tau_sig * P_signaling
    micro_sleep = tau * (1 - xbar) * (1 - tau_sig) * P_sleep_base * delta_micro
    idle = (1 - tau) * P_sleep_base * delta_idle

    total = data + signalling + micro_sleep + idle
    # Terms proportional to xbar: the data term, minus the xbar part of the
    # micro-sleep term (which carries a (1 - xbar) factor).
    load_dep = xbar * tau * (1 - tau_sig) * (P_active - P_sleep_base * delta_micro)
    return LoadSplit(total - load_dep, load_dep)
