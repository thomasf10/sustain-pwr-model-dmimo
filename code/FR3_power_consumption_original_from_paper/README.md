# FR3 power consumption model

<div align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/11464669"><img src="https://img.shields.io/badge/Xplore-%2300629B?style=flat&logo=IEEE" alt="IEEE Xplore"></a>
  &nbsp;
</div>
<br>

This repository contains the Python source code to generate Figure 2a, Figure 2b, and Figure 2c of the manuscript [1].
<br>If you use this code in your work, please cite our paper as in [1].

The requested Python libraries are: numpy, matplotlib, and sionna (no ray tracing).

Each file is structured in: (i) definitions of functions, (ii) initialization of parameters, (iii) loop over one or more variables, and (iv) plot of results.

In Fig_2a.py, the power consumption of digital processing and analog processing is computed by fixing the number of base station antennas to 1024 and changing the number of radio-frequency chains from 16 to 1024. The consumption of power amplifier (not plotted in this file) is constant as it depends only on the number of antennas.
In Fig_2b.py, the power amplifier consumption is plotted for different number of base station antennas, from 16 to 1024.
In Fig_2c.py, the ergodic sum rate in downlink and uplink is computed by iterating over (a) number of radio-frequency chains, (b) number of channel realizations where every realization is a tensor containing the channel at every antenna, user, and subcarrier, and (c) number of subcarriers. The computationally heaviest part of the code is the channel generation via sionna (gen_channel() function); using GPUs can speed up its computation time.

[1] E. Peschiera, S. Yun, Y. Lee, L. V. der Perre and F. Rottenberg, "A parametric power model of upper mid-band (FR3) base stations for 6G," in 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2026, pp. 21476-21480.


