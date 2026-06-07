# Phase Modulo Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Measured kernel: residual spectrum generation only.
- Ratio columns above 1 mean the alternative phase reduction is faster.

| case | batch | R | n_fft | active | max active phase rad | baseline s | fmod s | cycle s | uint s | threshold s | auto s | auto selected | baseline/fmod | baseline/cycle | baseline/uint | baseline/threshold | baseline/auto | auto rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huge_120k | 10 | 0 | 59049 | 0.2052 | 1.23e+05 | 0.006139 | 0.008683 | 0.008578 | 0.005810 | 0.005760 | 0.005507 | cython_log_pruned | 0.707x | 0.716x | 1.057x | 1.066x | 1.115x | 0.00e+00 |
| huge_120k | 10 | 100000 | 59049 | 0.001998 | 983 | 0.002617 | 0.002292 | 0.002437 | 0.002572 | 0.000557 | 0.000633 | cython_log_pruned_uintphase_threshold | 1.142x | 1.074x | 1.018x | 4.697x | 4.132x | 6.48e-13 |
| huge_240k | 10 | 0 | 83349 | 0.03132 | 1.87e+05 | 0.004782 | 0.004748 | 0.004438 | 0.004280 | 0.003828 | 0.004232 | cython_log_pruned_uintphase_threshold | 1.007x | 1.077x | 1.117x | 1.249x | 1.130x | 3.73e-12 |
| huge_240k | 10 | 100000 | 83349 | 0.0007439 | 975 | 0.003188 | 0.003898 | 0.003170 | 0.003877 | 0.000747 | 0.000855 | cython_log_pruned_uintphase_threshold | 0.818x | 1.006x | 0.822x | 4.270x | 3.730x | 2.22e-12 |
| huge_60k | 10 | 0 | 42525 | 0.4183 | 6.25e+04 | 0.005921 | 0.007591 | 0.005985 | 0.006316 | 0.007749 | 0.006433 | cython_log_pruned | 0.780x | 0.989x | 0.937x | 0.764x | 0.920x | 0.00e+00 |
| huge_60k | 10 | 100000 | 42525 | 0.005526 | 999 | 0.001874 | 0.001757 | 0.001906 | 0.001740 | 0.000465 | 0.000575 | cython_log_pruned_uintphase_threshold | 1.066x | 0.983x | 1.077x | 4.030x | 3.260x | 2.87e-13 |
| large_12k | 10 | 0 | 32805 | 1 | 1.41e+04 | 0.008992 | 0.011503 | 0.008909 | 0.008236 | 0.010253 | 0.010167 | cython_log_pruned | 0.782x | 1.009x | 1.092x | 0.877x | 0.884x | 0.00e+00 |
| large_12k | 10 | 100000 | 32805 | 0.07566 | 1.18e+03 | 0.002275 | 0.002171 | 0.002305 | 0.001843 | 0.001083 | 0.001133 | cython_log_pruned_uintphase_threshold | 1.048x | 0.987x | 1.234x | 2.100x | 2.008x | 1.99e-13 |
| xlarge_29k | 10 | 0 | 32805 | 0.7911 | 3.13e+04 | 0.006791 | 0.009911 | 0.007320 | 0.006647 | 0.008839 | 0.007159 | cython_log_pruned | 0.685x | 0.928x | 1.022x | 0.768x | 0.949x | 0.00e+00 |
| xlarge_29k | 10 | 100000 | 32805 | 0.01914 | 1.12e+03 | 0.001644 | 0.001583 | 0.002119 | 0.002206 | 0.000603 | 0.000620 | cython_log_pruned_uintphase_threshold | 1.039x | 0.776x | 0.745x | 2.727x | 2.654x | 4.03e-13 |
