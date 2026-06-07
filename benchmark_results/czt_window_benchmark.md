# CZT Window Benchmark

- Runtime includes residual spectrum generation plus profile transform.
- `requested_method` is the public method passed to FastIso; `spectrum_method` is the selected internal spectrum kernel.
- Transform-only columns assume the residual spectrum is already computed.
- `full_profile_s` computes the complete dense profile.
- `czt_window_s` computes only the requested residual mass window.
- Grid-matched rows report relative L2 against the same full-profile samples.

| case | method | n_fft | active | window Da | output dm | points | full profile s | czt window s | full/czt | xform full/czt | rel L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.1 | 0.002 | 51 | 0.000734 | 0.001136 | 0.65x | 0.52x | 1.57e-11
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.1 | 0.001 | 101 | 0.000734 | 0.001261 | 0.58x | 0.50x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.1 | 0.0005 | 201 | 0.000734 | 0.001017 | 0.72x | 0.59x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.2 | 0.002 | 101 | 0.000734 | 0.001016 | 0.72x | 0.59x | 3.55e-12
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.2 | 0.001 | 201 | 0.000734 | 0.001114 | 0.66x | 0.50x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.2 | 0.0005 | 401 | 0.000734 | 0.001195 | 0.61x | 0.49x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.5 | 0.002 | 251 | 0.000734 | 0.000964 | 0.76x | 0.48x | 7.30e-13
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.5 | 0.001 | 501 | 0.000734 | 0.001069 | 0.69x | 0.51x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 0.5 | 0.0005 | 1001 | 0.000734 | 0.001008 | 0.73x | 0.51x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 1 | 0.002 | 501 | 0.000734 | 0.001085 | 0.68x | 0.57x | 1.13e-12
large_12k | cython_log_pruned | 32805 | 0.07907 | 1 | 0.001 | 1001 | 0.000734 | 0.001027 | 0.71x | 0.58x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 1 | 0.0005 | 2001 | 0.000734 | 0.001059 | 0.69x | 0.52x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 2 | 0.002 | 1001 | 0.000734 | 0.000963 | 0.76x | 0.60x | 1.35e-12
large_12k | cython_log_pruned | 32805 | 0.07907 | 2 | 0.001 | 2001 | 0.000734 | 0.001021 | 0.72x | 0.51x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 2 | 0.0005 | 4001 | 0.000734 | 0.001447 | 0.51x | 0.36x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 5 | 0.002 | 2501 | 0.000734 | 0.001131 | 0.65x | 0.46x | 5.92e-12
large_12k | cython_log_pruned | 32805 | 0.07907 | 5 | 0.001 | 5001 | 0.000734 | 0.001639 | 0.45x | 0.31x | NA
large_12k | cython_log_pruned | 32805 | 0.07907 | 5 | 0.0005 | 10001 | 0.000734 | 0.002637 | 0.28x | 0.17x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.1 | 0.002 | 51 | 0.000732 | 0.000786 | 0.93x | 0.88x | 3.30e-13
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.1 | 0.001 | 101 | 0.000732 | 0.000729 | 1.01x | 0.78x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.1 | 0.0005 | 201 | 0.000732 | 0.000611 | 1.20x | 0.91x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.2 | 0.002 | 101 | 0.000732 | 0.000750 | 0.98x | 0.96x | 3.74e-13
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.2 | 0.001 | 201 | 0.000732 | 0.000903 | 0.81x | 0.84x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.2 | 0.0005 | 401 | 0.000732 | 0.000594 | 1.23x | 0.87x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.5 | 0.002 | 251 | 0.000732 | 0.000588 | 1.25x | 0.96x | 2.27e-13
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.5 | 0.001 | 501 | 0.000732 | 0.000701 | 1.05x | 0.96x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 0.5 | 0.0005 | 1001 | 0.000732 | 0.000694 | 1.06x | 0.84x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 1 | 0.002 | 501 | 0.000732 | 0.000622 | 1.18x | 0.95x | 4.83e-13
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 1 | 0.001 | 1001 | 0.000732 | 0.000742 | 0.99x | 0.82x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 1 | 0.0005 | 2001 | 0.000732 | 0.000921 | 0.80x | 0.56x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 2 | 0.002 | 1001 | 0.000732 | 0.000718 | 1.02x | 0.79x | 6.56e-13
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 2 | 0.001 | 2001 | 0.000732 | 0.000917 | 0.80x | 0.57x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 2 | 0.0005 | 4001 | 0.000732 | 0.001225 | 0.60x | 0.41x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 5 | 0.002 | 2501 | 0.000732 | 0.000954 | 0.77x | 0.54x | 2.68e-12
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 5 | 0.001 | 5001 | 0.000732 | 0.001397 | 0.52x | 0.35x | NA
xlarge_29k | cython_log_pruned | 32805 | 0.02018 | 5 | 0.0005 | 10001 | 0.000732 | 0.002447 | 0.30x | 0.19x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.1 | 0.002 | 51 | 0.000795 | 0.000438 | 1.82x | 1.46x | 4.71e-14
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.1 | 0.001 | 101 | 0.000795 | 0.000435 | 1.83x | 1.10x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.1 | 0.0005 | 201 | 0.000795 | 0.000662 | 1.20x | 1.20x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.2 | 0.002 | 101 | 0.000795 | 0.000550 | 1.44x | 1.36x | 1.03e-13
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.2 | 0.001 | 201 | 0.000795 | 0.000582 | 1.37x | 1.08x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.2 | 0.0005 | 401 | 0.000795 | 0.000565 | 1.41x | 1.26x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.5 | 0.002 | 251 | 0.000795 | 0.000621 | 1.28x | 1.22x | 1.27e-13
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.5 | 0.001 | 501 | 0.000795 | 0.000640 | 1.24x | 1.16x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 0.5 | 0.0005 | 1001 | 0.000795 | 0.000755 | 1.05x | 0.96x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 1 | 0.002 | 501 | 0.000795 | 0.000706 | 1.13x | 1.21x | 1.69e-13
huge_60k | cython_log_pruned | 42525 | 0.005691 | 1 | 0.001 | 1001 | 0.000795 | 0.000735 | 1.08x | 0.82x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 1 | 0.0005 | 2001 | 0.000795 | 0.001032 | 0.77x | 0.62x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 2 | 0.002 | 1001 | 0.000795 | 0.000942 | 0.84x | 0.92x | 3.33e-13
huge_60k | cython_log_pruned | 42525 | 0.005691 | 2 | 0.001 | 2001 | 0.000795 | 0.000966 | 0.82x | 0.53x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 2 | 0.0005 | 4001 | 0.000795 | 0.001448 | 0.55x | 0.41x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 5 | 0.002 | 2501 | 0.000795 | 0.001017 | 0.78x | 0.58x | 1.13e-12
huge_60k | cython_log_pruned | 42525 | 0.005691 | 5 | 0.001 | 5001 | 0.000795 | 0.002002 | 0.40x | 0.34x | NA
huge_60k | cython_log_pruned | 42525 | 0.005691 | 5 | 0.0005 | 10001 | 0.000795 | 0.003405 | 0.23x | 0.19x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.1 | 0.002 | 51 | 0.000942 | 0.000443 | 2.13x | 2.01x | 2.05e-15
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.1 | 0.001 | 101 | 0.000942 | 0.000441 | 2.14x | 1.99x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.1 | 0.0005 | 201 | 0.000942 | 0.000444 | 2.12x | 1.99x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.2 | 0.002 | 101 | 0.000942 | 0.000476 | 1.98x | 1.95x | 3.62e-15
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.2 | 0.001 | 201 | 0.000942 | 0.000475 | 1.99x | 1.72x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.2 | 0.0005 | 401 | 0.000942 | 0.000499 | 1.89x | 1.77x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.5 | 0.002 | 251 | 0.000942 | 0.000545 | 1.73x | 2.02x | 7.64e-15
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.5 | 0.001 | 501 | 0.000942 | 0.000526 | 1.79x | 1.72x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 0.5 | 0.0005 | 1001 | 0.000942 | 0.000629 | 1.50x | 1.28x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 1 | 0.002 | 501 | 0.000942 | 0.000536 | 1.76x | 1.69x | 1.06e-14
huge_120k | cython_log_pruned | 59049 | 0.001998 | 1 | 0.001 | 1001 | 0.000942 | 0.000667 | 1.41x | 1.36x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 1 | 0.0005 | 2001 | 0.000942 | 0.000816 | 1.15x | 0.69x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 2 | 0.002 | 1001 | 0.000942 | 0.000671 | 1.40x | 1.06x | 2.19e-14
huge_120k | cython_log_pruned | 59049 | 0.001998 | 2 | 0.001 | 2001 | 0.000942 | 0.000937 | 1.01x | 0.78x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 2 | 0.0005 | 4001 | 0.000942 | 0.001406 | 0.67x | 0.53x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 5 | 0.002 | 2501 | 0.000942 | 0.001025 | 0.92x | 0.82x | 6.41e-14
huge_120k | cython_log_pruned | 59049 | 0.001998 | 5 | 0.001 | 5001 | 0.000942 | 0.001423 | 0.66x | 0.47x | NA
huge_120k | cython_log_pruned | 59049 | 0.001998 | 5 | 0.0005 | 10001 | 0.000942 | 0.002739 | 0.34x | 0.25x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.1 | 0.002 | 51 | 0.003216 | 0.000487 | 6.60x | 6.14x | 1.59e-15
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.1 | 0.001 | 101 | 0.003216 | 0.000446 | 7.21x | 6.12x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.1 | 0.0005 | 201 | 0.003216 | 0.000477 | 6.74x | 5.95x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.2 | 0.002 | 101 | 0.003216 | 0.000459 | 7.01x | 6.35x | 2.76e-15
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.2 | 0.001 | 201 | 0.003216 | 0.000445 | 7.22x | 5.81x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.2 | 0.0005 | 401 | 0.003216 | 0.000521 | 6.17x | 5.53x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.5 | 0.002 | 251 | 0.003216 | 0.000493 | 6.52x | 6.31x | 6.39e-15
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.5 | 0.001 | 501 | 0.003216 | 0.000563 | 5.71x | 5.36x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 0.5 | 0.0005 | 1001 | 0.003216 | 0.000626 | 5.14x | 4.37x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 1 | 0.002 | 501 | 0.003216 | 0.000586 | 5.49x | 5.29x | 1.31e-14
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 1 | 0.001 | 1001 | 0.003216 | 0.000661 | 4.86x | 4.21x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 1 | 0.0005 | 2001 | 0.003216 | 0.001020 | 3.15x | 2.88x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 2 | 0.002 | 1001 | 0.003216 | 0.000799 | 4.02x | 3.40x | 2.61e-14
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 2 | 0.001 | 2001 | 0.003216 | 0.000989 | 3.25x | 2.23x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 2 | 0.0005 | 4001 | 0.003216 | 0.001601 | 2.01x | 1.78x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 5 | 0.002 | 2501 | 0.003216 | 0.001064 | 3.02x | 2.48x | 6.78e-14
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 5 | 0.001 | 5001 | 0.003216 | 0.001534 | 2.10x | 1.48x | NA
huge_240k | cython_log_pruned | 83349 | 0.0007439 | 5 | 0.0005 | 10001 | 0.003216 | 0.002885 | 1.11x | 0.66x | NA
