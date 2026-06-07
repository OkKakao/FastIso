# Parallel Workers Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Measured kernel: residual spectrum generation only.
- Parallelization is over formula rows; single-formula requests usually do not benefit.
- Speedup is relative to the same selected kernel with `workers=1`.

| case | batch | R | n_fft | active | selected | 1 worker s | 2 worker s | 4 worker s | 1 worker speedup | 2 worker speedup | 4 worker speedup | max rel L2 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| huge_240k | 1 | 0 | 83349 | 0.03187 | cython_log_pruned_attn32_uintphase_threshold | 0.000315 | 0.000299 | 0.000308 | 1.000x | 1.052x | 1.024x | 0.00e+00 |
| huge_240k | 1 | 100000 | 83349 | 0.0007439 | cython_log_pruned_attn32 | 0.000054 | 0.000051 | 0.000054 | 1.000x | 1.049x | 0.998x | 0.00e+00 |
| huge_240k | 10 | 0 | 83349 | 0.03132 | cython_log_pruned_attn32_uintphase_threshold | 0.004387 | 0.002246 | 0.002553 | 1.000x | 1.953x | 1.719x | 0.00e+00 |
| huge_240k | 10 | 100000 | 83349 | 0.0007439 | cython_log_pruned_attn32 | 0.000121 | 0.000170 | 0.000150 | 1.000x | 0.712x | 0.804x | 0.00e+00 |
| huge_240k | 50 | 0 | 84035 | 0.02918 | cython_log_pruned_attn32_uintphase_threshold | 0.018882 | 0.010117 | 0.005776 | 1.000x | 1.866x | 3.269x | 0.00e+00 |
| huge_240k | 50 | 100000 | 84035 | 0.0007159 | cython_log_pruned_attn32 | 0.000436 | 0.000332 | 0.000335 | 1.000x | 1.314x | 1.302x | 0.00e+00 |
| huge_240k | 100 | 0 | 91125 | 0.02685 | cython_log_pruned_attn32_uintphase_threshold | 0.039546 | 0.024288 | 0.013791 | 1.000x | 1.628x | 2.867x | 0.00e+00 |
| huge_240k | 100 | 100000 | 91125 | 0.0007021 | cython_log_pruned_attn32 | 0.000806 | 0.000599 | 0.000701 | 1.000x | 1.346x | 1.149x | 0.00e+00 |
| huge_60k | 1 | 0 | 42525 | 0.4227 | cython_log_pruned_attn32 | 0.000767 | 0.000540 | 0.000519 | 1.000x | 1.420x | 1.478x | 0.00e+00 |
| huge_60k | 1 | 100000 | 42525 | 0.005691 | cython_log_pruned_attn32 | 0.000053 | 0.000053 | 0.000051 | 1.000x | 0.994x | 1.027x | 0.00e+00 |
| huge_60k | 10 | 0 | 42525 | 0.4183 | cython_log_pruned_attn32 | 0.005646 | 0.002990 | 0.001910 | 1.000x | 1.888x | 2.955x | 0.00e+00 |
| huge_60k | 10 | 100000 | 42525 | 0.005526 | cython_log_pruned_attn32 | 0.000172 | 0.000147 | 0.000133 | 1.000x | 1.176x | 1.295x | 0.00e+00 |
| huge_60k | 50 | 0 | 45927 | 0.3984 | cython_log_pruned_attn32 | 0.039739 | 0.018243 | 0.011764 | 1.000x | 2.178x | 3.378x | 0.00e+00 |
| huge_60k | 50 | 100000 | 45927 | 0.005157 | cython_log_pruned_attn32 | 0.000769 | 0.000508 | 0.000469 | 1.000x | 1.514x | 1.640x | 0.00e+00 |
| huge_60k | 100 | 0 | 50421 | 0.3733 | cython_log_pruned_attn32 | 0.065917 | 0.036476 | 0.023744 | 1.000x | 1.807x | 2.776x | 0.00e+00 |
| huge_60k | 100 | 100000 | 50421 | 0.004717 | cython_log_pruned_attn32 | 0.001457 | 0.000901 | 0.000626 | 1.000x | 1.618x | 2.329x | 0.00e+00 |
| large_12k | 1 | 0 | 32805 | 1 | cython_log_pruned_attn32 | 0.001256 | 0.000973 | 0.000942 | 1.000x | 1.291x | 1.334x | 0.00e+00 |
| large_12k | 1 | 100000 | 32805 | 0.07907 | cython_log_pruned_attn32 | 0.000111 | 0.000098 | 0.000098 | 1.000x | 1.130x | 1.132x | 0.00e+00 |
| large_12k | 10 | 0 | 32805 | 1 | cython_log_pruned_attn32 | 0.008960 | 0.010175 | 0.006527 | 1.000x | 0.881x | 1.373x | 0.00e+00 |
| large_12k | 10 | 100000 | 32805 | 0.07566 | cython_log_pruned_attn32 | 0.000793 | 0.000462 | 0.000437 | 1.000x | 1.716x | 1.813x | 0.00e+00 |
| large_12k | 50 | 0 | 32805 | 1 | cython_log_pruned_attn32 | 0.047058 | 0.021975 | 0.013171 | 1.000x | 2.141x | 3.573x | 0.00e+00 |
| large_12k | 50 | 100000 | 32805 | 0.06254 | cython_log_pruned_attn32 | 0.003091 | 0.001799 | 0.001115 | 1.000x | 1.718x | 2.772x | 0.00e+00 |
| large_12k | 100 | 0 | 35721 | 0.9958 | cython_log_pruned_attn32 | 0.102704 | 0.047932 | 0.031260 | 1.000x | 2.143x | 3.285x | 0.00e+00 |
| large_12k | 100 | 100000 | 35721 | 0.05075 | cython_log_pruned_attn32 | 0.006453 | 0.004409 | 0.003795 | 1.000x | 1.464x | 1.700x | 0.00e+00 |
