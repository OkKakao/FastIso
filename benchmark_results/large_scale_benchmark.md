# Large-Scale Benchmark

- Cython backend: True
- Table construction time is excluded.
- Runtime is residual spectrum generation, not final irFFT profile generation.

| case | auto | n_fft | window Da | mean mass Da | active | log_full s | cython_pruned s | cython/full | rel L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| large_12k | True | 32805 | 65.6 | 12083.1 | 0.0792 | 0.024481 | 0.001978 | 12.37x | 1.03e-12 |
| xlarge_29k | True | 32805 | 65.6 | 28811.5 | 0.0202 | 0.022522 | 0.001472 | 15.30x | 1.39e-12 |
| huge_60k | True | 42525 | 85.0 | 59165.7 | 0.0057 | 0.042339 | 0.001756 | 24.11x | 6.74e-13 |
| huge_120k | True | 59049 | 118.1 | 118331.3 | 0.0020 | 0.062832 | 0.002244 | 28.00x | 7.49e-13 |
| huge_240k | True | 83349 | 166.7 | 236662.7 | 0.0007 | 0.060715 | 0.003376 | 17.98x | 5.25e-13 |
