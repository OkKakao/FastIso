# Ultra-Scale Push Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Cython backend: True
- Production storage is used for all tables.
- Residual timings use the selected Cython auto kernel.
- CZT window timing includes residual spectrum generation plus local profile transform.

| case | mass MDa | n_fft | table MiB | build s | active | residual 1w s | residual parallel s | speedup | CZT window s | full profile 1 formula s | rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra_256x_240k | 60.586 | 2066715 | 126.1 | 1.480 | 1.45e-05 | 0.000109 | 0.000141 | 0.77x | 0.077571 | 0.098570 | 0.00e+00 |
| ultra_512x_240k | 121.172 | 3720087 | 227.1 | 3.530 | 8.06e-06 | 0.000106 | 0.000133 | 0.80x | 0.112814 | 0.191419 | 0.00e+00 |
