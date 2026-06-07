# Ultra-Scale Push Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Cython backend: True
- Production storage is used for all tables.
- Residual timings use the selected Cython auto kernel.
- CZT window timing includes residual spectrum generation plus local profile transform.

| case | mass MDa | n_fft | table MiB | build s | active | residual 1w s | residual parallel s | speedup | CZT window s | full profile 1 formula s | rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra_2x_240k | 0.473 | 117649 | 7.2 | 0.124 | 0.000255 | 0.000101 | 0.000614 | 0.16x | 0.004115 | 0.006739 | 0.00e+00 |
| ultra_4x_240k | 0.947 | 165375 | 10.1 | 0.177 | 0.000181 | 0.000107 | 0.000122 | 0.88x | 0.004856 | 0.005902 | 0.00e+00 |
| ultra_8x_240k | 1.893 | 250047 | 15.3 | 0.171 | 0.000128 | 0.000103 | 0.000104 | 0.98x | 0.007783 | 0.010520 | 0.00e+00 |
| ultra_16x_240k | 3.787 | 352947 | 21.5 | 0.316 | 8.5e-05 | 0.000132 | 0.000134 | 0.99x | 0.010354 | 0.018323 | 0.00e+00 |
| ultra_32x_240k | 7.573 | 531441 | 32.4 | 0.388 | 6.02e-05 | 0.000113 | 0.000136 | 0.83x | 0.020510 | 0.021186 | 0.00e+00 |
| ultra_64x_240k | 15.146 | 759375 | 46.3 | 0.619 | 3.95e-05 | 0.000114 | 0.000170 | 0.67x | 0.024372 | 0.027218 | 0.00e+00 |
| ultra_128x_240k | 30.293 | 1240029 | 75.7 | 0.841 | 2.42e-05 | 0.000106 | 0.000116 | 0.91x | 0.036080 | 0.057897 | 0.00e+00 |
