# Ultra-Scale Push Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Cython backend: True
- Production storage is used for all tables.
- Residual timings use the selected Cython auto kernel.
- CZT window timing includes residual spectrum generation plus local profile transform.

| case | mass MDa | n_fft | table MiB | build s | active | residual 1w s | residual parallel s | speedup | CZT window s | full profile 1 formula s | rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra_8192x_240k | 1938.748 | 51667875 | 3153.6 | 90.311 | 5.81e-07 | 0.020864 | 0.000156 | 133.92x | 0.218482 | NA | 0.00e+00 |
