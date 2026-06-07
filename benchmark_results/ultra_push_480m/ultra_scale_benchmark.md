# Ultra-Scale Push Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Cython backend: True
- Production storage is used for all tables.
- Residual timings use the selected Cython auto kernel.
- CZT window timing includes residual spectrum generation plus local profile transform.

| case | mass MDa | n_fft | table MiB | build s | active | residual 1w s | residual parallel s | speedup | CZT window s | full profile 1 formula s | rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra_2048x_240k | 484.687 | 13286025 | 810.9 | 8.362 | 2.26e-06 | 0.000130 | 0.006546 | 0.02x | 0.403336 | 0.676529 | 0.00e+00 |
