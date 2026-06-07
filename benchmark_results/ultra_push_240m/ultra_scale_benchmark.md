# Ultra-Scale Push Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Cython backend: True
- Production storage is used for all tables.
- Residual timings use the selected Cython auto kernel.
- CZT window timing includes residual spectrum generation plus local profile transform.

| case | mass MDa | n_fft | table MiB | build s | active | residual 1w s | residual parallel s | speedup | CZT window s | full profile 1 formula s | rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra_1024x_240k | 242.344 | 6751269 | 412.1 | 8.967 | 4.44e-06 | 0.000126 | 0.000215 | 0.59x | 0.234999 | 0.494773 | 0.00e+00 |
