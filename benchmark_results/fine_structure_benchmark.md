# Fine-Structure Benchmark

- IsoSpecPy is the fine-structure reference.
- brainpy is an aggregated isotope-distribution baseline.
- pyOpenMS uses OpenMS FineIsotopePatternGenerator when pyopenms is installed.
- enviPat uses the R package through Rscript when available.
- Runtime includes dense Gaussian profile generation on the same grid.
- External backend timing is split into peak-list generation and Gaussian convolution.
- Fine-structure comparisons need dm smaller than the instrument sigma.
- pyOpenMS uses individual-peak threshold mode, not total-coverage mode.
- Cython backend: True.
- brainpy available: True.
- pyOpenMS available: True.
- enviPat available: True.
- IsoSpecPy coverage: 0.999.
- brainpy requested peaks: 200.
- pyOpenMS threshold: 1e-06; absolute: False.
- enviPat threshold: 0.0001; rel_to: 0; algo: 1.

| case | formula | backend | n_fft | total s | peak s | convolve s | speed vs FastIso | peaks | rel L2 vs IsoSpec | local maxima | apex shift Da |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_glucose | C6H12O6 | fastiso | 32805 | 0.001215 | NA | NA | 1.00x | NA | 4.38e-04 | 3 | 0.0000e+00 |
| small_glucose | C6H12O6 | IsoSpecPy | 32805 | 0.000217 | 0.000076 | 0.000141 | 0.18x | 7 | 0.00e+00 | 3 | 0.0000e+00 |
| small_glucose | C6H12O6 | brainpy | 32805 | 0.000181 | 0.000048 | 0.000133 | 0.15x | 9 | 1.64e-02 | 3 | 0.0000e+00 |
| small_glucose | C6H12O6 | envipat | 32805 | 0.279677 | 0.279233 | 0.000445 | 230.11x | 21 | 6.11e-04 | 3 | 0.0000e+00 |
| small_glucose | C6H12O6 | pyopenms | 32805 | 0.000249 | 0.000041 | 0.000208 | 0.20x | 21 | 5.09e-03 | 3 | 0.0000e+00 |
| medium_2p4k | C100H160N25O40S2 | fastiso | 83349 | 0.003802 | NA | NA | 1.00x | NA | 1.11e-03 | 7 | 0.0000e+00 |
| medium_2p4k | C100H160N25O40S2 | IsoSpecPy | 83349 | 0.004423 | 0.000168 | 0.004255 | 1.16x | 216 | 0.00e+00 | 7 | 0.0000e+00 |
| medium_2p4k | C100H160N25O40S2 | brainpy | 83349 | 0.000934 | 0.000548 | 0.000386 | 0.25x | 18 | 1.70e-01 | 8 | -4.0000e-04 |
| medium_2p4k | C100H160N25O40S2 | envipat | 83349 | 0.346337 | 0.339740 | 0.006597 | 91.09x | 623 | 5.84e-03 | 7 | 0.0000e+00 |
| medium_2p4k | C100H160N25O40S2 | pyopenms | 83349 | 0.008569 | 0.000324 | 0.008245 | 2.25x | 640 | 5.26e-03 | 7 | 0.0000e+00 |
| large_12k | C500H800N125O200S10 | fastiso | 194481 | 0.008841 | NA | NA | 1.00x | NA | 1.06e-03 | 20 | 0.0000e+00 |
| large_12k | C500H800N125O200S10 | IsoSpecPy | 194481 | 0.203953 | 0.000688 | 0.203266 | 23.07x | 7103 | 0.00e+00 | 19 | 0.0000e+00 |
| large_12k | C500H800N125O200S10 | brainpy | 194481 | 0.002647 | 0.000523 | 0.002124 | 0.30x | 35 | 4.57e-02 | 20 | -2.0000e-04 |
| large_12k | C500H800N125O200S10 | envipat | 194481 | 0.730892 | 0.421420 | 0.309472 | 82.67x | 13693 | 1.10e-02 | 19 | 0.0000e+00 |
| large_12k | C500H800N125O200S10 | pyopenms | 194481 | 0.361712 | 0.006387 | 0.355325 | 40.91x | 14235 | 7.25e-03 | 19 | 0.0000e+00 |

## Top Cluster Fine Structure

| case | backend | cluster | Iso fine peaks | Iso span Da | effective peaks | top peak share | backend peaks | centroid shift Da |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_glucose | brainpy | 0 | 1 | 0.0000 | 1.00 | 1.000 | 1 | -6.2400e-09 |
| small_glucose | brainpy | 1 | 3 | 0.0029 | 1.11 | 0.947 | 1 | 2.6156e-07 |
| small_glucose | brainpy | 2 | 2 | 0.0025 | 1.28 | 0.874 | 1 | 6.2286e-05 |
| small_glucose | brainpy | 3 | 1 | 0.0000 | 1.00 | 1.000 | 1 | 1.8593e-04 |
| small_glucose | pyopenms | 0 | 1 | 0.0000 | 1.00 | 1.000 | 1 | 2.2744e-06 |
| small_glucose | pyopenms | 1 | 3 | 0.0029 | 1.11 | 0.947 | 3 | 2.6931e-06 |
| small_glucose | pyopenms | 2 | 2 | 0.0025 | 1.28 | 0.874 | 6 | 7.0682e-05 |
| small_glucose | pyopenms | 3 | 1 | 0.0000 | 1.00 | 1.000 | 6 | 1.9306e-04 |
| small_glucose | envipat | 0 | 1 | 0.0000 | 1.00 | 1.000 | 1 | -4.4400e-09 |
| small_glucose | envipat | 1 | 3 | 0.0029 | 1.11 | 0.947 | 3 | 2.5877e-07 |
| small_glucose | envipat | 2 | 2 | 0.0025 | 1.28 | 0.874 | 6 | 6.1286e-05 |
| small_glucose | envipat | 3 | 1 | 0.0000 | 1.00 | 1.000 | 6 | 1.8270e-04 |
| medium_2p4k | brainpy | 1 | 5 | 0.0092 | 1.27 | 0.886 | 1 | -3.8138e-06 |
| medium_2p4k | brainpy | 0 | 1 | 0.0000 | 1.00 | 1.000 | 1 | -3.9080e-07 |
| medium_2p4k | brainpy | 2 | 17 | 0.0185 | 2.29 | 0.637 | 1 | -1.8918e-05 |
| medium_2p4k | brainpy | 3 | 29 | 0.0248 | 4.08 | 0.408 | 1 | -2.8447e-05 |
| medium_2p4k | brainpy | 4 | 40 | 0.0294 | 6.09 | 0.239 | 1 | -3.4996e-05 |
| medium_2p4k | brainpy | 5 | 43 | 0.0319 | 8.29 | 0.212 | 1 | -5.2489e-05 |
| medium_2p4k | pyopenms | 1 | 5 | 0.0092 | 1.27 | 0.886 | 5 | 5.4124e-06 |
| medium_2p4k | pyopenms | 0 | 1 | 0.0000 | 1.00 | 1.000 | 1 | 1.4140e-05 |
| medium_2p4k | pyopenms | 2 | 17 | 0.0185 | 2.29 | 0.637 | 17 | -1.9161e-05 |
| medium_2p4k | pyopenms | 3 | 29 | 0.0248 | 4.08 | 0.408 | 41 | -3.6935e-05 |
| medium_2p4k | pyopenms | 4 | 40 | 0.0294 | 6.09 | 0.239 | 71 | -6.1964e-05 |
| medium_2p4k | pyopenms | 5 | 43 | 0.0319 | 8.29 | 0.212 | 98 | -1.0601e-04 |
| medium_2p4k | envipat | 1 | 5 | 0.0092 | 1.27 | 0.886 | 5 | -4.2124e-06 |
| medium_2p4k | envipat | 0 | 1 | 0.0000 | 1.00 | 1.000 | 1 | -7.9440e-07 |
| medium_2p4k | envipat | 2 | 17 | 0.0185 | 2.29 | 0.637 | 17 | -1.9344e-05 |
| medium_2p4k | envipat | 3 | 29 | 0.0248 | 4.08 | 0.408 | 41 | -2.8933e-05 |
| medium_2p4k | envipat | 4 | 40 | 0.0294 | 6.09 | 0.239 | 71 | -3.5458e-05 |
| medium_2p4k | envipat | 5 | 43 | 0.0319 | 8.29 | 0.212 | 98 | -5.3009e-05 |
| large_12k | brainpy | 7 | 256 | 0.0496 | 11.15 | 0.193 | 1 | -5.9481e-05 |
| large_12k | brainpy | 8 | 319 | 0.0550 | 15.21 | 0.135 | 1 | -7.0181e-05 |
| large_12k | brainpy | 6 | 188 | 0.0450 | 7.89 | 0.269 | 1 | -4.9514e-05 |
| large_12k | brainpy | 9 | 412 | 0.0596 | 20.08 | 0.104 | 1 | -8.0203e-05 |
| large_12k | brainpy | 5 | 119 | 0.0386 | 5.42 | 0.365 | 1 | -3.9544e-05 |
| large_12k | brainpy | 10 | 495 | 0.0613 | 25.74 | 0.087 | 1 | -9.0286e-05 |
| large_12k | pyopenms | 7 | 256 | 0.0496 | 11.15 | 0.193 | 364 | -4.0336e-05 |
| large_12k | pyopenms | 8 | 319 | 0.0550 | 15.21 | 0.135 | 503 | -6.1839e-05 |
| large_12k | pyopenms | 6 | 188 | 0.0450 | 7.89 | 0.269 | 253 | -2.0514e-05 |
| large_12k | pyopenms | 9 | 412 | 0.0596 | 20.08 | 0.104 | 619 | -8.3492e-05 |
| large_12k | pyopenms | 5 | 119 | 0.0386 | 5.42 | 0.365 | 160 | -1.5253e-06 |
| large_12k | pyopenms | 10 | 495 | 0.0613 | 25.74 | 0.087 | 767 | -1.0631e-04 |
| large_12k | envipat | 7 | 256 | 0.0496 | 11.15 | 0.193 | 354 | -6.1467e-05 |
| large_12k | envipat | 8 | 319 | 0.0550 | 15.21 | 0.135 | 502 | -7.2174e-05 |
| large_12k | envipat | 6 | 188 | 0.0450 | 7.89 | 0.269 | 251 | -5.1542e-05 |
| large_12k | envipat | 9 | 412 | 0.0596 | 20.08 | 0.104 | 612 | -8.1936e-05 |
| large_12k | envipat | 5 | 119 | 0.0386 | 5.42 | 0.365 | 160 | -4.1582e-05 |
| large_12k | envipat | 10 | 495 | 0.0613 | 25.74 | 0.087 | 744 | -9.1654e-05 |
