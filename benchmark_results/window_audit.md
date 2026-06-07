# FFT Window Audit

- Edge threshold: 1e-06
- `sigma_radius_fits` checks whether +/- sigma_radius * profile_sigma fits within half the FFT mass window.
- `edge_fraction` is the fraction of profile area in the first plus last edge-width bins.

| case | min fft | auto | n_fft | window Da | mean mass Da | profile sigma Da | 6sigma halfwidth Da | fits | edge fraction | edge ok |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| large_12k | 32768 | True | 32805 | 65.6 | 12083.1 | 3.048 | 18.29 | True | 2.00e-13 | True |
| xlarge_29k | 32768 | True | 32805 | 65.6 | 28811.5 | 4.721 | 28.33 | True | 1.31e-08 | True |
| huge_60k | 32768 | True | 42525 | 85.0 | 59165.7 | 6.763 | 40.58 | True | 3.91e-08 | True |
| huge_120k | 32768 | True | 59049 | 118.1 | 118331.3 | 9.571 | 57.43 | True | 1.59e-08 | True |
| huge_240k | 32768 | True | 83349 | 166.7 | 236662.7 | 13.554 | 81.33 | True | 5.33e-09 | True |
| large_12k | 65536 | True | 65625 | 131.2 | 12083.1 | 3.048 | 18.29 | True | 1.09e-14 | True |
| xlarge_29k | 65536 | True | 65625 | 131.2 | 28811.5 | 4.721 | 28.33 | True | 1.85e-14 | True |
| huge_60k | 65536 | True | 65625 | 131.2 | 59165.7 | 6.763 | 40.58 | True | 1.77e-14 | True |
| huge_120k | 65536 | True | 65625 | 131.2 | 118331.3 | 9.571 | 57.43 | True | 6.36e-10 | True |
| huge_240k | 65536 | True | 83349 | 166.7 | 236662.7 | 13.554 | 81.33 | True | 5.33e-09 | True |
| large_12k | 131072 | True | 137781 | 275.6 | 12083.1 | 3.048 | 18.29 | True | 7.97e-15 | True |
| xlarge_29k | 131072 | True | 137781 | 275.6 | 28811.5 | 4.721 | 28.33 | True | 1.11e-14 | True |
| huge_60k | 131072 | True | 137781 | 275.6 | 59165.7 | 6.763 | 40.58 | True | 5.37e-15 | True |
| huge_120k | 131072 | True | 137781 | 275.6 | 118331.3 | 9.571 | 57.43 | True | 1.21e-14 | True |
| huge_240k | 131072 | True | 137781 | 275.6 | 236662.7 | 13.554 | 81.33 | True | 1.10e-14 | True |
