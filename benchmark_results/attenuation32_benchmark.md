# Float32 Attenuation And Production Storage Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Baseline is research storage with float64 attenuation and double phase.
- `research32_phase64` isolates the float32 attenuation effect while keeping double phase.
- `production32_auto` stores float32 attenuation, double phase, uint64 phase, and thresholds; it discards cyclephase-only tables.
- `minimal32_auto` stores float32 attenuation plus uint64 phase and thresholds; it discards double phase tables.
- Auto variants are timed after resolving the selected kernel, matching the table/profile call path where selection happens once.
- Speed ratios above 1 mean the variant kernel is faster than the research64 baseline.

| case | batch | R | n_fft | active | research64 s | attn32 s | attn32+uint s | production s | minimal s | prod selected | min selected | prod memory | min memory | prod rel L2 | prod speed | min speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| huge_120k | 1 | 0 | 59049 | 0.2099 | 0.000469 | 0.000487 | 0.000490 | 0.000518 | 0.000477 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.76e-08 | 0.905x | 0.983x |
| huge_120k | 1 | 100000 | 59049 | 0.001998 | 0.000045 | 0.000046 | 0.000046 | 0.000046 | 0.000045 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.18e-08 | 0.978x | 1.009x |
| huge_120k | 10 | 0 | 59049 | 0.2052 | 0.005235 | 0.006069 | 0.005758 | 0.005465 | 0.005566 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.74e-08 | 0.958x | 0.941x |
| huge_120k | 10 | 100000 | 59049 | 0.001998 | 0.000122 | 0.000123 | 0.000126 | 0.000124 | 0.000126 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 8.88e-09 | 0.984x | 0.964x |
| huge_120k | 50 | 0 | 64827 | 0.1882 | 0.027771 | 0.031138 | 0.031723 | 0.030448 | 0.030852 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.80e-08 | 0.912x | 0.900x |
| huge_120k | 50 | 100000 | 64827 | 0.001907 | 0.000582 | 0.000563 | 0.000520 | 0.000555 | 0.000536 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.18e-09 | 1.048x | 1.085x |
| huge_240k | 1 | 0 | 83349 | 0.03187 | 0.000341 | 0.000410 | 0.000412 | 0.000311 | 0.000302 | cython_log_pruned_attn32_uintphase_threshold | cython_log_pruned_attn32_uintphase_threshold | 0.681x | 0.468x | 1.89e-08 | 1.097x | 1.132x |
| huge_240k | 1 | 100000 | 83349 | 0.0007439 | 0.000052 | 0.000053 | 0.000060 | 0.000056 | 0.000054 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.40e-08 | 0.930x | 0.959x |
| huge_240k | 10 | 0 | 83349 | 0.03132 | 0.004299 | 0.004929 | 0.004824 | 0.003661 | 0.003892 | cython_log_pruned_attn32_uintphase_threshold | cython_log_pruned_attn32_uintphase_threshold | 0.681x | 0.468x | 1.77e-08 | 1.174x | 1.105x |
| huge_240k | 10 | 100000 | 83349 | 0.0007439 | 0.000112 | 0.000113 | 0.000112 | 0.000111 | 0.000113 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.16e-09 | 1.005x | 0.987x |
| huge_240k | 50 | 0 | 84035 | 0.02918 | 0.020345 | 0.024264 | 0.026112 | 0.019856 | 0.020030 | cython_log_pruned_attn32_uintphase_threshold | cython_log_pruned_attn32_uintphase_threshold | 0.681x | 0.468x | 1.61e-08 | 1.025x | 1.016x |
| huge_240k | 50 | 100000 | 84035 | 0.0007159 | 0.000424 | 0.000401 | 0.000414 | 0.000394 | 0.000396 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 7.15e-09 | 1.076x | 1.070x |
| huge_60k | 1 | 0 | 42525 | 0.4227 | 0.000539 | 0.000559 | 0.000583 | 0.000513 | 0.000538 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.69e-08 | 1.051x | 1.002x |
| huge_60k | 1 | 100000 | 42525 | 0.005691 | 0.000047 | 0.000047 | 0.000051 | 0.000052 | 0.000051 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 7.02e-09 | 0.905x | 0.907x |
| huge_60k | 10 | 0 | 42525 | 0.4183 | 0.005304 | 0.006116 | 0.005818 | 0.005628 | 0.005739 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.83e-08 | 0.942x | 0.924x |
| huge_60k | 10 | 100000 | 42525 | 0.005526 | 0.000156 | 0.000162 | 0.000169 | 0.000169 | 0.000169 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 8.73e-09 | 0.923x | 0.927x |
| huge_60k | 50 | 0 | 45927 | 0.3984 | 0.026811 | 0.028803 | 0.031956 | 0.039276 | 0.033082 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.85e-08 | 0.683x | 0.810x |
| huge_60k | 50 | 100000 | 45927 | 0.005157 | 0.000643 | 0.000806 | 0.000791 | 0.000749 | 0.000718 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.48e-09 | 0.858x | 0.895x |
| large_12k | 1 | 0 | 32805 | 1 | 0.000869 | 0.000881 | 0.000802 | 0.000795 | 0.000832 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 2.57e-08 | 1.092x | 1.044x |
| large_12k | 1 | 100000 | 32805 | 0.07907 | 0.000093 | 0.000095 | 0.000097 | 0.000094 | 0.000096 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.68e-09 | 0.982x | 0.966x |
| large_12k | 10 | 0 | 32805 | 1 | 0.007943 | 0.007893 | 0.008824 | 0.008062 | 0.007816 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 2.61e-08 | 0.985x | 1.016x |
| large_12k | 10 | 100000 | 32805 | 0.07566 | 0.000713 | 0.000735 | 0.000720 | 0.000801 | 0.000855 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.91e-09 | 0.891x | 0.834x |
| large_12k | 50 | 0 | 32805 | 1 | 0.038578 | 0.038713 | 0.039920 | 0.039629 | 0.039202 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 2.56e-08 | 0.973x | 0.984x |
| large_12k | 50 | 100000 | 32805 | 0.06254 | 0.003063 | 0.003161 | 0.003033 | 0.002945 | 0.003062 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.54e-09 | 1.040x | 1.000x |
| xlarge_29k | 1 | 0 | 32805 | 0.812 | 0.000759 | 0.000689 | 0.000808 | 0.000760 | 0.000692 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 2.02e-08 | 0.999x | 1.096x |
| xlarge_29k | 1 | 100000 | 32805 | 0.02018 | 0.000061 | 0.000056 | 0.000060 | 0.000061 | 0.000055 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.14e-08 | 1.002x | 1.099x |
| xlarge_29k | 10 | 0 | 32805 | 0.7911 | 0.006927 | 0.006734 | 0.006756 | 0.007607 | 0.006939 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.92e-08 | 0.911x | 0.998x |
| xlarge_29k | 10 | 100000 | 32805 | 0.01914 | 0.000267 | 0.000287 | 0.000278 | 0.000278 | 0.000283 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 9.36e-09 | 0.963x | 0.947x |
| xlarge_29k | 50 | 0 | 35721 | 0.7226 | 0.031444 | 0.032988 | 0.033435 | 0.033863 | 0.033785 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 1.96e-08 | 0.929x | 0.931x |
| xlarge_29k | 50 | 100000 | 35721 | 0.01597 | 0.001084 | 0.001178 | 0.001158 | 0.001333 | 0.001376 | cython_log_pruned_attn32 | cython_log_pruned_attn32_uintphase | 0.681x | 0.468x | 8.26e-09 | 0.813x | 0.788x |
