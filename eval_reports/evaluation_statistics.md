# Evaluation Report Statistics
*Generated: analyze_reports.py*

## ParaView MCP (ParaView Tasks)
**Number of trials:** 3

*ParaView MCP agent on standard ParaView visualization tasks*

### Metrics Summary
| Metric | Mean | Std Dev | Variance | Min | Max |
|--------|------|---------|----------|-----|-----|
| Overall Score | 29.80% | 9.46 | 89.41 | 23.80% | 40.70% |
| Overall Score (Points) | 723.33/2430 | 229.58 | 52705.33 | 578.00 | 988.00 |
| Completion Rate | 59.72% | 15.07 | 227.14 | 50.00% | 77.08% |
| Avg Vision Score | 27.10% | 8.84 | 78.12 | 21.70% | 37.30% |
| Vision Quality (Points) | 457.67/1670 | 148.46 | 22041.33 | 367.00 | 629.00 |
| PSNR | 12.00dB | 2.96 | 8.79 | 10.17dB | 15.42dB |
| SSIM | 0.54 | 0.14 | 0.02 | 0.46 | 0.71 |
| LPIPS | 0.49 | 0.14 | 0.02 | 0.33 | 0.57 |

### Pass@k and Pass^k Metrics
| k | pass@k | pass^k |
|---|--------|--------|
| 1 | 0.285 | 0.285 |
| 2 | 0.389 | 0.181 |
| 3 | 0.479 | 0.167 |

### Top 5 Failures
| Case | Success Rate |
|------|-------------|
| ABC | 0/3 (0.0%) |
| carp | 0/3 (0.0%) |
| chameleon_isosurface | 0/3 (0.0%) |
| chart-opacity | 0/3 (0.0%) |
| crayfish_streamline | 0/3 (0.0%) |

### Top 5 Successes
| Case | Success Rate |
|------|-------------|
| miranda | 3/3 (100.0%) |
| ml-iso | 3/3 (100.0%) |
| richtmyer | 3/3 (100.0%) |
| rti-velocity_slices | 3/3 (100.0%) |
| vortex | 3/3 (100.0%) |

## ChatVis (ParaView Tasks)
**Number of trials:** 5

*ChatVis agent on standard ParaView visualization tasks*

### Metrics Summary
| Metric | Mean | Std Dev | Variance | Min | Max |
|--------|------|---------|----------|-----|-----|
| Overall Score | 39.08% | 3.50 | 12.24 | 36.70% | 45.10% |
| Overall Score (Points) | 949.60/2430 | 84.48 | 7136.80 | 891.00 | 1095.00 |
| Completion Rate | 56.25% | 4.66 | 21.70 | 54.17% | 64.58% |
| Avg Vision Score | 33.64% | 3.09 | 9.55 | 31.30% | 38.70% |
| Vision Quality (Points) | 573.40/1670 | 46.75 | 2185.80 | 532.00 | 647.00 |
| PSNR | 10.50dB | 0.97 | 0.94 | 9.59dB | 12.08dB |
| SSIM | 0.50 | 0.04 | 0.00 | 0.48 | 0.57 |
| LPIPS | 0.50 | 0.04 | 0.00 | 0.43 | 0.52 |

### Pass@k and Pass^k Metrics
| k | pass@k | pass^k |
|---|--------|--------|
| 1 | 0.479 | 0.479 |
| 2 | 0.562 | 0.396 |
| 3 | 0.604 | 0.354 |
| 4 | 0.629 | 0.329 |
| 5 | 0.646 | 0.312 |

### Top 5 Failures
| Case | Success Rate |
|------|-------------|
| argon-bubble | 0/5 (0.0%) |
| bonsai | 0/5 (0.0%) |
| carp | 0/5 (0.0%) |
| chameleon_isosurface | 0/5 (0.0%) |
| engine | 0/5 (0.0%) |

### Top 5 Successes
| Case | Success Rate |
|------|-------------|
| solar-plume | 5/5 (100.0%) |
| supernova_isosurface | 5/5 (100.0%) |
| tornado | 5/5 (100.0%) |
| twoswirls_streamribbon | 5/5 (100.0%) |
| vortex | 5/5 (100.0%) |

## ParaView MCP (Anonymized Tasks)
**Number of trials:** 3

*ParaView MCP agent on "what obj" anonymized tasks*

### Metrics Summary
| Metric | Mean | Std Dev | Variance | Min | Max |
|--------|------|---------|----------|-----|-----|
| Overall Score | 42.77% | 1.53 | 2.33 | 41.10% | 44.10% |
| Overall Score (Points) | 394.67/923 | 9.71 | 94.33 | 384.00 | 403.00 |
| Completion Rate | 92.59% | 3.70 | 13.72 | 88.89% | 96.30% |
| Avg Vision Score | 51.90% | 4.10 | 16.81 | 47.80% | 56.00% |
| Vision Quality (Points) | 134.00/267 | 6.08 | 37.00 | 130.00 | 141.00 |

### Pass@k and Pass^k Metrics
| k | pass@k | pass^k |
|---|--------|--------|
| 1 | 0.259 | 0.259 |
| 2 | 0.358 | 0.160 |
| 3 | 0.407 | 0.111 |

### Top 5 Failures
| Case | Success Rate |
|------|-------------|
| dataset_002 | 0/3 (0.0%) |
| dataset_003 | 0/3 (0.0%) |
| dataset_005 | 0/3 (0.0%) |
| dataset_006 | 0/3 (0.0%) |
| dataset_007 | 0/3 (0.0%) |

### Top 5 Successes
| Case | Success Rate |
|------|-------------|
| dataset_016 | 2/3 (66.7%) |
| dataset_025 | 2/3 (66.7%) |
| dataset_015 | 3/3 (100.0%) |
| dataset_017 | 3/3 (100.0%) |
| dataset_027 | 3/3 (100.0%) |
