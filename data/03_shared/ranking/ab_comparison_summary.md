# Ranking engine A/B comparison

Disclaimer: precision/recall/F1 are graded against the same profile fields used to score each item, so they're circular -- not proof of real recommendation quality. Trust Kendall's tau and top-K overlap instead; they just measure how much Model A and B actually differ.

## synthetic_user_rising_basics

kendall_tau=0.721  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-09_002 (quiet_luxury, 0.700) | 2025-01_002 (basics, 0.500) |
| 2 | 2026-01_000 (basics, 0.700) | 2026-01_000 (basics, 0.500) |
| 3 | 2025-01_002 (basics, 0.600) | 2024-09_002 (quiet_luxury, 0.400) |
| 4 | 2024-11_000 (mob_wife, 0.300) | 2025-09_009 (basics, 0.300) |
| 5 | 2025-01_001 (quiet_luxury, 0.300) | 2025-12_000 (basics, 0.300) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_persistent_luxury

kendall_tau=0.912  top5_overlap=1.000

| rank | Model A | Model B |
|---|---|---|
| 1 | 2025-12_001 (quiet_luxury, 1.000) | 2025-12_001 (quiet_luxury, 0.925) |
| 2 | 2024-11_002 (quiet_luxury, 0.700) | 2024-11_002 (quiet_luxury, 0.725) |
| 3 | 2024-02_059 (quiet_luxury, 0.300) | 2024-02_059 (quiet_luxury, 0.525) |
| 4 | 2024-07_119 (quiet_luxury, 0.300) | 2024-07_119 (quiet_luxury, 0.525) |
| 5 | 2024-10_000 (quiet_luxury, 0.300) | 2024-10_000 (quiet_luxury, 0.525) |

Model A @k=5: precision=0.400 recall=1.000 f1=0.571

Model A @k=10: precision=0.200 recall=1.000 f1=0.333

Model B @k=5: precision=0.400 recall=1.000 f1=0.571

Model B @k=10: precision=0.200 recall=1.000 f1=0.333

## synthetic_user_seasonal_mobwife

kendall_tau=1.000  top5_overlap=1.000

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-11_000 (mob_wife, 1.000) | 2024-11_000 (mob_wife, 0.895) |
| 2 | 2024-10_000 (quiet_luxury, 0.400) | 2024-10_000 (quiet_luxury, 0.200) |
| 3 | 2025-01_002 (basics, 0.300) | 2025-01_002 (basics, 0.200) |
| 4 | 2025-09_009 (basics, 0.300) | 2025-09_009 (basics, 0.200) |
| 5 | 2025-09_010 (quiet_luxury, 0.300) | 2025-09_010 (quiet_luxury, 0.200) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_inactive_officesiren

kendall_tau=0.971  top5_overlap=1.000

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-07_119 (quiet_luxury, 0.700) | 2024-07_119 (quiet_luxury, 0.400) |
| 2 | 2024-07_002 (quiet_luxury, 0.400) | 2024-02_059 (quiet_luxury, 0.200) |
| 3 | 2024-02_059 (quiet_luxury, 0.300) | 2024-05_000 (quiet_luxury, 0.200) |
| 4 | 2024-05_000 (quiet_luxury, 0.300) | 2024-07_002 (quiet_luxury, 0.200) |
| 5 | 2024-09_002 (quiet_luxury, 0.300) | 2024-09_002 (quiet_luxury, 0.200) |

Model A @k=5: precision=0.000 recall=0.000 f1=0.000

Model A @k=10: precision=0.000 recall=0.000 f1=0.000

Model B @k=5: precision=0.000 recall=0.000 f1=0.000

Model B @k=10: precision=0.000 recall=0.000 f1=0.000
