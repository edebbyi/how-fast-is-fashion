# Ranking engine A/B comparison

Disclaimer: precision/recall/F1 are graded against the same profile fields used to score each item, so they're circular -- not proof of real recommendation quality. Trust Kendall's tau and top-K overlap instead; they just measure how much Model A and B actually differ.

## synthetic_user_rising_basics

kendall_tau=0.677  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-06_002 (office_siren, 0.700) | 2025-01_002 (basics, 0.800) |
| 2 | 2024-09_002 (quiet_luxury, 0.700) | 2026-01_000 (basics, 0.800) |
| 3 | 2026-01_000 (basics, 0.700) | 2026-01_002 (basics, 0.800) |
| 4 | 2026-01_002 (basics, 0.700) | 2025-09_009 (basics, 0.600) |
| 5 | 2025-01_002 (basics, 0.600) | 2025-12_000 (basics, 0.600) |

Model A @k=5: precision=0.400 recall=1.000 f1=0.571

Model A @k=10: precision=0.200 recall=1.000 f1=0.333

Model B @k=5: precision=0.400 recall=1.000 f1=0.571

Model B @k=10: precision=0.200 recall=1.000 f1=0.333

## synthetic_user_persistent_luxury

kendall_tau=0.598  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2025-12_001 (quiet_luxury, 1.000) | 2025-12_001 (quiet_luxury, 0.895) |
| 2 | 2024-11_002 (quiet_luxury, 0.700) | 2024-11_002 (quiet_luxury, 0.695) |
| 3 | 2024-11_001 (quiet_luxury, 0.400) | 2024-02_059 (quiet_luxury, 0.495) |
| 4 | 2024-02_059 (quiet_luxury, 0.300) | 2024-07_119 (quiet_luxury, 0.495) |
| 5 | 2024-04_000 (basics, 0.300) | 2024-10_000 (quiet_luxury, 0.495) |

Model A @k=5: precision=0.600 recall=1.000 f1=0.750

Model A @k=10: precision=0.300 recall=1.000 f1=0.462

Model B @k=5: precision=0.400 recall=0.667 f1=0.500

Model B @k=10: precision=0.300 recall=1.000 f1=0.462

## synthetic_user_seasonal_mobwife

kendall_tau=0.923  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-11_000 (mob_wife, 1.000) | 2024-11_000 (mob_wife, 0.805) |
| 2 | 2024-10_000 (quiet_luxury, 0.400) | 2024-08_000 (mob_wife, 0.205) |
| 3 | 2024-10_001 (quiet_luxury, 0.300) | 2024-09_001 (mob_wife, 0.205) |
| 4 | 2024-12_000 (quiet_luxury, 0.300) | 2024-10_000 (quiet_luxury, 0.200) |
| 5 | 2025-01_000 (quiet_luxury, 0.300) | 2024-10_001 (quiet_luxury, 0.200) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_inactive_officesiren

kendall_tau=0.911  top5_overlap=0.667

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-09_000 (quiet_luxury, 1.000) | 2024-09_000 (quiet_luxury, 0.600) |
| 2 | 2024-07_000 (quiet_luxury, 0.700) | 2025-08_026 (office_siren, 0.575) |
| 3 | 2024-07_119 (quiet_luxury, 0.700) | 2024-07_000 (quiet_luxury, 0.400) |
| 4 | 2024-09_001 (mob_wife, 0.700) | 2024-07_119 (quiet_luxury, 0.400) |
| 5 | 2024-12_000 (quiet_luxury, 0.700) | 2024-09_001 (mob_wife, 0.400) |

Model A @k=5: precision=0.000 recall=0.000 f1=0.000

Model A @k=10: precision=0.100 recall=0.500 f1=0.167

Model B @k=5: precision=0.200 recall=0.500 f1=0.286

Model B @k=10: precision=0.100 recall=0.500 f1=0.167
