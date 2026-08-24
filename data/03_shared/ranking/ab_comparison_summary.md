# Ranking engine A/B comparison

Disclaimer: precision/recall/F1 are graded against the same profile fields used to score each item, so they're circular -- not proof of real recommendation quality. Trust Kendall's tau and top-K overlap instead; they just measure how much Model A and B actually differ.

## synthetic_user_rising_basics

kendall_tau=0.898  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-06_002 (office_siren, 0.700) | 2026-01_000 (basics, 0.725) |
| 2 | 2024-09_002 (quiet_luxury, 0.700) | 2025-07_001 (basics, 0.525) |
| 3 | 2026-01_000 (basics, 0.700) | 2025-09_009 (basics, 0.525) |
| 4 | 2026-01_002 (quiet_luxury, 0.700) | 2024-06_002 (office_siren, 0.400) |
| 5 | 2025-01_002 (quiet_luxury, 0.600) | 2024-09_002 (quiet_luxury, 0.400) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_persistent_luxury

kendall_tau=0.508  top5_overlap=0.667

| rank | Model A | Model B |
|---|---|---|
| 1 | 2025-12_001 (quiet_luxury, 1.000) | 2025-12_001 (quiet_luxury, 0.895) |
| 2 | 2024-11_002 (quiet_luxury, 0.700) | 2024-11_002 (quiet_luxury, 0.695) |
| 3 | 2024-11_001 (quiet_luxury, 0.400) | 2024-04_000 (quiet_luxury, 0.495) |
| 4 | 2024-02_059 (office_siren, 0.300) | 2024-10_000 (quiet_luxury, 0.495) |
| 5 | 2024-04_000 (quiet_luxury, 0.300) | 2024-11_001 (quiet_luxury, 0.495) |

Model A @k=5: precision=0.600 recall=1.000 f1=0.750

Model A @k=10: precision=0.300 recall=1.000 f1=0.462

Model B @k=5: precision=0.600 recall=1.000 f1=0.750

Model B @k=10: precision=0.300 recall=1.000 f1=0.462

## synthetic_user_seasonal_mobwife

kendall_tau=0.756  top5_overlap=0.111

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-11_000 (mob_wife, 1.000) | 2024-11_000 (mob_wife, 0.925) |
| 2 | 2024-10_000 (quiet_luxury, 0.400) | 2024-05_000 (mob_wife, 0.325) |
| 3 | 2024-10_001 (quiet_luxury, 0.300) | 2024-08_000 (mob_wife, 0.325) |
| 4 | 2024-12_000 (quiet_luxury, 0.300) | 2024-09_001 (mob_wife, 0.325) |
| 5 | 2025-01_000 (quiet_luxury, 0.300) | 2025-04_002 (mob_wife, 0.325) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_inactive_officesiren

kendall_tau=0.821  top5_overlap=0.250

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-09_000 (quiet_luxury, 1.000) | 2024-07_119 (office_siren, 0.695) |
| 2 | 2024-07_000 (quiet_luxury, 0.700) | 2025-08_026 (office_siren, 0.695) |
| 3 | 2024-07_119 (office_siren, 0.700) | 2024-09_000 (quiet_luxury, 0.600) |
| 4 | 2024-09_001 (mob_wife, 0.700) | 2024-02_059 (office_siren, 0.495) |
| 5 | 2024-12_000 (quiet_luxury, 0.700) | 2024-06_001 (office_siren, 0.495) |

Model A @k=5: precision=0.200 recall=0.333 f1=0.250

Model A @k=10: precision=0.200 recall=0.667 f1=0.308

Model B @k=5: precision=0.600 recall=1.000 f1=0.750

Model B @k=10: precision=0.300 recall=1.000 f1=0.462
