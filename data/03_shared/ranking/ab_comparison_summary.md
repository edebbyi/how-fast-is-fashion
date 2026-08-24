# Ranking engine A/B comparison

Disclaimer: precision/recall/F1 are graded against the same profile fields used to score each item, so they're circular -- not proof of real recommendation quality. Trust Kendall's tau and top-K overlap instead; they just measure how much Model A and B actually differ.

## synthetic_user_rising_basics

kendall_tau=0.850  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-09_002 (quiet_luxury, 0.700) | 2026-01_000 (basics, 0.800) |
| 2 | 2026-01_000 (basics, 0.700) | 2025-09_009 (basics, 0.600) |
| 3 | 2024-11_000 (mob_wife, 0.300) | 2024-09_002 (quiet_luxury, 0.400) |
| 4 | 2025-01_000 (quiet_luxury, 0.300) | 2024-10_002 (basics, 0.400) |
| 5 | 2025-01_001 (quiet_luxury, 0.300) | 2024-11_000 (mob_wife, 0.200) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_persistent_luxury

kendall_tau=0.573  top5_overlap=0.429

| rank | Model A | Model B |
|---|---|---|
| 1 | 2025-12_001 (quiet_luxury, 1.000) | 2025-12_001 (quiet_luxury, 0.925) |
| 2 | 2024-11_002 (quiet_luxury, 0.700) | 2024-11_002 (quiet_luxury, 0.725) |
| 3 | 2024-11_001 (quiet_luxury, 0.400) | 2024-10_000 (quiet_luxury, 0.525) |
| 4 | 2024-02_059 (office_siren, 0.300) | 2024-11_001 (quiet_luxury, 0.525) |
| 5 | 2024-07_119 (office_siren, 0.300) | 2025-08_008 (quiet_luxury, 0.525) |

Model A @k=5: precision=0.600 recall=1.000 f1=0.750

Model A @k=10: precision=0.300 recall=1.000 f1=0.462

Model B @k=5: precision=0.600 recall=1.000 f1=0.750

Model B @k=10: precision=0.300 recall=1.000 f1=0.462

## synthetic_user_seasonal_mobwife

kendall_tau=0.929  top5_overlap=0.667

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-11_000 (mob_wife, 1.000) | 2024-11_000 (mob_wife, 0.895) |
| 2 | 2024-10_000 (quiet_luxury, 0.400) | 2024-05_000 (mob_wife, 0.295) |
| 3 | 2025-01_000 (quiet_luxury, 0.300) | 2024-10_000 (quiet_luxury, 0.200) |
| 4 | 2025-09_009 (basics, 0.300) | 2025-01_000 (quiet_luxury, 0.200) |
| 5 | 2025-09_010 (quiet_luxury, 0.300) | 2025-09_009 (basics, 0.200) |

Model A @k=5: precision=0.200 recall=1.000 f1=0.333

Model A @k=10: precision=0.100 recall=1.000 f1=0.182

Model B @k=5: precision=0.200 recall=1.000 f1=0.333

Model B @k=10: precision=0.100 recall=1.000 f1=0.182

## synthetic_user_inactive_officesiren

kendall_tau=0.913  top5_overlap=0.667

| rank | Model A | Model B |
|---|---|---|
| 1 | 2024-07_119 (office_siren, 0.700) | 2024-07_119 (office_siren, 0.575) |
| 2 | 2025-08_026 (office_siren, 0.700) | 2025-08_026 (office_siren, 0.575) |
| 3 | 2024-07_002 (quiet_luxury, 0.400) | 2024-02_059 (office_siren, 0.375) |
| 4 | 2024-02_003 (quiet_luxury, 0.300) | 2025-07_017 (office_siren, 0.375) |
| 5 | 2024-02_059 (office_siren, 0.300) | 2024-02_003 (quiet_luxury, 0.200) |

Model A @k=5: precision=0.400 recall=1.000 f1=0.571

Model A @k=10: precision=0.200 recall=1.000 f1=0.333

Model B @k=5: precision=0.400 recall=1.000 f1=0.571

Model B @k=10: precision=0.200 recall=1.000 f1=0.333
