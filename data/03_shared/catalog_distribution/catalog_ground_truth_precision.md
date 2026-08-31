# Catalog ground-truth precision

From 161 human judgments collected in Discover (👍/👎).

**Precision at the shipped threshold (0.70): 0.647** (33/51 correct - the subset of judged items whose max_sim clears the current threshold)

Precision across all 161 judgments ever collected (judged under whichever open-set threshold was shipped at the time - most of this sample predates the current threshold): 0.509 (82/161 correct)

## Per-trend precision (across all judgments, not threshold-filtered)

| Trend | Precision | n |
|---|---|---|
| basics | 0.500 | 54 |
| mob_wife | 0.800 | 10 |
| office_siren | 0.310 | 29 |
| quiet_luxury | 0.559 | 68 |

## Threshold sensitivity (within the judged sample)

Only checks thresholds >= the shipped threshold (0.70) - the judged sample only contains items that already cleared whatever threshold was shipped when they were judged, so precision below that isn't knowable from this data.

| Threshold | Precision | n |
|---|---|---|
| 0.70 | 0.647 | 51 |
| 0.72 | 0.735 | 34 |
| 0.75 | 0.533 | 15 |
| 0.80 | - | 0 |
