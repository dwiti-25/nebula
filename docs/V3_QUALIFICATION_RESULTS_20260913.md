# v3 completion evidence — TT/SS/FF scope

Required grid: TT/SS/FF × 1.71/1.80/1.89 V × 0/27/75/125 °C (36 conditions).
Physical DFE is excluded. The CTLE remains transistor-level with an ideal tail source; DFE is behavioral.

## Finalist qualification

Sequence status: **complete**.

| Design | Nominal 512 | Nominal 1024 | Corner passes / attempted | Status |
|---|---|---|---|---|
| pipeline_ep0 | PASS | PASS | 36/36 of 36 | qualified |
| pipeline_ep1 | PASS | PASS | 18/36 of 36 | rejected_pvt |

pipeline_ep0 nominal metrics:

- hd3_db: -75.61869714856154
- input_referred_noise_vrms: 0.00037197140533248727
- dfe_locked_phase_eye_height_v: 1.5502710248122609
- dfe_eye_width_ui: 0.7299999999999999
- dfe_error_count: 0
- simulated_bits: 960

### pipeline_ep0 measured corner ranges

| Metric | Minimum | Maximum |
|---|---|---|
| hd3_db | -79.8428 | -69.6987 |
| input_referred_noise_vrms | 0.000328262 | 0.00048729 |
| dfe_locked_phase_eye_height_v | 1.27665 | 1.65185 |
| dfe_eye_width_ui | 0.71 | 0.78 |
| ctle_power_w | 0.0011457 | 0.0012663 |
| dfe_error_count | 0 | 0 |


pipeline_ep1 nominal metrics:

- hd3_db: -72.59901206843718
- input_referred_noise_vrms: 0.00033988356868770736
- dfe_locked_phase_eye_height_v: 1.7095289912303424
- dfe_eye_width_ui: 0.81
- dfe_error_count: 0
- simulated_bits: 960

### pipeline_ep1 measured corner ranges

| Metric | Minimum | Maximum |
|---|---|---|
| hd3_db | -76.8201 | -67.0175 |
| input_referred_noise_vrms | 0.000297491 | 0.000389814 |
| dfe_locked_phase_eye_height_v | 1.36017 | 1.87363 |
| dfe_eye_width_ui | 0.75 | 0.85 |
| ctle_power_w | 0.0011457 | 0.0012663 |
| dfe_error_count | 0 | 0 |


Corners use 512-bit CANDIDATE fidelity including noise and 100mVpp HD3. Nominal FINAL uses 1024 bits and three HD3 amplitudes. Zero observed errors is not a BER guarantee.

## Tuning

| Target GHz | Measured GHz | Within ±5% |
|---|---|---|
| 1.25 | 1.320527183368927 | False |
| 1.875 | 1.908910958457706 | True |
| 2.5 | 2.295117021187582 | False |

Sampled nominal AC sweep only. Continuous coverage and receiver/noise/HD3 compliance at tuning settings remain unproven.

## Frozen-policy comparison

| Method | Requests | Strict passes | Complete batches |
|---|---|---|---|
| random_search | 72 | 0 | 3/3 |
| cem | 72 | 0 | 3/3 |
| fresh_v3 | 72 | 0 | 3/3 |
| trained_v3 | 72 | 0 | 3/3 |

Three shared random starting points, midpoint target, 24 requests per method/seed, horizon 12. This evaluates one frozen trained policy, not three independently trained models. Pretraining cost excluded.

## Outstanding submission limits

- Eye-height threshold is the repository's 0.1 V convention; the original statement still needs confirmation.
- Total receiver area and power are unproven: MOS channel area and CTLE power are partial accounting.
- Synthetic regression channel is used here, not PCIe compliance evidence.
- No jitter/mismatch sign-off, continuous tuning proof or guaranteed BER.
- Failed or unfinished corner checks never qualify a finalist.
- No model retraining was performed in this sequence.

## Source artifacts

- results/v3_qualification_final_20260913.json
- results/v3_nominal_final_20260913.jsonl
- results/v3_tuning_20260913.json
- results/v3_midpoint_comparison_20260913.json
- results/v3_qualification_20260913.json: interrupted 1024-bit-per-corner attempt; timeouts preserved, not treated as electrical failures.
