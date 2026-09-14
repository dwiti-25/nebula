# Agreed completion sequence — 2026-09-13

The current agreed process scope is **TT/SS/FF only**. SF and FS are not
required and are not evaluated by this workflow. The qualification grid is
three corners × 1.71/1.80/1.89 V × 0/27/75/125 °C = **36 conditions**.
Legacy five-corner data/constants are retained for historical reproduction,
not used as the submission acceptance requirement.

## Execution order

1. Deduplicate up to three nominally feasible candidates from the v3 pipeline.
2. Run 512-bit CANDIDATE checks: DC, AC, noise, HD3 and receiver eye.
3. Run nominal 1024-bit FINAL validation and three HD3 amplitudes.
4. Only nominal survivors proceed to the 36-condition grid. The staged grid
   uses 512-bit CANDIDATE measurements, including noise and 100mVpp HD3 at
   every point. Long-pattern/three-amplitude characterization is nominal only.
5. Produce measured specification reports; qualify only complete all-pass
   grids. Timeout/missing measurements never qualify.
6. Separately measure CDEG tuning and frozen-policy/baseline comparisons.
7. Generate the evidence dossier and disclose unresolved requirements.

This separates pattern-length validation from corner screening rather than
claiming that 1024-bit tests were completed at every corner. The first attempt
at 1024 bits per corner timed out under concurrent load; its partial output
is preserved separately.

## Commands

```powershell
.venv\Scripts\python.exe -m experiments.qualification_sequence --input results/v3_ui_inference_20260912.json --output results/NEW_qualification.json --cache results/qualification_cache
.venv\Scripts\python.exe -m experiments.tuning_characterization --input results/v3_ui_inference_20260912.json --output results/NEW_tuning.json --cache results/qualification_cache
.venv\Scripts\python.exe -m experiments.compare_v3 --checkpoint results/ppo_v3_tt_1000_policy.pt --output results/NEW_comparison.json --budget 24 --horizon 12 --seeds 1011 1012 1013 --random-start --target-mode midpoint --seconds-per-method 300
.venv\Scripts\python.exe -m analysis.qualification_report --output docs/NEW_evidence.md
```

Use new output filenames; the cache can be reused. The dossier and dashboard
currently consume the dated September-13 artifacts. Training checkpoints are
unchanged. Test fixtures do not count as physical evidence.

## Scope still requiring explicit evidence

- Nominal 1024-bit PRBS7 has 960 measured bits; no BER guarantee follows.
- Continuous tunability and receiver-level validation of tuning settings.
- Total physical area/power (only CTLE power and partial MOS area are available).
- A final application channel and the original eye-height specification.
- Better PPO performance on held-out targets and independent training seeds.
- Physical DFE is intentionally excluded, not silently claimed as complete.
