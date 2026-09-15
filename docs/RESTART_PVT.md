# Restarting PVT without retraining

From PowerShell in `C:\Vyom\AL Code\nebula-integration`:

```powershell
.\.venv\Scripts\python.exe -m experiments.qualification_sequence `
  --input results/web_ui_runs/d934b2cce3dc401ab4522cccb6b3b9cf.json `
  --output results/pvt27_restart_512_360.json `
  --cache results/pvt_restart_cache `
  --simulator-timeout-seconds 360 `
  --pvt-pattern-bits 512 `
  --workers 1 `
  --pvt-condition-set saved `
  --export-schematic results/pvt27_restart_512_360.cir
```

This loads fixed candidates, target, channel and recorded PVT conditions from
the old report and its adjacent graph/events files. It does not train or run
the policy. Keep those adjacent files with old UI reports. The example recovers
two candidates and 27 TT/SS/FF conditions on the IBM channel.

Each candidate must pass nominal 512-bit and 1024-bit validation before its
PVT sweep. Choose `--pvt-pattern-bits 512` (default) or
`--pvt-pattern-bits 1024`. The 512-bit mode includes noise and single-amplitude
HD3 at corners; 1024-bit mode uses FINAL fidelity, including three-amplitude
HD3 at corners. A 27-point result is not full 36-point
coverage. A 512-bit result is not a 1024-bit result or a BER guarantee.

Timeout is per SPICE invocation, not the entire run. One worker minimizes
resource contention; two workers are supported. Early training screening is
unchanged. No speedup or new electrical pass is claimed until measured.

## Retry a new restart report

Use its report as `--input`, choose a new `--output` and schematic filename,
and increase `--simulator-timeout-seconds` if needed. Reports are updated after
each recorded measurement, so an interrupted restart report can also be used.
Completed measurements are reused only when runtime fingerprints match;
timeouts are retried. Circuit, PDK, simulator, channel, code or target changes
invalidate reuse. Permanent electrical failures remain failures. Persistent
cache entries can also save exact repeated evaluations.

Old UI reports lack the new resume fingerprint. Their fixed designs can be
restarted, but their old measurements are not imported as new qualification
evidence. In particular, changing 1024-bit corners to 512 bits reruns them.
For an old run interrupted before recording every intended condition, choose
`--pvt-condition-set minimal27` or `full36` explicitly instead of `saved`.

Existing output files are refused. A schematic is exported only for a design
passing all requested conditions and nominal checks. Exit code 1 means no
qualified design; inspect the report for electrical failures versus incomplete
execution. Original reports and checkpoints are never modified.

## New UI runs

Restart the UI server and refresh the page to load the new controls. Set
**PVT / final SPICE timeout per invocation** (default 360 seconds, maximum
3600 in the UI). Use **PVT pattern length** to choose 512 or 1024 bits for
corner runs (512 is the default).
A selected real-PVT design must pass nominal 1024-bit validation even if the
optional HD3/noise checkbox is off. The CLI equivalent is
`--simulator-timeout-seconds 360` on `experiments.run_autockt_pipeline`.

Restarting saved runs is currently a PowerShell command, not a UI button.
