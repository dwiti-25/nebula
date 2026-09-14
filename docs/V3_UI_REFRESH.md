# v3 review fixes and dashboard refresh

The main UI now uses `analysis/v3_dashboard.py`, not legacy model comparisons.
Historical aggregation remains available in `analysis/performance_dashboard.py`
for reproducibility. Current plots are regenerated from v3 event records and
measured new evaluation reports; unknown PVT/HD3/noise data is not substituted
from Design A or a previous policy.

## Review fixes

- Non-object JSON cache entries become misses instead of exceptions.
- Final-report RF limits use strict boundaries, and area uses selected MOS sizes.
- New full checkpoints record initial indices; mismatched resume is rejected.
  Older checkpoints without this provenance remain valid for inference but
  are rejected for resume rather than guessing their starting configuration.
- UI watchdog termination targets the child process tree (Windows taskkill,
  POSIX process group), not only the parent Python process.
- Model dependency fingerprints include `.inc` and external `.lib` references.
- CI includes component/parser/MOS tests and Windows/Linux jobs.
- Graph responses revalidate rather than remaining stale for five minutes.

## Fresh real-SPICE measurements

`results/v3_ui_comparison_20260912.json`: random search, CEM, fresh v3 and
trained v3, 8 requests per method for each of two baseline seeds. All 64
requests completed. All methods had zero nominal strict passes in this small
grid-center study. PPO is deterministic here, so its two runs are repeats,
not independent policy-quality trials. Existing PPO training cost is excluded.

`results/v3_ui_inference_20260912.json`: the same trained v3 checkpoint,
verified starting state, 3 episodes, 15-request / 120-second search ceilings.
Two episodes produced nominally feasible candidates. This is not comparable
to the grid-center study and not FINAL or full-PVT qualification.

Twenty evidence graphs are available: saved v3 training reward/success/failure,
PPO health, physical metrics and timing, eye/power scatter, new method
comparison, fresh inference metrics, and measured serial/parallel throughput.
Training weights were not changed or retrained. Full PVT is explicitly shown
as not validated. The new grid-center comparison does not demonstrate PPO
superiority; larger held-out-target/multi-seed studies remain necessary.
