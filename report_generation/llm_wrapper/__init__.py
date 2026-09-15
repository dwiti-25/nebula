"""nebula.llm_wrapper -- a thin natural-language interface around the
existing NEBULA pipeline (experiments/run_autockt_pipeline.py).

This package adds NO circuit-design, SPICE, PPO, or measurement logic of
its own. It only: (1) turns a natural-language request into the existing
rl.target_spec.TargetSpec field schema, (2) invokes the existing,
unmodified experiments.run_autockt_pipeline CLI as a subprocess (the same
entry point experiments/web_ui.py already uses, for the same reason: crash
isolation and a single source of truth for selection/measurement logic),
and (3) formats that pipeline's own JSON output -- including its existing
PASS/FAIL/NOT CLAIMED verdict rows (analysis/final_specification.py) --
into a natural-language report. It is an orchestration/formatting layer,
not a circuit designer: it does not claim the LLM improves circuit
quality, and it never invents a metric value or verdict that the
underlying pipeline did not itself produce.
"""
