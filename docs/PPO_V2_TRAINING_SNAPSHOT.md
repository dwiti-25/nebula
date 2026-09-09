# PPO v2 training snapshot

This snapshot preserves the integrated PPO v2 implementation and the locally
trained synthetic checkpoints for seeds 101 through 105 before the circuit
parameter space is expanded with grouped MOS sizing and area optimization.

The `*_policy.pt` files are inference-only policy state dictionaries. The
`*_full.pt` files are resumable training checkpoints and must not be selected
as inference policies in the pipeline UI.

These checkpoints were trained against the synthetic evaluator and establish
training-mechanics evidence only. They are not evidence of real-SPICE circuit
performance and will not be compatible with the forthcoming expanded
state/action schemas.

Generated JSONL training/event logs remain ignored because they are bulky run
outputs. They can be reproduced with `experiments.train_autockt` using the
configuration recorded in the associated checkpoint metadata and project
documentation.
