"""Eight-dimensional algebraic test landscape; NEVER SPICE evidence."""
from dataclasses import asdict
import math
from simulator.receiver import ReceiverEvaluation, StageResult, EvaluationFidelity
from simulator.config import SimulationConditions
from simulator.provenance import stable_fingerprint
from .synthetic_benchmark import synthetic_evaluate_receiver

def synthetic_evaluate_receiver_v3(parameters, conditions=SimulationConditions(), fidelity=EvaluationFidelity.TRAINING, **kwargs):
    base = synthetic_evaluate_receiver(parameters, conditions, fidelity, **kwargs)
    metrics = dict(base.metrics)
    if metrics:
        geometry = (math.log(parameters.mos_width_um / 10.0) ** 2
                    + math.log(parameters.mos_length_um / 0.15) ** 2
                    + math.log(parameters.mos_multiplier) ** 2) / 20.0
        factor = math.exp(-geometry)
        for key in ("dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "dfe_min_margin_v"):
            metrics[key] *= factor
        metrics["ctle_power_w"] *= 1.0 + geometry
        metrics["peaking_db"] = 6.0 * factor
        metrics["dfe_error_count"] = 0.0 if metrics["dfe_min_margin_v"] > 0 else 1.0
    stage = StageResult("synthetic_v3", base.success, 0.0, metrics=metrics)
    return ReceiverEvaluation(base.success, parameters, conditions, fidelity, (stage,), metrics,
        base.failed_stage, 0.0, "synthetic-v3:" + stable_fingerprint(asdict(parameters)),
        {"backend": "synthetic", "not_spice_evidence": True, "channel_applied_to_waveform": False}, False)
