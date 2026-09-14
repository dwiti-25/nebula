"""Two independent episodes with serial policy sampling and concurrent SPICE.

Each batch partitions the remaining evaluation allowance before submission.
Trajectories remain contiguous for GAE; deadlines propagate into worker contexts.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from simulator.rl_adapter import ReceiverRLAdapter, RLBudget
from .autockt_env import AutoCktReceiverEnv
from .ppo_agent import Transition
from .events import build_step_event


def collect_parallel(env, agent, *, episodes, episode_index_start=0,
                     on_step=None, on_event=None, run_id="", update_index=0):
    from .trainer import EpisodeLog
    transitions, logs = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, episodes, 2):
            if env.budget_exhausted():
                break
            base_count = env.adapter.total_evaluations
            remaining = min(env.max_total_evaluations, env.adapter.budget.max_evaluations) - base_count
            count = min(2, episodes - start, remaining)
            workers = []
            for i in range(count):
                allowance = remaining // count + int(i < remaining % count)
                adapter = ReceiverRLAdapter(evaluator=env.adapter.evaluator,
                    conditions=env.adapter.conditions, fidelity=env.adapter.fidelity,
                    budget=RLBudget(allowance), seed=env.adapter.seed_value,
                    evaluator_kwargs=env.adapter.evaluator_kwargs, version=env.adapter.version)
                worker = AutoCktReceiverEnv(target_pool=env.target_pool,
                    initial_indices=env.initial_indices, horizon=env.horizon, adapter=adapter,
                    grids=env.grids, seed=env._episode_rng.randrange(2**32),
                    randomize_initial_state=env.randomize_initial_state, reward_fn=env.reward_fn,
                    evaluate_on_reset=env.evaluate_on_reset, state_schema=env.state_schema,
                    use_reward_v2=env.use_reward_v2, action_deltas=env.action_deltas)
                workers.append(worker)
            pending = [pool.submit(copy_context().run, worker.reset) for worker in workers]
            resets = [future.result() for future in pending]
            states = [result[0] for result in resets]
            batch_logs = [EpisodeLog(episode_index_start + start + i, result[1]["target"])
                          for i, result in enumerate(resets)]
            trajectories = [[] for _ in workers]
            active = [i for i, worker in enumerate(workers) if not worker.budget_exhausted()]
            while active:
                active = [i for i in active if not workers[i].budget_exhausted()]
                if not active:
                    break
                actions = {i: agent.act(states[i]) for i in active}
                before = {i: workers[i].indices for i in active}
                pending = {i: pool.submit(copy_context().run, workers[i].step, actions[i][0]) for i in active}
                results = {i: pending[i].result() for i in active}
                total = base_count + sum(w.adapter.total_evaluations for w in workers)
                next_active = []
                for i in active:
                    step, worker, log = results[i], workers[i], batch_logs[i]
                    if step.info["parameters"] is None:
                        continue  # Deadline elapsed between submission and execution.
                    choices, log_prob, value = actions[i]
                    bootstrap = agent.act(step.state)[2] if step.truncated and not step.done else 0.0
                    trajectories[i].append(Transition(states[i], choices, log_prob, value,
                        step.reward, step.done, step.truncated, bootstrap))
                    row = {"episode": log.episode_index, "step": step.info["step_count"],
                        "total_evaluation_count": total, "reward": step.reward,
                        "success": step.info["success"], "failure_stage": step.info["failure_stage"],
                        "spec_satisfied": step.info["spec_satisfied"],
                        "parameters": step.info["parameters"], "indices": step.info["indices"],
                        "target": log.target}
                    log.steps.append(row)
                    log.episode_reward += step.reward
                    log.spec_satisfied = step.done
                    if on_step:
                        on_step(row)
                    if on_event:
                        reward = step.info.get("reward_result")
                        on_event(build_step_event(run_id=run_id, step=row["step"],
                            episode=log.episode_index, update=update_index,
                            target_id=str(log.target), target_values=log.target,
                            state_before=states[i], state_after=step.state,
                            metrics_valid_mask=step.info["metrics_valid_mask"],
                            action_choice=choices, action_deltas=worker.action_deltas,
                            indices_before=before[i], indices_after=worker.indices,
                            parameters_before={k: worker.grids[k].value_at(v) for k, v in zip(worker.parameter_names, before[i])},
                            parameters_after=row["parameters"], raw_metrics=step.info["metrics"],
                            reward_components=reward.components if reward else {}, reward_total=step.reward,
                            strict_pass=reward.strict_target_pass if reward else None,
                            failure_stage=row["failure_stage"], log_prob=log_prob, value_estimate=value,
                            done=step.done, truncated=step.truncated, evaluation_count=total))
                    states[i] = step.state
                    if not step.done and not step.truncated:
                        next_active.append(i)
                active = next_active
            env.adapter.total_evaluations = base_count + sum(w.adapter.total_evaluations for w in workers)
            for trajectory, log in zip(trajectories, batch_logs):
                # An interrupted partial trajectory is excluded from learning.
                if trajectory and (trajectory[-1].terminated or trajectory[-1].truncated):
                    transitions.extend(trajectory)
                    logs.append(log)
    return transitions, 0.0, logs
