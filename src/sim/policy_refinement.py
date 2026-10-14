"""Outer reset-and-refine loop for execution-grounded SIM policies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from .harness import run_both, run_episode
from .policy_terms import LOOP_NAME, POLICY_TYPE


def _public_failure_report(result: dict | None, *, iteration: int, profile: str,
                           seed: int, error: str | None = None) -> dict:
    """Keep evaluator-only coordinates out of the Codex refinement input."""
    if result is None:
        result = {}
    program = result.get("program")
    history = program.get("history", []) if isinstance(program, dict) else []
    last_action = history[-1] if history else None
    report = {
        "iteration": iteration,
        "profile": profile,
        "seed": seed,
        "success": bool(result.get("success", False)),
        "policy_bundle": result.get("policy_bundle"),
        "program": program,
        "calls": result.get("calls", []),
        "failure_stage": (last_action.get("name") if isinstance(last_action, dict)
                           else None),
    }
    if error:
        report["harness_error"] = error
    return json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))


def _snapshot_bundle(bundle, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite policy snapshot: {destination}")
    shutil.copytree(bundle.root, destination)


def _run_development_episode(scene_path, task, bundle, controller, profile, seed,
                             output: Path, host: str, port: int) -> dict:
    return run_episode(scene_path, task, bundle.sources["run"][0], profile, seed,
                       output, host, port, on_pause=controller, bundle=bundle)


def run_policy_refinement(scene_path: Path, task, bundle, output: Path,
                          host: str, port: int, controller, *,
                          max_iterations: int = 3, dev_seed: int = 101,
                          heldout_seeds: tuple[int, ...] = (202,),
                          run_episode_fn=None, run_both_fn=None) -> dict:
    """Reset SIM episodes and refine a bundle in the same Codex conversation.

    ``max_iterations`` is the number of replacements allowed after the initial
    candidate. Every candidate gets a fresh exact episode first and, only when
    that passes, a fresh randomized development episode. Once both development
    profiles pass, the candidate is locked and evaluated on held-out seeds.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    heldout_seeds = tuple(heldout_seeds)
    if not heldout_seeds:
        raise ValueError("At least one held-out randomized seed is required")
    if dev_seed in heldout_seeds:
        raise ValueError("Development seed cannot be a held-out evaluation seed")
    run_episode_fn = run_episode if run_episode_fn is None else run_episode_fn
    run_both_fn = run_both if run_both_fn is None else run_both_fn
    output = Path(output)
    if (output / "refinement.json").exists() or (output / "policy" / "iteration_000").exists():
        raise FileExistsError(f"Refusing to overwrite an existing refinement run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    policy_dir = output / "policy"
    refinement_dir = output / "refinement"
    episodes_dir = output / "episodes"
    policy_dir.mkdir(exist_ok=True)
    refinement_dir.mkdir(exist_ok=True)
    episodes_dir.mkdir(exist_ok=True)

    initial_snapshot = policy_dir / "iteration_000" / "bundle"
    _snapshot_bundle(bundle, initial_snapshot)

    development = []
    failures = []
    refinement_errors = []
    reset_count = 0
    locked = False

    for iteration in range(max_iterations + 1):
        iteration_dir = episodes_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        candidate = {"iteration": iteration, "policy_hash": bundle.hash}
        exact_error = None
        try:
            exact = run_episode_fn(
                scene_path, task, bundle.sources["run"][0], "exact", 0,
                iteration_dir / "exact", host, port, on_pause=controller, bundle=bundle)
        except Exception as exc:
            exact = None
            exact_error = f"{type(exc).__name__}: {exc}"
        reset_count += 1
        candidate["exact"] = exact or {"success": False, "error": exact_error}
        if exact_error or not exact.get("success", False):
            failure = _public_failure_report(exact, iteration=iteration, profile="exact",
                                              seed=0, error=exact_error)
            failures.append(failure)
            (refinement_dir / f"iteration_{iteration:03d}_failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2))
            development.append(candidate)
            failed_profile, failed_seed = "exact", 0
        else:
            randomized_error = None
            try:
                randomized = run_episode_fn(
                    scene_path, task, bundle.sources["run"][0], "randomized", dev_seed,
                    iteration_dir / "randomized" / str(dev_seed), host, port,
                    on_pause=controller, bundle=bundle)
            except Exception as exc:
                randomized = None
                randomized_error = f"{type(exc).__name__}: {exc}"
            reset_count += 1
            candidate["randomized"] = randomized or {
                "success": False, "error": randomized_error,
            }
            if randomized_error or not randomized.get("success", False):
                failure = _public_failure_report(
                    randomized, iteration=iteration, profile="randomized", seed=dev_seed,
                    error=randomized_error)
                failures.append(failure)
                (refinement_dir / f"iteration_{iteration:03d}_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2))
                development.append(candidate)
                failed_profile, failed_seed = "randomized", dev_seed
            else:
                development.append(candidate)
                locked = True
                break

        if iteration >= max_iterations:
            break
        try:
            next_dir = policy_dir / f"iteration_{iteration + 1:03d}"
            if next_dir.exists():
                raise FileExistsError(f"Refusing to overwrite policy iteration: {next_dir}")
            next_root, bundle = controller.refine_bundle(failures[-1], bundle, next_dir)
            if Path(next_root) != bundle.root:
                raise RuntimeError("Refinement controller returned an inconsistent bundle root")
            (next_dir / "manifest.json").write_text(
                json.dumps(bundle.manifest(), ensure_ascii=False, indent=2))
        except Exception as exc:
            error = {"iteration": iteration, "profile": failed_profile,
                     "seed": failed_seed, "error": f"{type(exc).__name__}: {exc}"}
            refinement_errors.append(error)
            (refinement_dir / f"iteration_{iteration:03d}_refinement_error.json").write_text(
                json.dumps(error, ensure_ascii=False, indent=2))
            break

    final = None
    if locked:
        try:
            final = run_both_fn(
                scene_path, task, bundle.sources["run"][0], output / "final",
                host, port, seeds=heldout_seeds, on_pause=controller, bundle=bundle)
        except Exception as exc:
            final = {"accepted": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        reset_count += 1 + len(heldout_seeds)
    report = {
        "accepted": bool(locked and final and final.get("accepted", False)),
        "locked": locked,
        "loop": LOOP_NAME,
        "policy_type": POLICY_TYPE,
        "max_iterations": max_iterations,
        "development_seed": dev_seed,
        "heldout_seeds": list(heldout_seeds),
        "reset_count": reset_count,
        "iterations": development,
        "failures": failures,
        "refinement_errors": refinement_errors,
        "candidate_bundle": bundle.manifest(),
        "final_evaluation": final,
    }
    (output / "refinement.json").write_text(json.dumps(report, ensure_ascii=False,
                                                        indent=2, allow_nan=False))
    return report
