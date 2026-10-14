"""Run one Codex-generated Piper tabletop policy against exact and randomized SIM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from sim.harness import TabletopTask, run_both
from sim.codex_sdk import SimCodexSession
from sim.policy_loop_controller import (
    CodexPolicyLoopController,
    generate_bundle,
    policy_loop_prompt_for,
)
from sim.policy_bundle import PolicyBundle
from sim.policy_refinement import run_policy_refinement


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Source block name")
    parser.add_argument("--target", required=True, help="Target bowl/block name")
    parser.add_argument("--scene-kind", choices=("tabletop", "stage3_emergency"), default="tabletop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path,
                        help="Optional separately authenticated private Codex home; default uses Codex()")
    parser.add_argument("--program", type=Path,
                        help="Wrap an existing policy.py candidate into a policy bundle")
    parser.add_argument("--bundle", type=Path,
                        help="Reuse an existing directory containing policy.py and optional steps/")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="Held-out randomized seed (repeatable)")
    parser.add_argument("--dev-seed", type=int, default=101,
                        help="Randomized development seed used during refinement")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum replacement bundles after the initial candidate")
    parser.add_argument("--no-refinement", action="store_true",
                        help="Run one locked candidate without reset-and-refine")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()
    if (args.codex_home and args.codex_home.expanduser().resolve().is_relative_to(
            (Path.home() / ".codex").resolve())):
        parser.error("--codex-home must not use the global Codex home")
    task = TabletopTask(args.source, args.target,
                        f"Pick up the {args.source} and place it in the {args.target}."
                        if args.scene_kind == "stage3_emergency" else
                        f"Pick up the {args.source} and place it on the {args.target}.",
                        args.scene_kind)
    prompt = policy_loop_prompt_for(task)
    if any((args.output / path).exists() for path in (
            "evaluation.json", "refinement.json", "policy/iteration_000", "final/evaluation.json",
            "authoring_conversation.jsonl", "authoring_thread.json", "final_policy",
            "real_handoff.json")):
        parser.error(f"Refusing to overwrite an existing evaluation: {args.output}")
    if args.max_iterations < 0:
        parser.error("--max-iterations must be non-negative")
    if not args.no_refinement and args.dev_seed in (args.seed or (202,)):
        parser.error("--dev-seed must differ from every held-out --seed")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prompt.txt").write_text(prompt)
    if args.prompt_only:
        print(prompt)
        return
    if args.bundle and args.program:
        parser.error("Use only one of --bundle or --program")
    from perception.config import sam3_endpoint
    host, port = sam3_endpoint()
    transcript = args.output / "authoring_conversation.jsonl"
    transcript.write_text(json.dumps({"type": "task", "prompt": prompt}, ensure_ascii=False) + "\n")
    with SimCodexSession(args.model, ROOT, transcript,
                         codex_home=args.codex_home.expanduser() if args.codex_home else None) as session:
        if args.bundle:
            bundle_root = args.bundle
        elif args.program:
            bundle_root = args.output / "bundle"
            bundle_root.mkdir(parents=True, exist_ok=True)
            (bundle_root / "policy.py").write_text(args.program.read_text())
        else:
            bundle_root, _ = generate_bundle(prompt, session, args.output)
        bundle = PolicyBundle(bundle_root)
        if args.bundle or args.program:
            with transcript.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "supplied_bundle", "manifest": bundle.manifest()},
                                        ensure_ascii=False) + "\n")
            session.bootstrap(prompt)
        thread_manifest = {
            "authoring_thread_id": session.id,
            "codex_home": str(session.codex_home) if session.codex_home else None,
            "model": args.model,
        }
        (args.output / "authoring_thread.json").write_text(
            json.dumps(thread_manifest, ensure_ascii=False, indent=2))
        (args.output / "policy_hash.txt").write_text(bundle.hash + "\n")
        controller = CodexPolicyLoopController(prompt, session, args.output,
                                               generated=not (args.bundle or args.program))
        heldout_seeds = tuple(args.seed) if args.seed else (202,)
        if args.no_refinement:
            report = run_both(args.scene, task, bundle.sources["run"][0], args.output, host,
                              port, heldout_seeds, on_pause=controller, bundle=bundle)
        else:
            report = run_policy_refinement(
                args.scene, task, bundle, args.output, host, port, controller,
                max_iterations=args.max_iterations, dev_seed=args.dev_seed,
                heldout_seeds=heldout_seeds)
            (args.output / "policy_hash.txt").write_text(
                report["candidate_bundle"]["sha256"] + "\n")
            if report["accepted"]:
                latest = report["iterations"][-1]["iteration"]
                bundle = PolicyBundle(args.output / "policy" / f"iteration_{latest:03d}" / "bundle")
            report = {
                "accepted": report["accepted"], "locked": report["locked"],
                "reset_count": report["reset_count"],
                "iterations": [
                    {"iteration": item["iteration"], "policy_hash": item["policy_hash"],
                     "exact": item["exact"]["success"],
                     "randomized": item.get("randomized", {}).get("success")}
                    for item in report["iterations"]
                ],
                "final_evaluation": report["final_evaluation"],
                "report_path": str(args.output / "refinement.json"),
            }
        if report["accepted"]:
            delivery = args.output / "final_policy"
            if delivery.exists():
                raise FileExistsError(f"Refusing to overwrite final policy: {delivery}")
            shutil.copytree(bundle.root, delivery)
            (args.output / "real_handoff.json").write_text(json.dumps({
                "policy": str((delivery / "policy.py").resolve()),
                "bundle": str(delivery.resolve()),
                "policy_hash": bundle.hash,
                "authoring_conversation": str(transcript.resolve()),
                **thread_manifest,
            }, ensure_ascii=False, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
