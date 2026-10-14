"""Inspect local model prerequisites without downloading models."""

from __future__ import annotations

import argparse
import importlib.util
import json

from .config import (MODEL_INFO, ROOT, backend_settings, create_sam3, create_vggt,
                     local_metadata, release_backend, require_device)


def check_models(names, *, load=False) -> dict:
    records = {}
    for name in names:
        errors = []
        service = None
        record = {"errors": errors}
        records[name] = record
        try:
            settings = backend_settings(name)
            record.update(backend=settings["backend"])
            if settings["backend"] == "remote":
                record["status"] = "configured_not_contacted"
                if load:
                    raise ValueError("--load checks local weights; select the local backend")
                continue
            checkpoint = settings["checkpoint"]
            record.update(device=settings["device"], checkpoint=str(checkpoint))
            if not checkpoint.is_file():
                errors.append(f"Checkpoint missing: {checkpoint}")
            distribution = MODEL_INFO[name][3]
            import_name = "vggt_omega" if name == "vggt" else distribution
            dependencies = ("torch", "torchvision", import_name)
            if name == "sam3":
                dependencies += ("clip", "timm")
            for dependency in dependencies:
                if importlib.util.find_spec(dependency) is None:
                    errors.append(f"Dependency missing: {dependency}; follow docs/LOCAL_INFERENCE.md")
            source = ROOT / "external" / ("vggt_omega" if name == "vggt" else "ultralytics")
            if not (source / ".git").exists():
                errors.append(f"Pinned source is not initialized: {source}")
            installed = importlib.util.find_spec(import_name)
            if installed is not None and installed.origin:
                from pathlib import Path
                installed_source = Path(installed.origin).resolve()
                record["installed_source"] = str(installed_source)
                if not installed_source.is_relative_to(source.resolve()):
                    errors.append(f"{import_name} is not installed from the pinned editable checkout")
            if importlib.util.find_spec("torch") is not None:
                import torch
                require_device(torch, settings["device"])
                record.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
            if checkpoint.is_file():
                record["model"] = local_metadata(name, checkpoint, settings["device"])
                if record["model"]["source_revision"] != record["model"]["pinned_revision"]:
                    errors.append("Source revision differs from the pinned model revision")
            if load and not errors:
                service = (create_vggt if name == "vggt" else create_sam3)()
                service.load()
                record["model"] = service.metadata
                record["loaded"] = True
            record["status"] = "failed" if errors else "ready"
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            record["status"] = "failed"
        finally:
            release_backend(service)
    return {"passed": all(not record["errors"] for record in records.values()),
            "models": records, "inference_verified": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check dependencies, weights, source revision and device")
    check.add_argument("--models", nargs="+", choices=("vggt", "sam3"), default=["vggt", "sam3"])
    check.add_argument("--load", action="store_true", help="also load each local model, then release it")
    args = parser.parse_args(argv)
    report = check_models(args.models, load=args.load)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
