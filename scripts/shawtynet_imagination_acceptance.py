"""Real local hypothesis → 3D → preview/render/animation acceptance, no downloads.

Writes new immutable projects in the selected workspace (a separate acceptance
workspace by default). Optional saved evidence can seed its three hypotheses.
Never reads photographs or modifies operational measurement results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    from ophanim.core.artifacts import verify_run, write_json
    from ophanim.core.volumes import read_volume
    from ophanim.workspace import Workspace
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=repository / "var" / "imagined-3d-acceptance")
    parser.add_argument("--render", action="store_true", help="Render each of the three 3D model forms with installed Blender")
    parser.add_argument("--animate", action="store_true", help="Render one three-frame evolutionary sequence")
    parser.add_argument("--candidate-id", help="Optional saved candidate in the selected workspace")
    parser.add_argument("--source-project-id", help="Optional observation science project for that same candidate")
    args = parser.parse_args()
    workspace = Workspace(args.output)
    report = {"semantics": "Actual local numerical/renderer integration; imagined hypotheses, not new measurements", "projects": []}

    def stage(kind, payload):
        started = time.monotonic()
        submitted = workspace.submit(kind, payload)
        job_id = submitted["job"]["job_id"]
        last = None
        while time.monotonic() - started < 1500:
            status = workspace.status()
            job = status["job"]
            if job["job_id"] != job_id:
                raise RuntimeError("Unexpected workspace selection change")
            if job["stage"] != last:
                print(json.dumps({"kind": kind, "stage": job["stage"]}), flush=True)
                last = job["stage"]
            if job["state"] not in {"queued", "running", "cancelling"}:
                if job["state"] != "complete":
                    raise RuntimeError(job.get("error") or job["state"])
                return status["projects"][0]
            time.sleep(.2)
        workspace.cancel(job_id)
        raise TimeoutError("Acceptance exceeded its wall-clock budget; cancellation requested")

    try:
        for kind in ("wave_packet", "front", "sheared_jet"):
            payload = {"parameters": {"kind": kind, "seed": 42}, "style": "luminous"}
            if args.candidate_id:
                payload["candidate_id"] = args.candidate_id
            if args.source_project_id:
                payload["source_project_id"] = args.source_project_id
            project = stage("imagine", payload)
            if args.render:
                project = stage("render", {"project_id": project["project_id"]})
            if args.animate and kind == "wave_packet":
                project = stage("animation", {"project_id": project["project_id"], "frame_count": 3})
            saved = workspace.record(project["project_id"], "project")
            volume = read_volume(saved["volume_directory"])
            if volume.dataset.density.dims != ("time", "z", "y", "x"):
                raise AssertionError("The model did not produce a 3D sequence")
            if not args.source_project_id and "science_report_url" in project:
                raise AssertionError("Imagined-only project must not claim a scientific report")
            checks = {}
            for key in ("preview_url", "visual_report_url", "recipe_url", "export_url", "render_url", "animation_url", "animation_export_url"):
                if key not in project:
                    continue
                content, mime = workspace.asset(project[key].removeprefix("/workspace-assets/"))
                if len(content) < 10:
                    raise AssertionError("Empty export")
                checks[key] = {"bytes": len(content), "type": mime}
            for key, artifact_kind in (("volume_directory", "hypothesis"), ("visual_directory", "visual"),
                                       ("render_directory", "render"), ("animation_directory", "animation")):
                if key in saved:
                    verify_run(saved[key], kind=artifact_kind)
            report["projects"].append({"project": project, "artifacts": {key: value for key, value in saved.items() if key.endswith("_directory")},
                                       "shape": dict(volume.dataset.sizes), "exports": checks})
            write_json(args.output / "imagination-acceptance.json", report)
        print(json.dumps({"report": str(args.output / "imagination-acceptance.json"), "projects": len(report["projects"])}), flush=True)
    finally:
        workspace.close(15)


if __name__ == "__main__":
    main()
