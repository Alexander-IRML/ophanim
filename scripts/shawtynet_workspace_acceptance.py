"""Exercise a running local workspace without downloading or training anything.

Creates new immutable acceptance projects, leaving existing projects untouched.
Uses one saved native candidate if available, then a bounded hypothetical front,
a still, a three-frame animation and a manually masked local photo composite.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    from ophanim.core.artifacts import write_json
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://127.0.0.1:8765")
    parser.add_argument("--foreground-mask",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=repository/"var"/"shawtynet-v1-workflow.json")
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in {"127.0.0.1","localhost","::1"}:
        raise ValueError("Acceptance runner only operates against a local application")
    class TokenParser(HTMLParser):
        token = None
        def handle_starttag(self,tag,attrs):
            values = dict(attrs)
            if tag == "meta" and values.get("name") == "ophanim-token":
                self.token = values.get("content")
    with urlopen(args.url,timeout=10) as response:
        page = response.read().decode()
    document = TokenParser()
    document.feed(page)
    if not document.token:
        raise ValueError("Local application did not provide its mutation token")
    def get(path):
        with urlopen(args.url+path,timeout=30) as response:
            return json.load(response)
    def post(path,payload=None,content=None):
        data = content if content is not None else json.dumps(payload).encode()
        headers = {"Content-Type":"application/octet-stream" if content is not None else "application/json",
                   "X-OPHANIM-Token":document.token}
        with urlopen(Request(args.url+"/api/workspace/"+path,data=data,headers=headers),timeout=30) as response:
            return json.load(response)
    summary = {"stages":[],"semantics":"Local acceptance artifacts; hypothetical front and artistic photo, no new observations or training"}
    def stage(kind,payload):
        state = get("/api/workspace")
        if state.get("job",{}).get("state") in {"queued","running","cancelling"}:
            raise RuntimeError("Another workspace job is active; acceptance will not interrupt it")
        started = time.monotonic()
        job = post(kind,payload)["job"]["job_id"]
        last = None
        while time.monotonic()-started < 600:
            state = get("/api/workspace")
            current = state["job"]
            if current["job_id"] != job:
                raise RuntimeError("Workspace selection changed during acceptance")
            if current["stage"] != last:
                print(json.dumps({"kind":kind,"stage":current["stage"]}),flush=True)
                last = current["stage"]
            if current["state"] not in {"queued","running","cancelling"}:
                if current["state"] != "complete":
                    raise RuntimeError(current.get("error") or current["state"])
                project = state["projects"][0]
                summary["stages"].append({"kind":kind,"job_id":job,"elapsed_seconds":round(time.monotonic()-started,2),"project":project})
                return project
            time.sleep(.25)
        raise TimeoutError("Acceptance job exceeded its local time budget; inspect the application")
    state = get("/api/workspace")
    candidates = (state.get("last_scan") or {}).get("candidates",[])
    if candidates:
        stage("artify",{"candidate_id":candidates[0]["candidate_id"],"style":"ghost"})
    project = stage("scenario",{"parameters":{"kind":"moving_front","amplitude_tecu":5,"width_km":50,
                         "speed_m_s":20,"duration_minutes":240,"extent_km":1000,"spacing_km":25},
                         "style":"ghost","camera":{"heading_deg":0,"pitch_deg":70,"roll_deg":8,
                         "horizontal_fov_deg":75,"observer_latitude":30,"observer_longitude":-98,"observer_altitude_km":.2}})
    project = stage("render",{"project_id":project["project_id"]})
    project = stage("animation",{"project_id":project["project_id"],"frame_count":3})
    photo = post("photo",content=(repository/"examples/shawtynet/sunset-skies-nps.jpg").read_bytes())["photo"]
    mask = post("mask",content=args.foreground_mask.read_bytes())["mask"]
    project = stage("composite",{"project_id":project["project_id"],"photo_id":photo["photo_id"],
                    "foreground_mask_id":mask["mask_id"],"opacity":.55,"horizon_y":.52,"horizon_fade":.24,
                    "saturation":.75,"exposure":1.1,"grade_rgb":[1,.98,1.04]})
    summary["downloads"] = {}
    for key in ("science_report_url","visual_report_url","preview_url","render_url","composite_url","animation_url","animation_export_url","export_url","recipe_url"):
        with urlopen(args.url+project[key],timeout=30) as response:
            size = len(response.read())
            if size < 10:
                raise AssertionError("Empty exported artifact")
            summary["downloads"][key] = {"bytes":size,"content_type":response.headers.get("Content-Type")}
    write_json(args.output,summary)
    print(json.dumps({"report":str(args.output),"final_project_id":project["project_id"]}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
