"""Isolated Chrome/CDP UI smoke; no models, downloads, renders, or personal profile.

Uses Python's standard library and actual repository web assets. Workspace fetches
are intercepted inside the disposable browser to inspect form serialization.
On WSL, checks run through Windows-local Python and use its native temporary
directory by default. Results describe UI wiring, not scientific backend success.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.dont_write_bytecode = True


class CDP:
    def __init__(self, url):
        parsed = urlparse(url)
        if parsed.scheme != "ws" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("CDP checker only connects to a local isolated browser")
        self.socket = socket.create_connection((parsed.hostname, parsed.port), timeout=15)
        self.buffer = b""
        self.counter = 0
        self.events = []
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        request = (f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\n"
                   f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n")
        self.socket.sendall(request.encode())
        while b"\r\n\r\n" not in self.buffer:
            self.buffer += self.socket.recv(65536)
        headers, self.buffer = self.buffer.split(b"\r\n\r\n", 1)
        if b" 101 " not in headers.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket handshake failed: {headers[:200]!r}")

    def _read(self, count):
        while len(self.buffer) < count:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise EOFError("Chrome debugger disconnected")
            self.buffer += chunk
        result, self.buffer = self.buffer[:count], self.buffer[count:]
        return result

    def _send(self, content, opcode=1):
        mask = secrets.token_bytes(4)
        length = len(content)
        header = bytes([0x80 | opcode])
        if length < 126:
            header += bytes([0x80 | length])
        elif length <= 65535:
            header += bytes([0x80 | 126]) + struct.pack("!H", length)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", length)
        self.socket.sendall(header + mask + bytes(value ^ mask[i % 4] for i, value in enumerate(content)))

    def _receive(self):
        message = b""
        while True:
            first, second = self._read(2)
            opcode, length = first & 15, second & 127
            if length == 126:
                length = struct.unpack("!H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read(8))[0]
            mask = self._read(4) if second & 128 else None
            body = self._read(length)
            if mask:
                body = bytes(value ^ mask[i % 4] for i, value in enumerate(body))
            if opcode == 9:
                self._send(body, 10)
                continue
            if opcode == 8:
                raise EOFError("Chrome closed its debugger")
            if opcode in (0, 1):
                message += body
                if first & 128:
                    return json.loads(message)

    def call(self, method, parameters=None):
        self.counter += 1
        identity = self.counter
        self._send(json.dumps({"id": identity, "method": method, "params": parameters or {}}).encode())
        while True:
            response = self._receive()
            if response.get("id") == identity:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})
            self.events.append(response)

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")


MOCK = r"""
(() => {
  window.__smokeErrors = []; window.__smokeRequests = [];
  addEventListener('error', e => window.__smokeErrors.push(e.message));
  addEventListener('unhandledrejection', e => window.__smokeErrors.push(String(e.reason)));
  const realFetch = window.fetch.bind(window);
  const projectId = 'a'.repeat(24);
  const status = {ok:true, available:true, capabilities:{core:true,blender:true},
    projects:[],scans:[],scenarios:[],last_scan:null,job:null};
  window.fetch = async (input, options={}) => {
    const path = typeof input === 'string' ? input : input.url;
    if (!path.startsWith('/api/workspace')) return realFetch(input,options);
    if ((options.method || 'GET') === 'POST') {
      let body = null;
      if (typeof options.body === 'string') body = JSON.parse(options.body);
      window.__smokeRequests.push({path, body, content_type:options.headers?.['Content-Type'],
        has_auth_header:Boolean(options.headers?.['X-OPHANIM-Token']),
        binary:options.body instanceof Blob ? {size:options.body.size,type:options.body.type}:null});
      if (path.endsWith('/photo')) {
        // A user can change selection while an upload is in flight. The posted
        // composite must retain its original target and manually chosen values.
        const picker=document.getElementById('work-projects');
        picker.value='e'.repeat(24); picker.dispatchEvent(new Event('change',{bubbles:true}));
        document.getElementById('work-photo-opacity').value='.9';
        await new Promise(resolve=>setTimeout(resolve,30));
        return Response.json({ok:true,photo:{photo_id:'b'.repeat(24),width:2,height:2}});
      }
      if (path.endsWith('/mask')) return Response.json({ok:true,mask:{mask_id:'c'.repeat(24),width:2,height:2}});
      if (path.endsWith('/imagine')) status.projects = [{project_id:projectId,kind:'imagined_3d',
        scenario_id:'d'.repeat(24),title:'Imagined UI fixture',parameters:{kind:'wave_packet',frame_count:9,duration_minutes:55,altitude_min_km:210,altitude_max_km:410,spacing_km:25,...body.parameters},
        camera:body.camera,style:body.style,active_view:'preview',
        preview_url:`/workspace-assets/${projectId}/preview.png`}];
      if (path.endsWith('/scenario')) status.projects = [{project_id:projectId, kind:'hypothetical',
        scenario_id:'d'.repeat(24),title:'Isolated UI fixture',parameters:body.parameters,camera:body.camera,style:body.style,
        render_url:`/workspace-assets/${projectId}/render.png`,
        analysis_summary:{flow_status:'low_confidence',wave_status:'insufficient_samples',timeline_frames:12}}];
      if (path.endsWith('/scenario')) status.projects.push(
        {...status.projects[0],project_id:'e'.repeat(24),title:'Null-camera fixture',camera:null},
        {...status.projects[0],project_id:'f'.repeat(24),title:'Partial-camera fixture',camera:{heading_deg:5}});
      if (path.endsWith('/style')) status.projects[0] = {...status.projects[0],camera:body.camera,style:body.style};
      if (path.endsWith('/composite')) status.projects[0] = {...status.projects[0],
        composite_url:`/workspace-assets/${projectId}/composite.png`,active_view:'composite'};
      if (path.endsWith('/animation')) status.projects[0] = {...status.projects[0],
        animation_url:`/workspace-assets/${projectId}/animation.gif`,active_view:'animation'};
      status.job = {job_id:String(window.__smokeRequests.length),kind:'ui_test',state:'complete',stage:'complete',message:'UI fixture only'};
    }
    return Response.json(status);
  };
})();
"""


def wait_for(predicate, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(.1)
    raise TimeoutError("Isolated Chrome check did not become ready")


def browser_checks(client):
    results = {}
    wait_for(lambda: client.evaluate("document.readyState==='complete' && !!document.getElementById('work-scenario-form')"))
    results["default_form_valid"] = client.evaluate("document.getElementById('work-scenario-form').checkValidity()")
    results["layouts"] = []
    for width in (1365, 390):
        client.call("Emulation.setDeviceMetricsOverride", {"width":width,"height":900,"deviceScaleFactor":1,"mobile":False})
        for view in ("discover", "event", "studio", "experiments", "advanced"):
            client.evaluate(f"document.getElementById('tab-{view}').click()")
            layout = client.evaluate("({viewport:innerWidth,scroll:document.documentElement.scrollWidth,active:location.hash})")
            results["layouts"].append({"width":width,"view":view,**layout})
    client.call("Emulation.setDeviceMetricsOverride", {"width":1365,"height":900,"deviceScaleFactor":1,"mobile":False})
    client.evaluate("document.getElementById('tab-studio').click()")
    results["imagination_default"] = client.evaluate("({mode:document.getElementById('work-model-mode').value,legacy_disabled:document.getElementById('work-2d-fields').disabled,automatic:document.getElementById('work-camera-auto').checked})")
    client.evaluate("document.getElementById('work-scenario-form').requestSubmit()")
    wait_for(lambda: client.evaluate("window.__smokeRequests.some(r=>r.path.endsWith('/imagine')) && !document.getElementById('work-restyle').disabled"))
    results["imagination_saved_clean"] = client.evaluate("!document.getElementById('work-render').disabled && document.getElementById('work-project-kind').textContent==='IMAGINED 3D / ART'")
    client.evaluate("document.getElementById('work-imagine-seed').value='43'; document.getElementById('work-scenario-form').requestSubmit()")
    wait_for(lambda: client.evaluate("window.__smokeRequests.filter(r=>r.path.endsWith('/imagine')).length===2 && !document.getElementById('work-restyle').disabled"))
    # Exercise the retained 2D path independently, without submitting hidden 3D controls.
    client.evaluate("document.getElementById('work-model-mode').value='scenario'; document.getElementById('work-model-mode').dispatchEvent(new Event('change')); document.getElementById('work-style').value='ghost'")
    results["cross_mode_camera_reset"] = client.evaluate("Number(document.getElementById('work-camera-pitch').value)===90 && Number(document.getElementById('work-camera-heading').value)===0 && Number(document.getElementById('work-camera-fov').value)===70")
    camera = {"latitude":31.25,"longitude":-97.5,"altitude":.25,"heading":42,"pitch":50,"roll":-4,"fov":80}
    client.evaluate("Object.entries(" + json.dumps(camera) + ").forEach(([key,value])=>document.getElementById('work-camera-'+key).value=String(value))")
    results["camera_form_valid"] = client.evaluate("document.getElementById('work-scenario-form').checkValidity()")
    client.evaluate("document.getElementById('work-scenario-form').requestSubmit()")
    wait_for(lambda: client.evaluate("window.__smokeRequests.some(r=>r.path.endsWith('/scenario'))"))
    wait_for(lambda: client.evaluate("!document.getElementById('work-restyle').disabled"))
    # Loading a saved null/partial camera must reproduce export defaults, not
    # silently dirty a saved project because the form uses a different pitch.
    results["saved_camera_defaults"] = client.evaluate(r"""(() => {
      const select=document.getElementById('work-projects'); const found={};
      for (const [name,id] of [['null','e'],['partial','f']]) {
        select.value=id.repeat(24); select.dispatchEvent(new Event('change',{bubbles:true}));
        found[name]={pitch:Number(document.getElementById('work-camera-pitch').value),
          render_enabled:!document.getElementById('work-render').disabled,
          animate_enabled:!document.getElementById('work-animate').disabled};
      }
      select.value='a'.repeat(24); select.dispatchEvent(new Event('change',{bubbles:true}));
      return found;
    })()""")
    results["dirty_guards"] = []
    for field, value in (("camera-heading", "43"), ("style", "luminous")):
        client.evaluate("(() => {const input=document.getElementById('work-' + " + json.dumps(field)
                        + "); input.value=" + json.dumps(value) + "; input.dispatchEvent(new Event('input',{bubbles:true}));})()")
        before = client.evaluate("window.__smokeRequests.length")
        guard = client.evaluate("({render_disabled:document.getElementById('work-render').disabled,"
                                "animate_disabled:document.getElementById('work-animate').disabled,"
                                "note:document.getElementById('work-render-note').textContent})")
        client.evaluate("document.getElementById('work-render').click();document.getElementById('work-animate').click()")
        guard["no_unsaved_request"] = client.evaluate("window.__smokeRequests.length") == before
        client.evaluate("document.getElementById('work-restyle').click()")
        wait_for(lambda: client.evaluate("!document.getElementById('work-render').disabled && !document.getElementById('work-animate').disabled"))
        guard["saved_settings_reenabled"] = True
        guard["field"] = field
        results["dirty_guards"].append(guard)
    # Actual File/DataTransfer plumbing exercises async photo and mask handlers.
    client.evaluate(r"""(() => {
      for (const id of ['photo','foreground-mask','cloud-mask']) {
        const transfer = new DataTransfer(); transfer.items.add(new File([new Uint8Array([1,2,3])], id+'.png',{type:'image/png'}));
        const input=document.getElementById('work-'+id); input.files=transfer.files; input.dispatchEvent(new Event('change',{bubbles:true}));
      }
      for (const [key,value] of Object.entries({opacity:.4,horizon:.7,fade:.3,saturation:.9,exposure:1.2,red:1.1,green:.9,blue:.8}))
        document.getElementById('work-photo-'+key).value=String(value);
    })()""")
    wait_for(lambda: client.evaluate("!document.getElementById('work-composite').disabled"))
    client.evaluate("document.getElementById('work-composite').click()")
    wait_for(lambda: client.evaluate("window.__smokeRequests.some(r=>r.path.endsWith('/composite'))"))
    wait_for(lambda: client.evaluate("document.getElementById('work-preview-image').getAttribute('src')?.endsWith('/composite.png')"))
    wait_for(lambda: client.evaluate("!document.getElementById('work-animate').disabled"))
    client.evaluate("document.getElementById('work-animate').click()")
    wait_for(lambda: client.evaluate("window.__smokeRequests.some(r=>r.path.endsWith('/animation'))"))
    wait_for(lambda: client.evaluate("document.getElementById('work-preview-image').getAttribute('src')?.endsWith('/animation.gif')"))
    results["animation_active_view"] = client.evaluate("({image:document.getElementById('work-preview-image').getAttribute('src'),"
        "caption:document.getElementById('work-preview-caption').textContent,"
        "old_composite_retained:[...document.querySelectorAll('#work-project-links a')].some(a=>a.href.endsWith('/composite.png'))})")
    results["requests"] = client.evaluate("window.__smokeRequests")
    results["javascript_errors"] = client.evaluate("window.__smokeErrors")
    results["runtime_exceptions"] = [event["params"].get("exceptionDetails", {}) for event in client.events if event.get("method") == "Runtime.exceptionThrown"]
    results["visible_error"] = client.evaluate("document.getElementById('work-error').hidden ? null : document.getElementById('work-error').textContent")
    results["semantics"] = "Actual isolated Chrome UI; workspace HTTP responses and upload receipts mocked. No scientific jobs or external downloads."
    expected_camera = {"observer_latitude":31.25,"observer_longitude":-97.5,"observer_altitude_km":.25,
                       "heading_deg":42,"pitch_deg":50,"roll_deg":-4,"horizontal_fov_deg":80}
    scenario = next(item["body"] for item in results["requests"] if item["path"].endswith("/scenario"))
    imagination = next(item["body"] for item in results["requests"] if item["path"].endswith("/imagine"))
    revision = [item["body"] for item in results["requests"] if item["path"].endswith("/imagine")][1]
    results["saved_imagination_settings_preserved"] = all(revision["parameters"].get(key) == value for key,value in
        {"seed":43,"frame_count":9,"duration_minutes":55,"altitude_min_km":210,"altitude_max_km":410,"spacing_km":25}.items())
    results["imagination_serialization_correct"] = (imagination["camera"] is None and imagination["style"] == "luminous"
        and imagination["parameters"] == {"seed":42} and results["imagination_default"] == {"mode":"imagined_3d","legacy_disabled":True,"automatic":True})
    composite = next(item["body"] for item in results["requests"] if item["path"].endswith("/composite"))
    results["camera_serialization_correct"] = scenario["camera"] == expected_camera
    results["photo_serialization_correct"] = (composite.get("project_id") == "a"*24 and composite.get("opacity") == .4
                                               and composite.get("horizon_y") == .7 and composite.get("grade_rgb") == [1.1,.9,.8]
                                               and composite.get("foreground_mask_id") == "c"*24 and composite.get("cloud_mask_id") == "c"*24)
    defaults = results["saved_camera_defaults"]
    results["saved_camera_defaults_correct"] = (defaults["null"]["pitch"] == 90 and defaults["partial"]["pitch"] == 45
        and all(item["render_enabled"] and item["animate_enabled"] for item in defaults.values()))
    results["dirty_settings_guard_correct"] = all(item["render_disabled"] and item["animate_disabled"] and "Apply style & camera" in item["note"]
        and item["no_unsaved_request"] and item["saved_settings_reenabled"] for item in results["dirty_guards"])
    view = results["animation_active_view"]
    results["active_view_correct"] = view["image"].endswith("/animation.gif") and "Short artistic animation" in view["caption"] and view["old_composite_retained"]
    results["passed"] = (results["cross_mode_camera_reset"] and results["saved_imagination_settings_preserved"] and results["imagination_serialization_correct"] and results["imagination_saved_clean"]
                         and results["default_form_valid"] and results["camera_form_valid"] and results["camera_serialization_correct"]
                         and results["photo_serialization_correct"] and not results["javascript_errors"] and not results["runtime_exceptions"]
                         and results["saved_camera_defaults_correct"] and results["dirty_settings_guard_correct"] and results["active_view_correct"]
                         and not results["visible_error"] and all(item["scroll"] <= item["viewport"] for item in results["layouts"]))
    return results


def live_project_check(client, project_id, screenshot=None):
    """Read-only check of one real saved project's UI and actual served image."""
    if len(project_id) != 24 or any(char not in "0123456789abcdef" for char in project_id):
        raise ValueError("Live project must be a saved 24-character workspace identifier")
    wait_for(lambda: client.evaluate("document.readyState==='complete' && !!document.getElementById('work-projects')"))
    wait_for(lambda: client.evaluate("[...document.getElementById('work-projects').options].some(option=>option.value===" + json.dumps(project_id) + ")"))
    client.call("Emulation.setDeviceMetricsOverride", {"width":1365,"height":1000,"deviceScaleFactor":1,"mobile":False})
    client.evaluate("document.getElementById('tab-studio').click(); const picker=document.getElementById('work-projects'); picker.value=" + json.dumps(project_id) + "; picker.dispatchEvent(new Event('change',{bubbles:true}))")
    wait_for(lambda: client.evaluate("document.getElementById('work-preview-image').complete && document.getElementById('work-preview-image').naturalWidth>0"))
    result = client.evaluate("({title:document.getElementById('work-preview-title').textContent,kind:document.getElementById('work-project-kind').textContent,caption:document.getElementById('work-preview-caption').textContent,image:document.getElementById('work-preview-image').getAttribute('src'),width:document.getElementById('work-preview-image').naturalWidth,mode:document.getElementById('work-model-mode').value,error:document.getElementById('work-error').hidden?null:document.getElementById('work-error').textContent,overflow:document.documentElement.scrollWidth>innerWidth})")
    result["semantics"] = "Read-only actual Chrome check against the running local application, no mocked workspace responses or submitted jobs"
    result["passed"] = (result["kind"] == "IMAGINED 3D / ART" and result["mode"] == "imagined_3d"
                        and result["width"] > 0 and not result["error"] and not result["overflow"])
    if screenshot:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(base64.b64decode(client.call("Page.captureScreenshot", {"format":"png"})["data"]))
        result["screenshot"] = str(screenshot)
    return result


@contextmanager
def application_url(existing=None):
    if existing:
        parsed = urlparse(existing)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("Only the isolated loopback test application is allowed")
        yield existing
        return
    from ophanim.desktop import DesktopController, OphanimHTTPServer
    with tempfile.TemporaryDirectory(prefix="ophanim-ui-smoke-data-") as data:
        controller = DesktopController(data_directory=data)
        server = OphanimHTTPServer(("127.0.0.1", 0), controller=controller, token=secrets.token_urlsafe(32))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/"
        finally:
            server.shutdown()
            server.server_close()
            controller.close()
            worker.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")
    parser.add_argument("--profile-parent", help="Temporary browser-profile parent; defaults to the browser host's temporary directory")
    parser.add_argument("--windows-python", default="/mnt/c/Program Files/Blender Foundation/Blender 5.1/5.1/python/bin/python.exe",
                        help="WSL runs the stdlib-only check Windows-local to reach Chrome's private debugger")
    parser.add_argument("--url", help=argparse.SUPPRESS)
    parser.add_argument("--live-project", help="Read-only verification of a real saved project; requires --url")
    parser.add_argument("--screenshot", type=Path, help="Optional live-project screenshot artifact")
    arguments = parser.parse_args()
    if arguments.live_project and not arguments.url:
        parser.error("--live-project requires the running local application's --url")
    if os.name != "nt" and arguments.chrome.endswith(".exe"):
        if not Path(arguments.windows_python).is_file():
            raise SystemExit("Windows Chrome requires a Windows-local Python. Supply --windows-python; no network relay is created.")
        translate = lambda path: subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        with application_url(arguments.url) as url:
            command = [arguments.windows_python, translate(Path(__file__).resolve()),
                                    "--chrome", translate(arguments.chrome), "--url", url]
            if arguments.profile_parent:
                command.extend(["--profile-parent", translate(arguments.profile_parent)])
            if arguments.live_project:
                command.extend(["--live-project", arguments.live_project])
            if arguments.screenshot:
                command.extend(["--screenshot", translate(arguments.screenshot.resolve())])
            return subprocess.call(command)

    with ExitStack() as resources:
        url = resources.enter_context(application_url(arguments.url))
        profile = resources.enter_context(tempfile.TemporaryDirectory(prefix="ophanim-ui-smoke-profile-", dir=arguments.profile_parent))
        client = None
        process = None
        try:
            profile_argument = profile
            process = subprocess.Popen([arguments.chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--disable-sync", "--disable-background-networking", "--remote-debugging-port=0", "--user-data-dir="+profile_argument, "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            active = Path(profile) / "DevToolsActivePort"
            wait_for(active.is_file)
            port = int(active.read_text().splitlines()[0])
            targets = wait_for(lambda: json.load(urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2)))
            target = next(item for item in targets if item["type"] == "page")
            client = CDP(target["webSocketDebuggerUrl"])
            client.call("Runtime.enable")
            client.call("Page.enable")
            if not arguments.live_project:
                client.call("Page.addScriptToEvaluateOnNewDocument", {"source":MOCK})
            client.call("Page.navigate", {"url":url})
            result = live_project_check(client, arguments.live_project, arguments.screenshot) if arguments.live_project else browser_checks(client)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        finally:
            if client is not None:
                try:
                    client.call("Browser.close")
                except (OSError, EOFError):
                    pass
                client.socket.close()
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
