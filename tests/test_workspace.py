"""Durable user journeys, resource coordination, and authenticated workspace API."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ophanim.core.jobs import WorkCancelled, heavy_work
from ophanim.workspace import Workspace, WorkspaceBusy, WorkspaceUnavailable, _camera_payload


def finished(workspace, timeout=15):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        status = workspace.status()
        if status["job"]["state"] not in ("queued", "running", "cancelling"):
            return status
        time.sleep(.01)
    raise AssertionError("Workspace job did not finish within the test budget")


class WorkspaceLifecycleTests(unittest.TestCase):
    def test_manual_camera_controls_are_bounded_and_coordinates_optional(self):
        values = {"observer_latitude":30.25,"observer_longitude":-98.5,"observer_altitude_km":.25,
                  "heading_deg":90,"pitch_deg":30,"roll_deg":-12,"horizontal_fov_deg":70}
        self.assertEqual(_camera_payload(values), values)
        self.assertEqual(_camera_payload({"roll_deg":0}), {"roll_deg":0})
        for invalid in ({"observer_latitude":91}, {"roll_deg":181}, {"pitch_deg":True},
                        {"observer_longitude":float("nan")}, {"filename":"camera.json"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _camera_payload(invalid)

    def test_photo_controls_and_mask_size_are_validated_before_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda:True)
            try:
                project = workspace._save("project", {"render_directory":"test-render"})
                photo = workspace._save("photo", {"width":100,"height":80,"directory":"test-photo"})
                mask = workspace._save("mask", {"width":50,"height":80,"directory":"test-mask"})
                payload = {"project_id":project["project_id"],"photo_id":photo["photo_id"]}
                validated = workspace._validate("composite", {**payload,"horizon_y":.42,"exposure":1.2,"grade_rgb":[1,.8,1.2]})
                self.assertEqual(validated["horizon_y"], .42)
                self.assertEqual(validated["saturation"], .85)
                for changes in ({"horizon_y":0}, {"exposure":5}, {"grade_rgb":[1,1]}, {"saturation":True},
                                {"foreground_mask_id":mask["mask_id"]}, {"cloud_mask_directory":"/private"}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        workspace._validate("composite", {**payload,**changes})
                self.assertIsNone(workspace.status()["job"])
            finally:
                workspace.close()

    def test_background_job_preserves_result_and_restart_history(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            def runner(kind, request, progress):
                entered.set()
                release.wait(3)
                workspace._save("scan", {"status":"complete", "candidates":[], "source":{}})
            workspace = Workspace(directory, runner=runner, availability=lambda: True)
            try:
                workspace.submit("scan", {"source":"cached"})
                self.assertTrue(entered.wait(2))
                with self.assertRaises(WorkspaceBusy):
                    workspace.submit("scan", {"source":"cached"})
                release.set()
                result = finished(workspace)
                self.assertEqual(result["job"]["state"], "complete")
                identity = result["last_scan"]["scan_id"]
            finally:
                release.set()
                workspace.close(5)
            restored = Workspace(directory, availability=lambda: True)
            try:
                self.assertEqual(restored.status()["last_scan"]["scan_id"],identity)
                self.assertEqual(restored.status()["job"]["state"],"complete")
            finally:
                restored.close()

    def test_cancel_and_failure_do_not_erase_last_success(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            def runner(kind, request, progress):
                entered.set()
                release.wait(3)
                progress("test", "safe cancellation boundary")
            workspace = Workspace(directory, runner=runner, availability=lambda: True)
            try:
                saved = workspace._save("scan",{"status":"complete","candidates":[]})
                workspace.submit("scan",{"source":"cached"})
                entered.wait(2)
                workspace.cancel()
                release.set()
                self.assertEqual(finished(workspace)["job"]["state"],"cancelled")
                def fails(*args):
                    raise ValueError("source unavailable")
                workspace._runner=fails
                workspace.submit("scan",{"source":"cached"})
                status=finished(workspace)
                self.assertEqual(status["job"]["state"],"failed")
                self.assertIn("source unavailable",status["job"]["error"])
                self.assertEqual(status["last_scan"]["scan_id"],saved["scan_id"])
            finally:
                release.set()
                workspace.close(5)

    def test_restart_marks_unfinished_job_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Workspace(directory,availability=lambda:True)
            with workspace._connection() as connection:
                connection.execute("INSERT INTO jobs VALUES ('aaaaaaaaaaaaaaaaaaaaaaaa','scan','running','acquiring','working',0,'{}','2024-01-01',NULL,NULL)")
                connection.commit()
            workspace.close()
            restored=Workspace(directory,availability=lambda:True)
            try:
                self.assertEqual(restored.status()["job"]["state"],"interrupted")
            finally:
                restored.close()

    def test_validation_and_asset_paths_fail_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Workspace(directory,availability=lambda:True)
            try:
                for kind, request in (("scan",{"source":"url","url":"https://example.org"}),
                                      ("scan",{"days":True}), ("scan",{"days":1000}),
                                      ("render",{"project_id":"../../etc/passwd"}),
                                      ("artify",{}), ("style",{"project_id":"0"*24,"executable":"/bin/sh"})):
                    with self.subTest(kind=kind,request=request),self.assertRaises(ValueError):
                        workspace.submit(kind,request)
                for value in ("../private", "0"*24+"/../../private", "0"*24+"/science/%2e%2e/config.json"):
                    with self.assertRaises(ValueError):
                        workspace.asset(value)
                self.assertIsNone(workspace.status()["job"])
            finally:
                workspace.close()

    def test_missing_optional_dependencies_have_actionable_status(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Workspace(directory,availability=lambda:False)
            try:
                self.assertFalse(workspace.status()["available"])
                with self.assertRaises(WorkspaceUnavailable):
                    workspace.submit("scan",{})
            finally:
                workspace.close()

    def test_reusing_immutable_result_selects_it_without_changing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory, availability=lambda:True)
            try:
                first = {"status":"complete", "candidates":[], "source":{"source_kind":"synthetic"}}
                saved = workspace._save("scan", first)
                workspace._save("scan", {"status":"insufficient_baseline", "candidates":[]})
                reused = workspace._save("scan", first)
                self.assertEqual(reused["scan_id"], saved["scan_id"])
                self.assertEqual(workspace.status()["last_scan"], saved)
                self.assertEqual(len(workspace.status()["scans"]), 2)
            finally:
                workspace.close()


class SharedResourceTests(unittest.TestCase):
    def test_waiter_can_cancel_without_entering_or_interrupting_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            entered, release, cancel = threading.Event(),threading.Event(),threading.Event()
            errors=[]
            def owner():
                with heavy_work(directory):
                    entered.set()
                    release.wait(3)
            def waiter():
                try:
                    with heavy_work(directory,cancel.is_set):
                        errors.append("unexpected entry")
                except WorkCancelled:
                    errors.append("cancelled")
            one=threading.Thread(target=owner)
            two=threading.Thread(target=waiter)
            one.start()
            self.assertTrue(entered.wait(2))
            two.start()
            cancel.set()
            two.join(2)
            self.assertEqual(errors,["cancelled"])
            release.set()
            one.join(2)


class WorkspaceHTTPTests(unittest.TestCase):
    def test_new_routes_use_existing_local_auth_and_keep_legacy_api(self):
        from ophanim.desktop import DesktopController, OphanimHTTPServer
        with tempfile.TemporaryDirectory() as directory:
            workspace=Workspace(directory,runner=lambda *args:None,availability=lambda:True)
            controller=DesktopController(data_directory=directory,workspace=workspace)
            server=OphanimHTTPServer(('127.0.0.1',0),controller=controller,token='test-token')
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/api/workspace',timeout=3) as response:
                    self.assertTrue(json.load(response)['available'])
                with urlopen(base+'/api/state',timeout=3) as response:
                    self.assertTrue(json.load(response)['ok'])
                request=Request(base+'/api/workspace/scan',data=b'{"source":"cached"}',headers={'Content-Type':'application/json'})
                with self.assertRaises(HTTPError) as result:
                    urlopen(request,timeout=3)
                result.exception.close()
                request.add_header('X-OPHANIM-Token','test-token')
                with urlopen(request,timeout=3) as response:
                    self.assertEqual(response.status,202)
                self.assertEqual(finished(workspace)['job']['state'],'complete')
                imagination=Request(base+'/api/workspace/imagine',data=b'{"parameters":{"seed":42}}',headers={'Content-Type':'application/json'})
                with self.assertRaises(HTTPError) as result:
                    urlopen(imagination,timeout=3)
                self.assertEqual(result.exception.code,400)
                result.exception.close()
                imagination.add_header('X-OPHANIM-Token','test-token')
                with urlopen(imagination,timeout=3) as response:
                    self.assertEqual(response.status,202)
                self.assertEqual(finished(workspace)['job']['state'],'complete')
                invalid_photo=Request(base+'/api/workspace/photo',data=b'not an image',headers={'Content-Type':'image/png'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(invalid_photo,timeout=3)
                error.exception.close()
                invalid_photo.add_header('X-OPHANIM-Token','test-token')
                with self.assertRaises(HTTPError) as error:
                    urlopen(invalid_photo,timeout=3)
                self.assertEqual(error.exception.code,400 if importlib.util.find_spec('PIL') else 503)
                error.exception.close()
                invalid_mask=Request(base+'/api/workspace/mask',data=b'not an image',headers={'Content-Type':'image/png'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(invalid_mask,timeout=3)
                self.assertEqual(error.exception.code,400)
                self.assertIn('token',json.load(error.exception)['error'].lower())
                error.exception.close()
                invalid_mask.add_header('X-OPHANIM-Token','test-token')
                with self.assertRaises(HTTPError) as error:
                    urlopen(invalid_mask,timeout=3)
                self.assertEqual(error.exception.code,400 if importlib.util.find_spec('PIL') else 503)
                error.exception.close()
            finally:
                workspace.close(5)
                server.shutdown()
                server.server_close()
                thread.join(3)
                controller.close()


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('numpy','scipy','xarray','zarr','PIL')), 'requires science extra')
class WorkspacePipelineTests(unittest.TestCase):
    def test_new_animation_is_selected_without_losing_prior_photo_output(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            try:
                original = workspace._save('project',{'science_directory':directory,'visual_directory':directory,
                    'render_directory':'old-render','composite_directory':'old-photo','active_view':'composite','style':'ghost'})
                with patch.object(workspace,'_visual_settings',return_value=(None,None)), patch('ophanim.shawtynet.studio.animation_run',return_value=Path(directory)/'new-animation'):
                    workspace._run('animation',{'project_id':original['project_id'],'frame_count':3},lambda *args:None)
                child = workspace.status()['projects'][0]
                self.assertEqual(child['active_view'],'animation')
                self.assertIn('composite_url',child)
                self.assertIn('animation_url',child)
                self.assertEqual(workspace.record(original['project_id'],'project')['active_view'],'composite')
            finally:
                workspace.close()

    def test_front_preview_does_not_select_uniform_plateau_after_feature_exits(self):
        import numpy as np
        from ophanim.experiments.scenarios import make_scenario, scenario_from_candidate
        from ophanim.workspace import _representative_frame
        recipe = scenario_from_candidate(None,{"kind":"moving_front","duration_minutes":240,"width_km":50,"speed_m_s":100})
        source = make_scenario(recipe)
        frame, method = _representative_frame(source,"moving_front")
        self.assertIn("gradient",method)
        self.assertGreater(float(np.ptp(source.tec.values[frame])),4)
        self.assertLess(frame,source.sizes["time"]-2)

    def test_window_policy_keeps_display_and_final_targets_without_relaxing_gates(self):
        import numpy as np
        from ophanim.workspace import _analysis_frame_times, _observed_wave_settings
        times = np.datetime64('2024-01-01') + np.arange(100)*np.timedelta64(1,'h')
        selected = _analysis_frame_times(times, representative=37)
        self.assertLessEqual(len(selected),12)
        self.assertIn('2024-01-02T13:00:00Z', selected)
        self.assertIn('2024-01-05T03:00:00Z', selected)
        settings, hours = _observed_wave_settings(times)
        self.assertEqual(settings.min_period_minutes, 360)
        self.assertEqual(settings.minimum_samples_per_period, 6)
        self.assertEqual(settings.minimum_cycles, 3)
        self.assertGreaterEqual(hours,18)

    def test_standalone_scenario_is_separate_and_exports_verified_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Workspace(directory)
            try:
                with patch('ophanim.science_reports.science_diagnostics'),patch('ophanim.science_reports.visual_diagnostics'):
                    workspace.submit('scenario',{'parameters':{'kind':'quiet','extent_km':200,'spacing_km':25,'width_km':50,'duration_minutes':30}})
                    status=finished(workspace)
                    self.assertEqual(status['job']['state'],'complete',status['job']['error'])
                    project=status['projects'][0]
                    self.assertEqual(project['kind'],'hypothetical')
                    self.assertEqual(project['representative_frame']['status'],'chosen')
                    science = Path(workspace.record(project['project_id'],'project')['science_directory'])
                    primary = json.loads((science/'event.json').read_text())
                    timeline = json.loads((science/'events.json').read_text())
                    self.assertNotEqual(primary['time'], project['representative_frame']['time'])
                    self.assertEqual(len(timeline['frames']), project['analysis_summary']['timeline_frames'])
                    self.assertEqual(project['analysis_summary']['wave_status'], 'insufficient_samples')
                    self.assertEqual(json.loads((science/'science_packet.json').read_text())['source_kind'],'synthetic')
                    self.assertIsNone(status['last_scan'])
                    for name in ('preview_url','export_url','recipe_url'):
                        content,mime=workspace.asset(project[name].removeprefix('/workspace-assets/'))
                        self.assertGreater(len(content),10)
                        if name == 'recipe_url':
                            from ophanim.experiments.scenarios import make_scenario
                            self.assertEqual(make_scenario(json.loads(content)).sizes['time'],7)
                    workspace.submit('style',{'project_id':project['project_id'],'style':'quiet'})
                    second=finished(workspace)
                    self.assertEqual(second['job']['state'],'complete',second['job']['error'])
                    self.assertNotEqual(second['projects'][0]['project_id'],project['project_id'])
                    self.assertEqual(second['projects'][0]['scenario_id'],project['scenario_id'])
            finally:
                workspace.close(5)


if __name__ == '__main__':
    unittest.main()
