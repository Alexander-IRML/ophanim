"""Background research lifecycle, bounded API and manifested asset serving."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ophanim.desktop import DesktopController, OphanimHTTPServer
from ophanim.shawtynet_desktop import (
    ShawtyNetBusy, ShawtyNetDesktopJobs, ShawtyNetUnavailable,
)


class ShawtyNetDesktopJobsTests(unittest.TestCase):
    def test_job_is_nonblocking_exclusive_and_completed_status_survives_restart(self):
        entered = threading.Event()
        finish = threading.Event()
        def runner(request, progress):
            entered.set()
            progress("Working on a known field")
            self.assertTrue(finish.wait(5))
            return {"science_run_id": "a" * 24, "source_kind": "synthetic"}
        with TemporaryDirectory() as directory:
            jobs = ShawtyNetDesktopJobs(directory, runner=runner, availability=lambda: True)
            status = jobs.analyze({"source": "synthetic", "case": "plane_wave"})
            self.assertEqual(status["state"], "running")
            self.assertTrue(entered.wait(2))
            self.assertEqual(jobs.status()["state"], "running")
            with self.assertRaises(ShawtyNetBusy):
                jobs.analyze({"source": "synthetic"})
            finish.set()
            self.assertTrue(jobs.close(timeout=5))
            self.assertEqual(jobs.status()["state"], "complete")
            restored = ShawtyNetDesktopJobs(directory, availability=lambda: True)
            self.assertEqual(restored.status()["result"]["science_run_id"], "a" * 24)

    def test_failed_job_can_retry_without_losing_error_provenance(self):
        calls = []
        def runner(request, progress):
            calls.append(request)
            if len(calls) == 1:
                raise ValueError("insufficient temporal coverage")
            return {"science_run_id": "b" * 24}
        with TemporaryDirectory() as directory:
            jobs = ShawtyNetDesktopJobs(directory, runner=runner, availability=lambda: True)
            jobs.analyze({"source": "desktop"})
            jobs._thread.join(5)
            self.assertEqual(jobs.status()["state"], "failed")
            self.assertIn("temporal coverage", jobs.status()["error"])
            jobs.analyze({"source": "synthetic"})
            # close() now also cancels a worker waiting for shared compute. Wait
            # for this successful retry explicitly instead of racing shutdown.
            jobs._thread.join(5)
            self.assertTrue(jobs.close(timeout=5))
            self.assertEqual(jobs.status()["state"], "complete")

    def test_close_while_waiting_for_shared_compute_is_resumable_interruption(self):
        from ophanim.core.jobs import heavy_work
        calls = []
        with TemporaryDirectory() as directory:
            jobs = ShawtyNetDesktopJobs(directory, runner=lambda *args: calls.append(args),
                                        availability=lambda: True)
            with heavy_work(directory):
                jobs.analyze({"source": "synthetic"})
                self.assertTrue(jobs.close(timeout=5))
                self.assertEqual(jobs.status()["state"], "interrupted")
                self.assertEqual(calls, [])
            restored = ShawtyNetDesktopJobs(directory, availability=lambda: True)
            self.assertEqual(restored.status()["state"], "interrupted")
            self.assertEqual(restored.status()["request"]["source"], "synthetic")

    def test_unavailable_and_untrusted_requests_do_not_start_work(self):
        with TemporaryDirectory() as directory:
            unavailable = ShawtyNetDesktopJobs(directory, availability=lambda: False)
            with self.assertRaises(ShawtyNetUnavailable):
                unavailable.analyze({"source": "synthetic"})
            jobs = ShawtyNetDesktopJobs(directory, availability=lambda: True)
            for request in (
                {"source": "ionex", "path": "/etc/passwd"},
                {"source": "desktop", "output_root": "/tmp/arbitrary"},
                {"source": "synthetic", "case": "unknown"},
                {"bounds": {"south": 0, "north": 90, "west": -180, "east": 180}},
                {"bounds": {"south": 20, "north": 42, "west": -112, "east": float("nan")}},
            ):
                with self.subTest(request=request):
                    with self.assertRaises(ValueError):
                        jobs.analyze(request)
            self.assertEqual(jobs.status()["state"], "idle")
            self.assertFalse(jobs.root.exists())

    def test_interrupted_job_is_reported_on_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "shawtynet"
            root.mkdir()
            (root / "desktop-state.json").write_text(json.dumps({"schema": "ophanim-shawtynet-desktop/1", "state": "running", "result": None}))
            jobs = ShawtyNetDesktopJobs(directory, availability=lambda: True)
            self.assertEqual(jobs.status()["state"], "interrupted")

    def _asset(self, jobs):
        run = jobs.root / ("a" * 24)
        diagnostics = run / "diagnostics"
        diagnostics.mkdir(parents=True)
        report = diagnostics / "report.html"
        report.write_text("<html><p>scientific diagnostic</p></html>")
        (run / "manifest.json").write_text(json.dumps({
            "schema": "ophanim-science-run/1", "kind": "science", "run_id": run.name,
            "status": "complete", "files": {"diagnostics/report.html": sha256(report.read_bytes()).hexdigest()},
        }))
        return run.name + "/diagnostics/report.html", report

    def test_only_manifested_bounded_assets_are_served(self):
        with TemporaryDirectory() as directory:
            jobs = ShawtyNetDesktopJobs(directory)
            relative, report = self._asset(jobs)
            content, mime = jobs.asset(relative)
            self.assertIn(b"scientific diagnostic", content)
            self.assertEqual(mime, "text/html; charset=utf-8")
            for invalid in ("../ophanim.sqlite3", "%2e%2e/ophanim.sqlite3", "/etc/passwd",
                            "a" * 24 + "/dynamic.zarr/.zattrs", relative.replace("report.html", "unknown.png")):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        jobs.asset(invalid)
            report.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                jobs.asset(relative)

    def test_asset_symlinks_are_rejected(self):
        with TemporaryDirectory() as directory:
            jobs = ShawtyNetDesktopJobs(directory)
            relative, report = self._asset(jobs)
            original = report.with_name("original.html")
            report.rename(original)
            report.symlink_to(original)
            with self.assertRaisesRegex(ValueError, "symlink"):
                jobs.asset(relative)


class ShawtyNetHTTPTests(unittest.TestCase):
    def test_research_routes_require_existing_mutation_auth_and_serve_new_panel(self):
        entered = threading.Event()
        finish = threading.Event()
        def runner(request, progress):
            entered.set()
            finish.wait(5)
            return {"science_run_id": "a" * 24}
        with TemporaryDirectory() as directory:
            jobs = ShawtyNetDesktopJobs(directory, runner=runner, availability=lambda: True)
            controller = DesktopController(data_directory=directory, shawtynet_jobs=jobs)
            server = OphanimHTTPServer(("127.0.0.1", 0), controller=controller, token="research-token")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/", timeout=5) as response:
                    self.assertIn(b'id="shawtynet-panel"', response.read())
                with urlopen(base + "/shawtynet.js", timeout=5) as response:
                    self.assertIn(b"X-OPHANIM-Token", response.read())
                with urlopen(base + "/api/shawtynet/status", timeout=5) as response:
                    self.assertEqual(json.load(response)["state"], "idle")
                request = Request(base + "/api/shawtynet/analyze", data=b'{"source":"synthetic"}', headers={"Content-Type":"application/json"})
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=5)
                self.assertEqual(context.exception.code, 400)
                context.exception.close()
                request.add_header("X-OPHANIM-Token", "research-token")
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 202)
                    self.assertEqual(json.load(response)["state"], "running")
                self.assertTrue(entered.wait(2))
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=5)
                self.assertEqual(context.exception.code, 409)
                context.exception.close()
                with urlopen(base + "/api/health", timeout=5) as response:
                    self.assertTrue(json.load(response)["ok"])
            finally:
                finish.set()
                jobs.close(timeout=5)
                server.shutdown()
                server.server_close()
                thread.join(5)
                controller.close()


if __name__ == "__main__":
    unittest.main()
