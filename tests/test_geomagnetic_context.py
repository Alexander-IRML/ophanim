"""NOAA context is optional, bounded, timestamp-matched and noncausal."""

from copy import deepcopy
import json
import unittest
from urllib.error import URLError

from ophanim.core.context import SOURCE_URL, MAX_BYTES, annotate_candidates, fetch_geomagnetic_context


class Response:
    status = 200
    headers = {}

    def __init__(self, payload, *, url=SOURCE_URL):
        self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.url = url

    def geturl(self):
        return self.url

    def read(self, size):
        return self.raw[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class GeomagneticContextTests(unittest.TestCase):
    def fetch(self, payload, **kwargs):
        return fetch_geomagnetic_context(opener=lambda request, timeout: Response(payload, **kwargs))

    def test_object_records_are_utc_sorted_and_hashed(self):
        result = self.fetch([{"time_tag": "2024-01-01 03:00:00", "Kp": 6}, {"time_tag": "2024-01-01 00:00:00", "Kp": 3}])
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["records"][0], {"time": "2024-01-01T00:00:00Z", "Kp": 3})
        self.assertEqual(len(result["sha256"]), 64)
        json.dumps(result, allow_nan=False)

    def test_legacy_header_table_is_supported(self):
        result = self.fetch([["time_tag", "Kp", "a_running"], ["2024-01-01 00:00:00", "5.33", "57"]])
        self.assertEqual(result["records"][0]["Kp"], 5.33)

    def test_untrusted_urls_oversize_invalid_and_future_data_are_unavailable(self):
        valid = [{"time_tag": "2024-01-01 00:00:00", "Kp": 3}]
        self.assertEqual(self.fetch(valid, url="https://evil.invalid/data")["status"], "unavailable")
        for payload in (b"x" * (MAX_BYTES + 1), b"bad json", [],
                        [{"time_tag": "2024-01-01", "Kp": float("nan")}],
                        [{"time_tag": "2024-01-01", "Kp": True}],
                        [{"time_tag": "2099-01-01", "Kp": 4}]):
            self.assertEqual(self.fetch(payload)["status"], "unavailable")

    def test_network_failure_is_optional_unavailability(self):
        def unavailable(request, timeout):
            self.assertEqual(timeout, 10)
            raise URLError("offline fixture")
        result = fetch_geomagnetic_context(opener=unavailable)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["records"], [])

    def test_overlap_is_interval_based_and_does_not_mutate_candidates(self):
        context = self.fetch([{"time_tag": "2024-01-01 00:00:00", "Kp": 7}, {"time_tag": "2024-01-01 03:00:00", "Kp": 2}])
        candidates = [{"start_time": "2024-01-01T02:00:00Z", "end_time": "2024-01-01T03:00:00Z", "hypotheses": []}]
        original = deepcopy(candidates)
        annotated = annotate_candidates(candidates, context)
        self.assertEqual(candidates, original)
        self.assertEqual(annotated[0]["geomagnetic_context"]["maximum_kp"], 7)
        self.assertIn("Association only", annotated[0]["hypotheses"][-1]["limitations"])
        candidates[0]["start_time"] = candidates[0]["end_time"]
        boundary = annotate_candidates(candidates, context)
        self.assertEqual(boundary[0]["geomagnetic_context"]["maximum_kp"], 2)
        self.assertIn("does not establish", boundary[0]["hypotheses"][-1]["limitations"])

    def test_no_temporal_overlap_never_implies_calm(self):
        context = self.fetch([{"time_tag": "2024-01-01 00:00:00", "Kp": 1}])
        result = annotate_candidates([{"start_time": "2024-01-02T00:00:00Z", "end_time": "2024-01-02T01:00:00Z"}], context)
        self.assertEqual(result[0]["geomagnetic_context"]["status"], "unavailable")
        self.assertIn("not evidence of calm", result[0]["hypotheses"][-1]["limitations"])


if __name__ == "__main__":
    unittest.main()
