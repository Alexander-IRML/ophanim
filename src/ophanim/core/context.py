"""Optional timestamp-matched NOAA context, never a causal classifier."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from http.client import HTTPException
import json
import math
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


SOURCE_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
MAX_BYTES = 256 * 1024
TIMEOUT_SECONDS = 10


def _trusted_url(value):
    parsed = urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname != "services.swpc.noaa.gov"
            or parsed.port not in (None, 443) or parsed.username or parsed.password):
        raise ValueError("NOAA context redirects must remain on the official HTTPS host")


class _NOAARedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        absolute = urljoin(req.full_url, newurl)
        _trusted_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


def _utc(value, *, provider=False):
    if not isinstance(value, str):
        raise ValueError("context timestamp must be a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        if not provider:
            raise ValueError("event timestamp must be timezone-aware")
        # NOAA SWPC time_tag is published in UTC even when its string omits Z.
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _stamp(value):
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fetch_geomagnetic_context(*, opener=None):
    """Fetch bounded public NOAA 3-hour planetary Kp context, or unavailable.

    Supports current object records and the older header-row table. Malformed
    or future records are never presented as calm conditions. Missing context
    does not prevent scanning or alter candidate detection/ranking.
    """
    downloaded = datetime.now(UTC)
    result = {"status": "unavailable", "source_url": SOURCE_URL,
              "downloaded_at": _stamp(downloaded), "sha256": None, "records": []}
    request = Request(SOURCE_URL, headers={"Accept": "application/json", "User-Agent": "OPHANIM-research/0.1"})
    try:
        open_url = opener or build_opener(_NOAARedirect()).open
        with open_url(request, timeout=TIMEOUT_SECONDS) as response:
            _trusted_url(response.geturl())
            if getattr(response, "status", 200) != 200:
                raise ValueError("NOAA returned a non-success HTTP response")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_BYTES:
                raise ValueError("NOAA response exceeds size limit")
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("NOAA response exceeds size limit")
        payload = json.loads(raw)
        if not isinstance(payload, list) or not payload or len(payload) > 4096:
            raise ValueError("NOAA records must be a bounded nonempty list")
        if isinstance(payload[0], list):
            headers = payload[0]
            if not all(isinstance(item, str) for item in headers) or len(set(headers)) != len(headers):
                raise ValueError("NOAA header row is invalid")
            rows = []
            for row in payload[1:]:
                if not isinstance(row, list) or len(row) != len(headers):
                    raise ValueError("NOAA data row differs from header")
                rows.append(dict(zip(headers, row)))
        else:
            rows = payload
        selected = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("NOAA record is not an object")
            timestamp = _utc(row["time_tag"], provider=True)
            value = row["Kp"] if "Kp" in row else row.get("kp")
            if isinstance(value, bool):
                raise ValueError("Kp must be numeric, not a boolean")
            kp = float(value)
            if not math.isfinite(kp) or not 0 <= kp <= 9:
                raise ValueError("Kp must be finite and within 0..9")
            if timestamp > downloaded:
                continue
            if timestamp in selected and selected[timestamp] != kp:
                raise ValueError("conflicting duplicate NOAA epochs")
            selected[timestamp] = kp
        if not selected:
            raise ValueError("no published NOAA epochs are available")
        result.update(status="available", sha256=sha256(raw).hexdigest(),
                      records=[{"time": _stamp(stamp), "Kp": selected[stamp]} for stamp in sorted(selected)],
                      interval_hours=3, timestamp_semantics="start of 3-hour UTC Kp interval")
    except (OSError, HTTPException, ValueError, TypeError, KeyError) as error:
        result["reason"] = f"NOAA context could not be verified: {error}"
    return result


def annotate_candidates(candidates, context):
    """Copy candidates and append matching context without changing evidence.

    Kp >=5 is geomagnetic-activity context, not a diagnosis of a local TEC
    feature. Absent temporal overlap is explicitly unavailable, never calm.
    """
    result = deepcopy(candidates)
    for candidate in result:
        start = _utc(candidate["start_time"])
        end = _utc(candidate["end_time"])
        if start > end:
            raise ValueError("candidate start_time follows end_time")
        matching = []
        if context.get("status") == "available":
            for record in context.get("records", []):
                stamp = _utc(record["time"])
                if stamp <= end and stamp + timedelta(hours=3) > start:
                    matching.append(deepcopy(record))
        candidate["geomagnetic_context"] = {
            "status": "matched" if matching else "unavailable",
            "source_url": context.get("source_url", SOURCE_URL),
            "downloaded_at": context.get("downloaded_at"), "sha256": context.get("sha256"),
            "records": matching, "maximum_kp": max((row["Kp"] for row in matching), default=None),
        }
        if matching:
            peak = max(row["Kp"] for row in matching)
            active = peak >= 5
            hypothesis = {
                "label": "Geomagnetic activity overlaps" if active else "Available geomagnetic context below Kp 5",
                "evidence": f"NOAA planetary Kp reaches {peak:g} in {len(matching)} matching 3-hour UTC interval(s).",
                "limitations": ("Association only; cause undetermined. A global 3-hour index does not establish a local TEC cause or event timing."
                                if active else "This does not establish locally quiet conditions or exclude an ionospheric disturbance. Kp is global and coarse in time."),
            }
        else:
            hypothesis = {"label": "Geomagnetic context unavailable for this event",
                          "evidence": "No verified NOAA Kp interval overlaps the observed event times.",
                          "limitations": "Missing or out-of-window context is not evidence of calm conditions; physical cause remains undetermined."}
        candidate.setdefault("hypotheses", []).append(hypothesis)
    json.dumps(result, allow_nan=False)
    return result
