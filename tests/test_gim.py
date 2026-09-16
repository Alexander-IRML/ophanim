"""Tests for deterministic, bounded CODE GIM acquisition."""

from __future__ import annotations

import gzip
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError

from ophanim.gim import (
    CODE_PRODUCT,
    CODE_PROVIDER,
    CodeGIMSource,
    GIMContentError,
    GIMEdition,
    GIMFetchRequest,
    GIMNotFound,
    GIMSourceUnavailable,
    code_gim_candidate,
    code_gim_filename,
)


def _valid_ionex_gzip() -> bytes:
    content = b"\n".join(
        (
            b"     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE",
            b"    25                                                      # OF MAPS IN FILE",
            b"     2                                                      MAP DIMENSION",
            b"  87.5 -87.5  -2.5                                         LAT1 / LAT2 / DLAT",
            b"-180.0 180.0   5.0                                         LON1 / LON2 / DLON",
            b"                                                            END OF HEADER",
            b"     1                                                      START OF TEC MAP",
        )
    )
    return gzip.compress(content)


def _unix_compress_literals(content: bytes) -> bytes:
    """Build a small valid .Z stream without adding a test dependency."""

    if len(content) > 255:
        raise ValueError("literal-only .Z fixture must stay on nine-bit codes")
    output = bytearray(b"\x1f\x9d\x09")
    accumulator = 0
    bit_count = 0
    for code in content:
        accumulator |= code << bit_count
        bit_count += 9
        while bit_count >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bit_count -= 8
    if bit_count:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _valid_ionex_unix_compress() -> bytes:
    content = b"\n".join(
        (
            b"IONEX VERSION / TYPE",
            b"# OF MAPS IN FILE",
            b"MAP DIMENSION",
            b"LAT1 / LAT2 / DLAT",
            b"LON1 / LON2 / DLON",
            b"END OF HEADER",
            b"START OF TEC MAP",
        )
    )
    return _unix_compress_literals(content)


_UNIX_DECODER_AVAILABLE = any(
    path.is_file() for path in (Path("/usr/bin/gzip"), Path("/bin/gzip"))
)


class _Response:
    def __init__(
        self,
        content: bytes,
        *,
        url: str,
        status: int = 200,
        content_type: str = "application/gzip",
    ) -> None:
        self._content = content
        self._url = url
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self, amount: int = -1) -> bytes:
        return self._content if amount < 0 else self._content[:amount]

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _OpenURL:
    def __init__(self, actions: dict[str, object]) -> None:
        self.actions = actions
        self.requests = []

    def __call__(self, request, *, timeout: float):
        self.requests.append((request, timeout))
        action = self.actions.get(request.full_url)
        if action is None:
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        if isinstance(action, BaseException):
            raise action
        return action


class CodeGIMSourceTests(unittest.TestCase):
    def test_modern_filenames_and_sources_are_deterministic(self) -> None:
        product_date = date(2024, 2, 29)

        rapid = code_gim_candidate(GIMEdition.RAPID, product_date)
        final = code_gim_candidate(GIMEdition.FINAL, product_date)

        self.assertEqual(
            rapid.filename,
            "COD0OPSRAP_20240600000_01D_01H_GIM.INX.gz",
        )
        self.assertEqual(
            final.filename,
            "COD0OPSFIN_20240600000_01D_01H_GIM.INX.gz",
        )
        self.assertEqual(len(rapid.source_uris), 2)
        self.assertTrue(rapid.source_uris[0].startswith("https://www.aiub.unibe.ch/"))
        self.assertIn("/2024/", final.source_uris[0])

    def test_legacy_filenames_and_year_sources_are_deterministic(self) -> None:
        product_date = date(2008, 2, 29)

        rapid = code_gim_candidate(GIMEdition.RAPID, product_date)
        final = code_gim_candidate(GIMEdition.FINAL, product_date)

        self.assertEqual(rapid.filename, "CORG0600.08I.Z")
        self.assertEqual(final.filename, "CODG0600.08I.Z")
        self.assertEqual(len(rapid.source_uris), 1)
        self.assertTrue(all("/2008/" in uri for uri in rapid.source_uris))
        self.assertTrue(all("/2008/" in uri for uri in final.source_uris))

    def test_filename_transition_and_archive_floor_are_explicit(self) -> None:
        self.assertEqual(
            code_gim_filename(GIMEdition.FINAL, date(2022, 11, 25)),
            "CODG3290.22I.Z",
        )
        self.assertEqual(
            code_gim_filename(GIMEdition.FINAL, date(2022, 11, 26)),
            "COD0OPSFIN_20223300000_01D_01H_GIM.INX.gz",
        )
        GIMFetchRequest(product_date=date(2002, 1, 1))
        with self.assertRaisesRegex(ValueError, "2002-01-01"):
            GIMFetchRequest(product_date=date(2001, 12, 31))

    @unittest.skipUnless(
        _UNIX_DECODER_AVAILABLE,
        "system gzip is required for legacy .Z compatibility",
    )
    def test_legacy_exact_date_returns_validated_unix_compress_bytes(self) -> None:
        product_date = date(2006, 1, 2)
        candidate = code_gim_candidate(GIMEdition.FINAL, product_date)
        compressed = _valid_ionex_unix_compress()
        opener = _OpenURL(
            {
                candidate.source_uris[0]: _Response(
                    compressed,
                    url=candidate.source_uris[0],
                    content_type="application/x-compress",
                )
            }
        )

        fetched = CodeGIMSource(open_url=opener).fetch_date(
            product_date,
            edition=GIMEdition.FINAL,
        )

        self.assertEqual(fetched.loaded_source.filename, "CODG0020.06I.Z")
        self.assertEqual(fetched.loaded_source.content, compressed)
        self.assertEqual(fetched.ingestion_request.revision, "final")
        self.assertEqual(fetched.ingestion_request.revision_priority, 20)

    def test_exact_date_returns_validated_bytes_and_ingestion_identity(self) -> None:
        product_date = date(2026, 9, 9)
        candidate = code_gim_candidate(GIMEdition.RAPID, product_date)
        final_uri = (
            "https://zhw-b.s3.cloud.switch.ch/aiub/CODE/" + candidate.filename
        )
        opener = _OpenURL(
            {
                candidate.source_uris[0]: _Response(
                    _valid_ionex_gzip(),
                    url=final_uri,
                )
            }
        )
        source = CodeGIMSource(open_url=opener)

        fetched = source.fetch_date(product_date)

        self.assertEqual(fetched.product_date, product_date)
        self.assertEqual(fetched.edition, GIMEdition.RAPID)
        self.assertEqual(fetched.source_uri, candidate.source_uris[0])
        self.assertEqual(fetched.resolved_uri, final_uri)
        self.assertEqual(fetched.loaded_source.filename, candidate.filename)
        self.assertTrue(fetched.loaded_source.content.startswith(b"\x1f\x8b"))
        self.assertEqual(fetched.ingestion_request.provider, CODE_PROVIDER)
        self.assertEqual(fetched.ingestion_request.product, CODE_PRODUCT)
        self.assertEqual(fetched.ingestion_request.revision, "rapid")
        self.assertEqual(fetched.ingestion_request.revision_priority, 10)
        request, timeout = opener.requests[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIn("OPHANIM-GIM", request.get_header("User-agent"))
        self.assertEqual(timeout, 30.0)

    def test_rapid_uses_public_bkg_mirror_when_primary_is_missing(self) -> None:
        product_date = date(2026, 9, 9)
        candidate = code_gim_candidate(GIMEdition.RAPID, product_date)
        opener = _OpenURL(
            {
                candidate.source_uris[1]: _Response(
                    _valid_ionex_gzip(),
                    url=candidate.source_uris[1],
                )
            }
        )

        fetched = CodeGIMSource(open_url=opener).fetch_date(product_date)

        self.assertEqual(fetched.source_uri, candidate.source_uris[1])
        self.assertEqual(
            [request.full_url for request, _ in opener.requests],
            list(candidate.source_uris),
        )

    def test_latest_scans_completed_utc_days_newest_first(self) -> None:
        newest = date(2026, 9, 9)
        available = date(2026, 9, 8)
        candidate = code_gim_candidate(GIMEdition.RAPID, available)
        opener = _OpenURL(
            {
                candidate.source_uris[0]: _Response(
                    _valid_ionex_gzip(),
                    url=(
                        "https://download.aiub.unibe.ch/CODE/"
                        + candidate.filename
                    ),
                )
            }
        )
        source = CodeGIMSource(
            open_url=opener,
            clock=lambda: datetime(2026, 9, 10, 1, tzinfo=UTC),
        )

        fetched = source.fetch_latest()

        self.assertEqual(fetched.product_date, available)
        newest_candidate = code_gim_candidate(GIMEdition.RAPID, newest)
        self.assertEqual(
            [request.full_url for request, _ in opener.requests[:2]],
            list(newest_candidate.source_uris),
        )
        self.assertEqual(opener.requests[2][0].full_url, candidate.source_uris[0])

    def test_operational_failure_does_not_backscan_to_stale_date(self) -> None:
        newest = code_gim_candidate(GIMEdition.RAPID, date(2026, 9, 9))
        older = code_gim_candidate(GIMEdition.RAPID, date(2026, 9, 8))
        opener = _OpenURL(
            {
                newest.source_uris[0]: URLError("offline"),
                older.source_uris[0]: _Response(
                    _valid_ionex_gzip(),
                    url=older.source_uris[0],
                ),
            }
        )
        source = CodeGIMSource(
            open_url=opener,
            clock=lambda: datetime(2026, 9, 10, tzinfo=UTC),
        )

        with self.assertRaisesRegex(GIMSourceUnavailable, "offline"):
            source.fetch_latest()

        self.assertNotIn(
            older.source_uris[0],
            [request.full_url for request, _ in opener.requests],
        )

    def test_exact_date_reports_not_found_without_substituting_another_day(self) -> None:
        opener = _OpenURL({})
        source = CodeGIMSource(open_url=opener)

        with self.assertRaisesRegex(GIMNotFound, "2026-09-09"):
            source.fetch(
                GIMFetchRequest(
                    edition=GIMEdition.RAPID,
                    product_date=date(2026, 9, 9),
                )
            )

        self.assertEqual(len(opener.requests), 2)

    def test_html_invalid_gzip_and_oversize_responses_are_rejected(self) -> None:
        product_date = date(2026, 9, 9)
        candidate = code_gim_candidate(GIMEdition.FINAL, product_date)
        cases = (
            (b"<html>login</html>", "text/html", 1024, "web/login"),
            (b"not gzip", "application/octet-stream", 1024, "not a gzip"),
            (_valid_ionex_gzip(), "application/gzip", 10, "exceeds"),
        )
        for content, content_type, max_bytes, message in cases:
            with self.subTest(message=message):
                opener = _OpenURL(
                    {
                        candidate.source_uris[0]: _Response(
                            content,
                            url=candidate.source_uris[0],
                            content_type=content_type,
                        )
                    }
                )
                with self.assertRaisesRegex(GIMContentError, message):
                    CodeGIMSource(
                        open_url=opener,
                        max_bytes=max_bytes,
                    ).fetch_date(
                        product_date,
                        edition=GIMEdition.FINAL,
                    )

    def test_untrusted_resolved_host_is_rejected(self) -> None:
        product_date = date(2026, 9, 9)
        candidate = code_gim_candidate(GIMEdition.FINAL, product_date)
        opener = _OpenURL(
            {
                candidate.source_uris[0]: _Response(
                    _valid_ionex_gzip(),
                    url=f"https://evil.example/{candidate.filename}",
                )
            }
        )

        with self.assertRaisesRegex(GIMSourceUnavailable, "trusted HTTPS"):
            CodeGIMSource(open_url=opener).fetch_date(
                product_date,
                edition=GIMEdition.FINAL,
            )

    def test_clock_must_be_aware(self) -> None:
        source = CodeGIMSource(
            open_url=_OpenURL({}),
            clock=lambda: datetime(2026, 9, 10),
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            source.fetch_latest()

    def test_filename_rejects_untyped_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "GIMEdition"):
            code_gim_filename("rapid", date(2026, 1, 1))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
