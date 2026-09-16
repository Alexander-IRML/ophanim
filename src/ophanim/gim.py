"""Zero-configuration acquisition of public CODE global ionosphere maps.

The IGS combined GIM is distributed by CDDIS behind NASA Earthdata Login.
For the laptop application we therefore use CODE's public HTTPS product: a
standard 5-degree longitude by 2.5-degree latitude IONEX grid produced by the
Center for Orbit Determination in Europe.  The source identity deliberately
remains ``code``; this module never represents the single-center product as an
IGS combined map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ophanim.ingestion import (
    GZIP_MAGIC,
    UNIX_COMPRESS_MAGIC,
    IngestionRequest,
    LoadedSource,
    decompress_ionex,
)


CODE_PROVIDER = "code"
CODE_PRODUCT = "gim"
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
CODE_ARCHIVE_START = date(2002, 1, 1)
MODERN_FILENAME_START = date(2022, 11, 26)

_PRIMARY_ROOT = "https://www.aiub.unibe.ch/download/CODE/"
_BKG_RAPID_MIRROR_ROOT = (
    "https://igs.bkg.bund.de/root_ftp/IGSac/CODE/CODE/"
)
_ALLOWED_PATH_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "www.aiub.unibe.ch": ("/download/CODE/",),
    "download.aiub.unibe.ch": ("/CODE/",),
    "zhw-b.s3.cloud.switch.ch": ("/aiub/CODE/",),
    "igs.bkg.bund.de": ("/root_ftp/IGSac/CODE/CODE/",),
}


class GIMEdition(StrEnum):
    """Operational maturity of a CODE daily GIM."""

    RAPID = "rapid"
    FINAL = "final"

    @property
    def revision_priority(self) -> int:
        return 10 if self is GIMEdition.RAPID else 20

    @property
    def filename_code(self) -> str:
        return "RAP" if self is GIMEdition.RAPID else "FIN"


@dataclass(frozen=True, slots=True)
class GIMFetchRequest:
    """Select an exact UTC product day or the latest available complete day."""

    edition: GIMEdition = GIMEdition.RAPID
    product_date: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.edition, GIMEdition):
            raise TypeError("edition must be a GIMEdition")
        if self.product_date is not None and not isinstance(
            self.product_date, date
        ):
            raise TypeError("product_date must be a date or None")
        if (
            self.product_date is not None
            and self.product_date < CODE_ARCHIVE_START
        ):
            raise ValueError(
                "automatic GIM acquisition supports CODE archive products "
                f"on or after {CODE_ARCHIVE_START.isoformat()}"
            )


@dataclass(frozen=True, slots=True)
class GIMCandidate:
    """Deterministic CODE filename and trusted download entry points."""

    product_date: date
    edition: GIMEdition
    filename: str
    source_uris: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FetchedGIM:
    """Validated compressed bytes ready for the ingestion commit boundary."""

    product_date: date
    edition: GIMEdition
    ingestion_request: IngestionRequest
    loaded_source: LoadedSource
    resolved_uri: str

    @property
    def source_uri(self) -> str:
        return self.ingestion_request.source_uri


class GIMAcquisitionError(RuntimeError):
    """Base class for an automatic GIM acquisition failure."""


class GIMNotFound(GIMAcquisitionError):
    """No product exists for an exact day or bounded latest-day search."""


class GIMSourceUnavailable(GIMAcquisitionError):
    """A trusted source could not return a usable HTTP response."""


class GIMContentError(GIMAcquisitionError):
    """A response is not a bounded compressed IONEX product."""


class _Response(Protocol):
    status: int
    headers: Any

    def read(self, amount: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...


OpenURL = Callable[..., _Response]


class _CandidateMissing(Exception):
    pass


class _RestrictedRedirectHandler(HTTPRedirectHandler):
    """Permit only the documented HTTPS hosts and CODE archive paths."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        absolute = urljoin(req.full_url, newurl)
        _validate_trusted_uri(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


class CodeGIMSource:
    """Download public CODE rapid/final GIMs over restricted HTTPS.

    A latest rapid search considers the previous seven completed UTC days.
    A latest search starts with the preceding complete UTC day and walks a
    bounded window: seven days for rapid products and twenty-one for finals.
    Only a genuine 404/410 advances to an older date; authentication, server,
    network, redirect, and content errors remain visible rather than
    masquerading as product latency.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        open_url: OpenURL | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = MAX_COMPRESSED_BYTES,
        rapid_lookback_days: int = 7,
        final_start_age_days: int = 1,
        final_lookback_days: int = 21,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if rapid_lookback_days < 1:
            raise ValueError("rapid_lookback_days must be positive")
        if final_start_age_days < 1:
            raise ValueError("final_start_age_days must be positive")
        if final_lookback_days < final_start_age_days:
            raise ValueError(
                "final_lookback_days must be at least final_start_age_days"
            )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._open_url = open_url or build_opener(
            _RestrictedRedirectHandler()
        ).open
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._rapid_lookback_days = rapid_lookback_days
        self._final_start_age_days = final_start_age_days
        self._final_lookback_days = final_lookback_days

    def fetch(self, request: GIMFetchRequest | None = None) -> FetchedGIM:
        selection = request or GIMFetchRequest()
        if selection.product_date is not None:
            try:
                return self._fetch_candidate(
                    code_gim_candidate(
                        selection.edition,
                        selection.product_date,
                    )
                )
            except _CandidateMissing as error:
                raise GIMNotFound(
                    f"No CODE {selection.edition.value} GIM is available for "
                    f"{selection.product_date.isoformat()}"
                ) from error

        checked: list[date] = []
        for product_date in self._latest_dates(selection.edition):
            checked.append(product_date)
            try:
                return self._fetch_candidate(
                    code_gim_candidate(selection.edition, product_date)
                )
            except _CandidateMissing:
                continue
        oldest = checked[-1]
        newest = checked[0]
        raise GIMNotFound(
            f"No CODE {selection.edition.value} GIM was found from "
            f"{newest.isoformat()} through {oldest.isoformat()}"
        )

    def fetch_date(
        self,
        product_date: date,
        *,
        edition: GIMEdition = GIMEdition.RAPID,
    ) -> FetchedGIM:
        return self.fetch(
            GIMFetchRequest(edition=edition, product_date=product_date)
        )

    def fetch_latest(
        self,
        *,
        edition: GIMEdition = GIMEdition.RAPID,
    ) -> FetchedGIM:
        return self.fetch(GIMFetchRequest(edition=edition))

    def _latest_dates(self, edition: GIMEdition) -> tuple[date, ...]:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("GIM acquisition clock must be timezone-aware")
        today = now.astimezone(UTC).date()
        if edition is GIMEdition.RAPID:
            ages = range(1, self._rapid_lookback_days + 1)
        else:
            ages = range(
                self._final_start_age_days,
                self._final_lookback_days + 1,
            )
        return tuple(today - timedelta(days=age) for age in ages)

    def _fetch_candidate(self, candidate: GIMCandidate) -> FetchedGIM:
        operational_errors: list[GIMAcquisitionError] = []
        missing_count = 0
        for source_uri in candidate.source_uris:
            try:
                loaded, resolved_uri = self._download(
                    source_uri,
                    expected_filename=candidate.filename,
                )
            except _CandidateMissing:
                missing_count += 1
                continue
            except GIMAcquisitionError as error:
                operational_errors.append(error)
                continue
            return FetchedGIM(
                product_date=candidate.product_date,
                edition=candidate.edition,
                ingestion_request=IngestionRequest(
                    provider=CODE_PROVIDER,
                    product=CODE_PRODUCT,
                    source_uri=source_uri,
                    revision=candidate.edition.value,
                    revision_priority=candidate.edition.revision_priority,
                ),
                loaded_source=loaded,
                resolved_uri=resolved_uri,
            )
        if operational_errors:
            raise operational_errors[0]
        if missing_count == len(candidate.source_uris):
            raise _CandidateMissing
        raise GIMSourceUnavailable(
            f"No trusted CODE source could return {candidate.filename}"
        )

    def _download(
        self,
        source_uri: str,
        *,
        expected_filename: str,
    ) -> tuple[LoadedSource, str]:
        _validate_trusted_uri(source_uri, expected_filename=expected_filename)
        request = Request(
            source_uri,
            headers={
                "Accept": (
                    "application/gzip, application/x-compress;q=0.9, "
                    "application/octet-stream;q=0.8"
                ),
                "User-Agent": "OPHANIM-GIM/1.0 (+local-science-experiment)",
            },
            method="GET",
        )
        try:
            response_context = self._open_url(
                request,
                timeout=self._timeout_seconds,
            )
        except HTTPError as error:
            status = error.code
            error.close()
            if status in {404, 410}:
                raise _CandidateMissing from error
            if status in {401, 403}:
                raise GIMSourceUnavailable(
                    "The CODE GIM archive refused access; no credentials should "
                    "be required for this public feed"
                ) from error
            raise GIMSourceUnavailable(
                f"CODE GIM archive returned HTTP {status}"
            ) from error
        except (URLError, OSError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise GIMSourceUnavailable(
                f"Could not reach the CODE GIM archive: {reason}"
            ) from error

        try:
            with response_context as response:
                status = getattr(response, "status", 200)
                if status in {404, 410}:
                    raise _CandidateMissing
                if status != 200:
                    raise GIMSourceUnavailable(
                        f"CODE GIM archive returned HTTP {status}"
                    )
                resolved_uri = response.geturl()
                _validate_trusted_uri(
                    resolved_uri,
                    expected_filename=expected_filename,
                )
                content_type = _response_content_type(response.headers)
                content = response.read(self._max_bytes + 1)
        except _CandidateMissing:
            raise
        except GIMAcquisitionError:
            raise
        except (OSError, TimeoutError) as error:
            raise GIMSourceUnavailable(
                f"CODE GIM download ended unexpectedly: {error}"
            ) from error

        if len(content) > self._max_bytes:
            raise GIMContentError(
                f"Downloaded GIM exceeds {self._max_bytes} bytes"
            )
        _validate_compressed_ionex(content, content_type=content_type)
        return LoadedSource(content=content, filename=expected_filename), resolved_uri


def code_gim_filename(edition: GIMEdition, product_date: date) -> str:
    """Return CODE's deterministic daily filename for a supported date."""

    if not isinstance(edition, GIMEdition):
        raise TypeError("edition must be a GIMEdition")
    if not isinstance(product_date, date):
        raise TypeError("product_date must be a date")
    if product_date < CODE_ARCHIVE_START:
        raise ValueError(
            "CODE GIM filenames are supported on or after "
            f"{CODE_ARCHIVE_START.isoformat()}"
        )
    day_of_year = product_date.timetuple().tm_yday
    if product_date < MODERN_FILENAME_START:
        legacy_prefix = "CORG" if edition is GIMEdition.RAPID else "CODG"
        return (
            f"{legacy_prefix}{day_of_year:03d}0."
            f"{product_date.year % 100:02d}I.Z"
        )
    year_and_day = f"{product_date.year:04d}{day_of_year:03d}"
    return (
        f"COD0OPS{edition.filename_code}_{year_and_day}0000_"
        "01D_01H_GIM.INX.gz"
    )


def code_gim_candidate(
    edition: GIMEdition,
    product_date: date,
) -> GIMCandidate:
    """Build trusted public entry points for one CODE product day."""

    filename = code_gim_filename(edition, product_date)
    if product_date < MODERN_FILENAME_START:
        year = f"{product_date.year:04d}"
        source_uris = (
            f"{_PRIMARY_ROOT}{year}/{filename}",
        )
    elif edition is GIMEdition.RAPID:
        source_uris = (
            _PRIMARY_ROOT + filename,
            _BKG_RAPID_MIRROR_ROOT + filename,
        )
    else:
        source_uris = (
            f"{_PRIMARY_ROOT}{product_date.year:04d}/{filename}",
        )
    return GIMCandidate(
        product_date=product_date,
        edition=edition,
        filename=filename,
        source_uris=source_uris,
    )


def _validate_trusted_uri(
    uri: str,
    *,
    expected_filename: str | None = None,
) -> None:
    parsed = urlparse(uri)
    host = (parsed.hostname or "").casefold()
    prefixes = _ALLOWED_PATH_PREFIXES.get(host)
    try:
        port = parsed.port
    except ValueError as error:
        raise GIMSourceUnavailable(
            "CODE GIM download used an invalid archive address"
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or prefixes is None
        or not any(parsed.path.startswith(prefix) for prefix in prefixes)
        or parsed.query
        or parsed.fragment
    ):
        raise GIMSourceUnavailable(
            "CODE GIM download redirected outside the trusted HTTPS archive"
        )
    filename = PurePosixPath(parsed.path).name
    if expected_filename is not None and filename != expected_filename:
        raise GIMSourceUnavailable(
            "CODE GIM download resolved to an unexpected filename"
        )


def _response_content_type(headers: Any) -> str:
    if hasattr(headers, "get_content_type"):
        return str(headers.get_content_type()).casefold()
    if hasattr(headers, "get"):
        return str(headers.get("Content-Type", "")).split(";", 1)[0].casefold()
    return ""


def _validate_compressed_ionex(content: bytes, *, content_type: str) -> None:
    if content_type in {"text/html", "application/xhtml+xml"}:
        raise GIMContentError(
            "The GIM source returned a web/login page instead of IONEX data"
        )
    if not content.startswith((GZIP_MAGIC, UNIX_COMPRESS_MAGIC)):
        preview = content[:256].lstrip().lower()
        if preview.startswith((b"<!doctype html", b"<html")):
            raise GIMContentError(
                "The GIM source returned a web/login page instead of IONEX data"
            )
        raise GIMContentError(
            "Downloaded GIM is not a gzip or Unix-compress stream"
        )
    try:
        decompressed = decompress_ionex(
            content,
            max_decompressed_bytes=MAX_DECOMPRESSED_BYTES,
        )
    except ValueError as error:
        if "maximum decompressed size" in str(error):
            raise GIMContentError(
                "Downloaded GIM exceeds the decompressed safety limit"
            ) from error
        raise GIMContentError(
            f"Downloaded GIM has invalid compressed data: {error}"
        ) from error
    required_labels = (
        b"IONEX VERSION / TYPE",
        b"# OF MAPS IN FILE",
        b"MAP DIMENSION",
        b"LAT1 / LAT2 / DLAT",
        b"LON1 / LON2 / DLON",
        b"END OF HEADER",
        b"START OF TEC MAP",
    )
    if any(label not in decompressed for label in required_labels):
        raise GIMContentError(
            "Downloaded data does not contain a complete IONEX TEC product"
        )


def _validate_gzip_ionex(content: bytes, *, content_type: str) -> None:
    """Backward-compatible private alias for the generalized validator."""

    _validate_compressed_ionex(content, content_type=content_type)


__all__ = [
    "CODE_ARCHIVE_START",
    "CODE_PRODUCT",
    "CODE_PROVIDER",
    "CodeGIMSource",
    "FetchedGIM",
    "GIMAcquisitionError",
    "GIMCandidate",
    "GIMContentError",
    "GIMEdition",
    "GIMFetchRequest",
    "GIMNotFound",
    "GIMSourceUnavailable",
    "code_gim_candidate",
    "code_gim_filename",
]
