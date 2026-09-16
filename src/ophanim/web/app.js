"use strict";

(() => {
  const maxUploadBytes = 64 * 1024 * 1024;
  const defaultMambaBounds = Object.freeze({
    south: 29.0,
    west: -100.5,
    north: 32.5,
    east: -96.0,
  });
  const token = document
    .querySelector('meta[name="ophanim-token"]')
    ?.getAttribute("content");

  const elements = {
    connection: document.querySelector("#connection-status"),
    connectionLabel: document.querySelector("#connection-label"),
    stopButton: document.querySelector("#stop-button"),
    stopDialog: document.querySelector("#stop-dialog"),
    confirmStop: document.querySelector("#confirm-stop"),
    stoppedScreen: document.querySelector("#stopped-screen"),
    sourcePanel: document.querySelector("#source-panel"),
    loadForm: document.querySelector("#load-form"),
    loadButton: document.querySelector("#load-button"),
    loadNote: document.querySelector("#load-note"),
    automaticSourceTab: document.querySelector("#source-mode-automatic"),
    localSourceTab: document.querySelector("#source-mode-local"),
    automaticSourcePanel: document.querySelector("#automatic-source-panel"),
    localSourcePanel: document.querySelector("#local-source-panel"),
    gimEdition: document.querySelector("#gim-edition"),
    gimDate: document.querySelector("#gim-date"),
    editionDetail: document.querySelector("#edition-detail"),
    fileInput: document.querySelector("#source-file"),
    fileDrop: document.querySelector("#file-drop"),
    fileTitle: document.querySelector("#file-title"),
    fileDetail: document.querySelector("#file-detail"),
    provider: document.querySelector("#provider"),
    product: document.querySelector("#product"),
    revision: document.querySelector("#revision"),
    revisionField: document.querySelector("#revision-field"),
    revisionPriority: document.querySelector("#revision-priority"),
    revisionPriorityField: document.querySelector("#revision-priority-field"),
    expectedCells: document.querySelector("#expected-cells"),
    north: document.querySelector("#north"),
    west: document.querySelector("#west"),
    south: document.querySelector("#south"),
    east: document.querySelector("#east"),
    loadedSummary: document.querySelector("#loaded-summary"),
    loadedName: document.querySelector("#loaded-name"),
    loadedDetail: document.querySelector("#loaded-detail"),
    replaceFile: document.querySelector("#replace-file"),
    regionalPanel: document.querySelector("#regional-panel"),
    regionalGridForm: document.querySelector("#regional-grid-form"),
    regionalEpoch: document.querySelector("#regional-epoch"),
    regionalResolution: document.querySelector("#regional-resolution"),
    regionalGridButton: document.querySelector("#regional-grid-button"),
    regionalGridStatus: document.querySelector("#regional-grid-status"),
    regionalResult: document.querySelector("#regional-result"),
    regionalResultTitle: document.querySelector("#regional-result-title"),
    regionalStaleBadge: document.querySelector("#regional-stale-badge"),
    regionalResultDetail: document.querySelector("#regional-result-detail"),
    regionalNativeResolution: document.querySelector("#regional-native-resolution"),
    regionalInterpolationMethod: document.querySelector("#regional-interpolation-method"),
    regionalOutputLabel: document.querySelector("#regional-output-label"),
    regionalOutputResolution: document.querySelector("#regional-output-resolution"),
    regionalOutputKind: document.querySelector("#regional-output-kind"),
    regionalTableBody: document.querySelector("#regional-table-body"),
    regionalCsvButton: document.querySelector("#regional-csv-button"),
    mambaPanel: document.querySelector("#mamba-panel"),
    mambaStatusBadge: document.querySelector("#mamba-status-badge"),
    mambaStatusTitle: document.querySelector("#mamba-status-title"),
    mambaStatusCopy: document.querySelector("#mamba-status-copy"),
    mambaActionButton: document.querySelector("#mamba-action-button"),
    mambaRegionDetail: document.querySelector("#mamba-region-detail"),
    mambaModelDetail: document.querySelector("#mamba-model-detail"),
    mambaTrainingDetail: document.querySelector("#mamba-training-detail"),
    mambaLastCheck: document.querySelector("#mamba-last-check"),
    mambaProgressWrap: document.querySelector("#mamba-progress-wrap"),
    mambaProgressLabel: document.querySelector("#mamba-progress-label"),
    mambaProgressCount: document.querySelector("#mamba-progress-count"),
    mambaProgress: document.querySelector("#mamba-progress"),
    mambaMonitorNote: document.querySelector("#mamba-monitor-note"),
    mambaResult: document.querySelector("#mamba-result"),
    mambaResultTitle: document.querySelector("#mamba-result-title"),
    mambaResultDetail: document.querySelector("#mamba-result-detail"),
    mambaResultBadge: document.querySelector("#mamba-result-badge"),
    mambaReadoutCount: document.querySelector("#mamba-readout-count"),
    mambaUsableCount: document.querySelector("#mamba-usable-count"),
    mambaCandidateCount: document.querySelector("#mamba-candidate-count"),
    mambaMaxScore: document.querySelector("#mamba-max-score"),
    mambaCandidateEmpty: document.querySelector("#mamba-candidate-empty"),
    mambaTableScroll: document.querySelector("#mamba-table-scroll"),
    mambaCandidateBody: document.querySelector("#mamba-candidate-body"),
    mambaProvenanceList: document.querySelector("#mamba-provenance-list"),
    mambaLiveSummary: document.querySelector("#mamba-live-summary"),
    spatialPanel: document.querySelector("#spatial-panel"),
    spatialStatusBadge: document.querySelector("#spatial-status-badge"),
    spatialStatusTitle: document.querySelector("#spatial-status-title"),
    spatialStatusCopy: document.querySelector("#spatial-status-copy"),
    spatialActionButton: document.querySelector("#spatial-action-button"),
    spatialRegionDetail: document.querySelector("#spatial-region-detail"),
    spatialModelDetail: document.querySelector("#spatial-model-detail"),
    spatialTrainingDetail: document.querySelector("#spatial-training-detail"),
    spatialLastCheck: document.querySelector("#spatial-last-check"),
    spatialProgressWrap: document.querySelector("#spatial-progress-wrap"),
    spatialProgressLabel: document.querySelector("#spatial-progress-label"),
    spatialProgressCount: document.querySelector("#spatial-progress-count"),
    spatialProgress: document.querySelector("#spatial-progress"),
    spatialMonitorNote: document.querySelector("#spatial-monitor-note"),
    spatialResult: document.querySelector("#spatial-result"),
    spatialResultTitle: document.querySelector("#spatial-result-title"),
    spatialResultDetail: document.querySelector("#spatial-result-detail"),
    spatialResultBadge: document.querySelector("#spatial-result-badge"),
    spatialMapCount: document.querySelector("#spatial-map-count"),
    spatialUsableCount: document.querySelector("#spatial-usable-count"),
    spatialCandidateCount: document.querySelector("#spatial-candidate-count"),
    spatialMaxScore: document.querySelector("#spatial-max-score"),
    spatialComparison: document.querySelector("#spatial-comparison"),
    spatialComparisonCopy: document.querySelector("#spatial-comparison-copy"),
    spatialCandidateEmpty: document.querySelector("#spatial-candidate-empty"),
    spatialTableScroll: document.querySelector("#spatial-table-scroll"),
    spatialCandidateBody: document.querySelector("#spatial-candidate-body"),
    spatialProvenanceList: document.querySelector("#spatial-provenance-list"),
    spatialLiveSummary: document.querySelector("#spatial-live-summary"),
    forecastPanel: document.querySelector("#forecast-panel"),
    forecastForm: document.querySelector("#forecast-form"),
    forecastButton: document.querySelector("#forecast-button"),
    epochCount: document.querySelector("#epoch-count"),
    origin: document.querySelector("#origin"),
    horizon: document.querySelector("#horizon"),
    validTime: document.querySelector("#valid-time"),
    forecastMode: document.querySelector("#forecast-mode"),
    modeExplanation: document.querySelector("#mode-explanation"),
    results: document.querySelector("#results"),
    reconcileButton: document.querySelector("#reconcile-button"),
    predictedValue: document.querySelector("#predicted-value"),
    predictionCaption: document.querySelector("#prediction-caption"),
    resultOrigin: document.querySelector("#result-origin"),
    resultValid: document.querySelector("#result-valid"),
    forecastStatus: document.querySelector("#forecast-status"),
    anomalyHeading: document.querySelector("#anomaly-heading"),
    anomalyCopy: document.querySelector("#anomaly-copy"),
    residualValue: document.querySelector("#residual-value"),
    anomalyScore: document.querySelector("#anomaly-score"),
    anomalyThreshold: document.querySelector("#anomaly-threshold"),
    disturbanceHeading: document.querySelector("#disturbance-heading"),
    disturbanceCopy: document.querySelector("#disturbance-copy"),
    provenanceList: document.querySelector("#provenance-list"),
    notice: document.querySelector("#notice"),
    noticeMessage: document.querySelector("#notice-message"),
    noticeClose: document.querySelector("#notice-close"),
    stepSource: document.querySelector("#step-source"),
    stepForecast: document.querySelector("#step-forecast"),
    stepResult: document.querySelector("#step-result"),
  };

  let selectedFile = null;
  let sourceMode = "automatic";
  let loadedDataset = null;
  let regionalReadout = null;
  let lastResult = null;
  let noticeTimer = null;
  let mambaAction = null;
  let mambaStatus = null;
  let mambaPollTimer = null;
  let mambaPollInFlight = false;
  let mambaStopped = false;
  let spatialAction = null;
  let spatialStatus = null;
  let spatialPollTimer = null;
  let spatialPollInFlight = false;

  function setConnection(online, label) {
    elements.connection.classList.toggle("is-online", online);
    elements.connection.classList.toggle("is-offline", !online);
    elements.connectionLabel.textContent = label;
  }

  function setStep(activeStep) {
    const steps = [
      elements.stepSource,
      elements.stepForecast,
      elements.stepResult,
    ];

    steps.forEach((step, index) => {
      const number = index + 1;
      step.classList.toggle("is-active", number === activeStep);
      step.classList.toggle("is-complete", number < activeStep);
      if (number === activeStep) {
        step.setAttribute("aria-current", "step");
      } else {
        step.removeAttribute("aria-current");
      }
      if (number < activeStep) {
        step.querySelector(".step-number").textContent = "✓";
      } else {
        step.querySelector(".step-number").textContent = String(number).padStart(
          2,
          "0",
        );
      }
    });
  }

  function showNotice(message, kind = "info", duration = 6000) {
    window.clearTimeout(noticeTimer);
    elements.noticeMessage.textContent = message;
    elements.notice.classList.toggle("is-error", kind === "error");
    elements.notice.classList.toggle("is-success", kind === "success");
    elements.notice.hidden = false;
    if (duration > 0) {
      noticeTimer = window.setTimeout(() => {
        elements.notice.hidden = true;
      }, duration);
    }
  }

  function hideNotice() {
    window.clearTimeout(noticeTimer);
    elements.notice.hidden = true;
  }

  function setButtonBusy(button, busy, busyLabel) {
    button.disabled = busy;
    button.classList.toggle("is-loading", busy);
    button.setAttribute("aria-busy", String(busy));
    const label = button.querySelector(".button-label");
    if (!label) return;
    if (!label.dataset.restingLabel) {
      label.dataset.restingLabel = label.textContent;
    }
    label.textContent = busy ? busyLabel : label.dataset.restingLabel;
  }

  function setLoadButtonLabel(label) {
    const buttonLabel = elements.loadButton.querySelector(".button-label");
    buttonLabel.textContent = label;
    buttonLabel.dataset.restingLabel = label;
  }

  function setRegionalButtonLabel(label) {
    const buttonLabel = elements.regionalGridButton.querySelector(".button-label");
    buttonLabel.textContent = label;
    buttonLabel.dataset.restingLabel = label;
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.method && options.method !== "GET") {
      headers.set("X-OPHANIM-Token", token || "");
    }

    let response;
    try {
      response = await fetch(path, {
        ...options,
        headers,
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch (error) {
      setConnection(false, "Application unavailable");
      throw new Error("Cannot reach the local OPHANIM application.", {
        cause: error,
      });
    }

    let payload;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error(`The application returned an unreadable response (${response.status}).`, {
        cause: error,
      });
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || `Request failed (${response.status}).`);
    }
    setConnection(true, "Running locally");
    return payload;
  }

  function formatUtc(value) {
    if (value === null || value === undefined || value === "") return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const formatted = new Intl.DateTimeFormat("en-GB", {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
      timeZone: "UTC",
    }).format(date);
    return `${formatted} UTC`;
  }

  function formatNumber(value, maximumFractionDigits = 2) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return new Intl.NumberFormat(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits,
    }).format(number);
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }

  function metricLabel(metric) {
    return metric === "mean_vtec" ? "Mean VTEC" : "Median VTEC";
  }

  function epochOptionLabel(epoch) {
    return (
      `${formatUtc(epoch.observed_at)} · median ` +
      `${formatNumber(epoch.median_vtec_tecu)} TECU · ` +
      `${formatNumber(epoch.coverage_fraction * 100, 0)}% coverage`
    );
  }

  function formatGridStep(value) {
    return formatNumber(value, 2);
  }

  function formatGridResolution(grid) {
    return (
      `${formatGridStep(grid?.longitude_step_degrees)}° lon × ` +
      `${formatGridStep(grid?.latitude_step_degrees)}° lat`
    );
  }

  function isCoreGrid(grid) {
    const latitudeStep = Number(grid?.latitude_step_degrees);
    const longitudeStep = Number(grid?.longitude_step_degrees);
    return (
      Number.isFinite(latitudeStep) &&
      Number.isFinite(longitudeStep) &&
      Math.abs(latitudeStep - 0.5) < Number.EPSILON &&
      Math.abs(longitudeStep - 0.5) < Number.EPSILON
    );
  }

  function isPlainObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function mambaStatusFromPayload(payload) {
    if (!isPlainObject(payload)) {
      throw new Error("The Mamba status response was not an object.");
    }
    let nested = null;
    for (const key of ["monitor", "mamba", "status"]) {
      if (isPlainObject(payload[key])) {
        nested = payload[key];
        break;
      }
    }
    if (nested) {
      return {
        ...payload,
        ...nested,
        model: nested.model || payload.model || null,
        scan: nested.scan || payload.scan || {},
        active_job: nested.active_job || nested.job || payload.active_job || payload.job || null,
      };
    }
    return payload;
  }

  function normalizedMambaState(status) {
    if (status?.available === false) return "unavailable";
    const job = status?.active_job || status?.job;
    const jobStatus = String(job?.status || "").toLowerCase();
    if (["queued", "running", "cancel_requested"].includes(jobStatus)) {
      const kind = String(job?.kind || job?.job_type || "").toLowerCase();
      return kind.includes("scan") ? "scanning" : "initializing";
    }
    const raw = String(
      status?.state ||
      (typeof status?.status === "string" ? status.status : "") ||
      "",
    ).toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    if (["initializing", "backfilling", "preparing", "training"].includes(raw)) {
      return "initializing";
    }
    if (["scanning", "checking"].includes(raw)) return "scanning";
    if (["failed", "error"].includes(raw)) return "failed";
    if (["interrupted", "paused"].includes(raw)) return "interrupted";
    if (["unavailable", "disabled"].includes(raw)) return "unavailable";
    if (["ready", "initialized", "complete", "completed", "idle"].includes(raw)) {
      return "ready";
    }
    return status?.model || status?.model_version ? "ready" : "uninitialized";
  }

  function humanizeIdentifier(value, fallback = "Working") {
    const normalized = String(value || "").trim();
    if (!normalized) return fallback;
    const words = normalized.replaceAll("_", " ").replaceAll("-", " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  function isDefaultMambaBounds(south, west, north, east) {
    const values = { south, west, north, east };
    return Object.entries(defaultMambaBounds).every(
      ([key, expected]) => Number.isFinite(values[key]) && Math.abs(values[key] - expected) < 1e-9,
    );
  }

  function mambaRegionName(south, west, north, east, suppliedName = "") {
    if (isDefaultMambaBounds(south, west, north, east)) return "Central Texas";
    const normalized = String(suppliedName || "").trim();
    if (normalized && normalized.toLowerCase() !== "central texas") return normalized;
    return "Custom region";
  }

  function currentMambaRegion() {
    const controls = [elements.south, elements.west, elements.north, elements.east];
    for (const control of controls) {
      if (!control.reportValidity()) return null;
    }
    const south = Number(elements.south.value);
    const west = Number(elements.west.value);
    const north = Number(elements.north.value);
    const east = Number(elements.east.value);
    const region = {
      name: mambaRegionName(south, west, north, east),
      south,
      west,
      north,
      east,
    };
    if (Object.values(region).slice(1).some((value) => !Number.isFinite(value))) {
      showNotice("The baseline region needs four finite coordinates.", "error");
      return null;
    }
    if (region.south >= region.north) {
      showNotice("The baseline region’s south edge must be south of its north edge.", "error");
      return null;
    }
    if (region.west >= region.east) {
      showNotice("This baseline version needs a west edge west of its east edge.", "error");
      return null;
    }
    return region;
  }

  function formatMambaRegion(region) {
    if (!isPlainObject(region)) {
      const south = Number(elements.south.value);
      const west = Number(elements.west.value);
      const north = Number(elements.north.value);
      const east = Number(elements.east.value);
      if ([south, west, north, east].every(Number.isFinite)) {
        const name = mambaRegionName(south, west, north, east);
        return (
          `${name} · ${formatNumber(south, 3)}° to ` +
          `${formatNumber(north, 3)}° lat · ${formatNumber(west, 3)}° to ` +
          `${formatNumber(east, 3)}° lon`
        );
      }
      return "Baseline region";
    }
    const south = Number(region.south ?? region.south_latitude_degrees);
    const west = Number(region.west ?? region.west_longitude_degrees);
    const north = Number(region.north ?? region.north_latitude_degrees);
    const east = Number(region.east ?? region.east_longitude_degrees);
    const suppliedName = region.name || region.region_name;
    const name = mambaRegionName(south, west, north, east, suppliedName);
    if (![south, west, north, east].every(Number.isFinite)) return name;
    return (
      `${name} · ${formatNumber(south, 3)}° to ${formatNumber(north, 3)}° lat · ` +
      `${formatNumber(west, 3)}° to ${formatNumber(east, 3)}° lon`
    );
  }

  function setMambaButton(action, label, { busy = false, disabled = false } = {}) {
    mambaAction = action;
    const labelElement = elements.mambaActionButton.querySelector(".button-label");
    labelElement.textContent = label;
    elements.mambaActionButton.disabled = disabled || busy || action === null;
    elements.mambaActionButton.classList.toggle("is-loading", busy);
    elements.mambaActionButton.setAttribute("aria-busy", String(busy));
  }

  function setMambaBadge(label, kind) {
    elements.mambaStatusBadge.textContent = label;
    elements.mambaStatusBadge.className = `badge ${kind}`;
  }

  function mambaModel(status) {
    return status?.model || status?.model_version || null;
  }

  function mambaScan(status) {
    return status?.scan || status?.scan_state || {};
  }

  function mambaLastResult(status) {
    const scan = mambaScan(status);
    return scan.last_result || scan.latest_result || status?.last_result || status?.last_scan || null;
  }

  function mambaActiveJob(status) {
    return status?.active_job || status?.job || null;
  }

  function renderMambaProgress(job, active) {
    elements.mambaProgressWrap.hidden = !active;
    if (!active) return;

    const phase = job?.phase || job?.stage || job?.kind;
    elements.mambaProgressLabel.textContent = humanizeIdentifier(phase, "Preparing");
    const completed = Number(job?.completed_units ?? job?.completed ?? job?.processed_units);
    const total = Number(job?.total_units ?? job?.total);
    let fraction = Number(job?.progress_fraction ?? job?.progress);
    if (!Number.isFinite(fraction) && Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
      fraction = completed / total;
    }
    if (Number.isFinite(fraction) && fraction > 1 && fraction <= 100) fraction /= 100;
    if (Number.isFinite(fraction)) {
      fraction = Math.max(0, Math.min(1, fraction));
      elements.mambaProgress.value = fraction * 100;
      const numericCopy = Number.isFinite(completed) && Number.isFinite(total) && total > 0
        ? `${formatNumber(completed, 0)} of ${formatNumber(total, 0)} · `
        : "";
      elements.mambaProgressCount.textContent =
        `${numericCopy}${formatNumber(fraction * 100, 0)}% complete`;
    } else {
      elements.mambaProgress.removeAttribute("value");
      elements.mambaProgressCount.textContent = String(job?.message || "Job is running");
    }
  }

  function mambaCandidateArea(candidate) {
    if (typeof candidate?.area === "string" && candidate.area.trim()) return candidate.area;
    if (typeof candidate?.location === "string" && candidate.location.trim()) {
      return candidate.location;
    }
    if (typeof candidate?.region_name === "string" && candidate.region_name.trim()) {
      return candidate.region_name;
    }
    const bounds = candidate?.bounds || candidate?.area_bounds;
    if (isPlainObject(bounds)) {
      const south = Number(bounds.south ?? bounds.south_latitude_degrees);
      const west = Number(bounds.west ?? bounds.west_longitude_degrees);
      const north = Number(bounds.north ?? bounds.north_latitude_degrees);
      const east = Number(bounds.east ?? bounds.east_longitude_degrees);
      if ([south, west, north, east].every(Number.isFinite)) {
        return (
          `${formatNumber(south, 2)}°–${formatNumber(north, 2)}° lat, ` +
          `${formatNumber(west, 2)}°–${formatNumber(east, 2)}° lon`
        );
      }
    }
    const peak = candidate?.peak || candidate;
    const latitude = Number(peak.latitude_degrees ?? peak.latitude);
    const longitude = Number(peak.longitude_degrees ?? peak.longitude);
    if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
      return `${formatNumber(latitude, 2)}°, ${formatNumber(longitude, 2)}°`;
    }
    return "Regional readout";
  }

  function renderMambaCandidates(candidates, reportedCount, { showEmpty = true } = {}) {
    const rows = Array.isArray(candidates) ? candidates : [];
    const fragment = document.createDocumentFragment();
    rows.forEach((candidate) => {
      const peak = candidate?.peak || candidate || {};
      const values = [
        formatUtc(candidate?.observed_at || candidate?.first_observed_at),
        mambaCandidateArea(candidate),
        formatNumber(candidate?.anomaly_score ?? candidate?.score ?? peak.anomaly_score, 3),
        Number.isFinite(Number(candidate?.residual_tecu ?? peak.residual_tecu))
          ? `${formatNumber(candidate?.residual_tecu ?? peak.residual_tecu, 3)} TECU`
          : "—",
        formatNumber(candidate?.contributing_cell_count ?? candidate?.cell_count, 0),
        "Candidate",
      ];
      const tableRow = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        tableRow.append(cell);
      });
      fragment.append(tableRow);
    });
    elements.mambaCandidateBody.replaceChildren(fragment);
    elements.mambaTableScroll.hidden = rows.length === 0;
    elements.mambaCandidateEmpty.hidden = rows.length > 0 || !showEmpty;
    if (rows.length > 0 && reportedCount > rows.length) {
      elements.mambaMonitorNote.textContent =
        `Showing ${formatNumber(rows.length, 0)} of ${formatNumber(reportedCount, 0)} ` +
        "candidate readouts, highest-scoring first.";
    }
  }

  function shortCursor(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "string" || typeof value === "number") return String(value);
    try {
      const encoded = JSON.stringify(value);
      return encoded.length > 180 ? `${encoded.slice(0, 177)}…` : encoded;
    } catch (_error) {
      return "Stored cursor";
    }
  }

  function renderMambaProvenance(status, result) {
    const model = mambaModel(status) || {};
    const scan = mambaScan(status);
    const rows = [
      ["Model version", model.model_version_id ?? model.version_id ?? model.id],
      ["Algorithm", model.algorithm],
      ["Decision threshold", model.threshold ?? result.threshold, "number"],
      ["Model artifact", model.artifact_checksum_sha256 ?? model.checksum_sha256],
      ["Training data hash", model.training_data_hash],
      ["Training start", model.training_start ?? model.training_started_at, "date"],
      ["Training end", model.training_end ?? model.training_ended_at, "date"],
      ["Regularized samples", model.regularized_sample_count, "integer"],
      ["Missing baseline slots", model.missing_slot_count, "integer"],
      ["Longest missing span", model.maximum_missing_gap_hours, "hours"],
      ["Scan run", result.scan_run_id ?? result.run_id ?? result.scan_id],
      ["Cursor start", result.cursor_start ?? result.start_cursor],
      ["Cursor end", result.cursor_end ?? result.end_cursor ?? scan.last_successful_cursor],
      ["Completed", result.completed_at ?? result.finished_at ?? scan.last_successful_at, "date"],
    ];
    const fragment = document.createDocumentFragment();
    rows.forEach(([term, rawValue, format]) => {
      const group = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      const hasValue = rawValue !== null && rawValue !== undefined && rawValue !== "";
      if (format === "date") {
        dd.textContent = hasValue ? formatUtc(rawValue) : "—";
      } else if (format === "number") {
        dd.textContent = hasValue ? formatNumber(rawValue, 3) : "—";
      } else if (format === "integer") {
        dd.textContent = hasValue ? formatNumber(rawValue, 0) : "—";
      } else if (format === "hours") {
        dd.textContent = hasValue ? `${formatNumber(rawValue, 1)} hours` : "—";
      } else {
        dd.textContent = shortCursor(rawValue);
      }
      group.append(dt, dd);
      fragment.append(group);
    });
    elements.mambaProvenanceList.replaceChildren(fragment);
  }

  function optionalCount(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : null;
  }

  function mambaSourceDaySummary(result) {
    const requested = optionalCount(result.requested_day_count);
    const available = optionalCount(result.available_day_count);
    const missing = optionalCount(result.missing_day_count);
    if (requested === null && available === null && missing === null) return "";
    return (
      `Source days: ${requested === null ? "—" : formatNumber(requested, 0)} requested · ` +
      `${available === null ? "—" : formatNumber(available, 0)} available · ` +
      `${missing === null ? "—" : formatNumber(missing, 0)} missing`
    );
  }

  function setMambaLiveSummary(message) {
    if (elements.mambaLiveSummary.textContent !== message) {
      elements.mambaLiveSummary.textContent = message;
    }
  }

  function renderMambaResult(status) {
    const result = mambaLastResult(status);
    if (!isPlainObject(result)) {
      elements.mambaResult.hidden = true;
      return;
    }
    const candidates = result.candidates || result.candidate_events || [];
    const readoutCount = Number(result.readout_count ?? result.readouts_checked ?? 0);
    const usableCount = Number(result.usable_readout_count ?? result.usable_count ?? readoutCount);
    const candidateCount = Number(
      result.candidate_readout_count ??
      result.candidate_event_count ??
      result.candidate_count ??
      candidates.length,
    );
    const maximumScore = result.max_anomaly_score ?? result.maximum_score ?? result.max_score;
    const firstObserved = result.first_observed_at || result.observed_start;
    const lastObserved = result.last_observed_at || result.observed_end;
    const completedAt = result.completed_at || result.finished_at || mambaScan(status).last_successful_at;

    elements.mambaReadoutCount.textContent = formatNumber(readoutCount, 0);
    elements.mambaUsableCount.textContent = formatNumber(usableCount, 0);
    elements.mambaCandidateCount.textContent = formatNumber(candidateCount, 0);
    const hasScorableReadouts = Number.isFinite(usableCount) && usableCount > 0;
    elements.mambaMaxScore.textContent = hasScorableReadouts
      ? formatNumber(maximumScore, 3)
      : "—";
    if (!hasScorableReadouts) {
      elements.mambaResultTitle.textContent = "No new readouts were available to assess";
      elements.mambaResultBadge.textContent = "No data";
      elements.mambaResultBadge.className = "badge neutral";
      elements.mambaCandidateEmpty.textContent =
        "No new readouts were available to assess in this scan.";
    } else if (candidateCount > 0) {
      elements.mambaResultTitle.textContent =
        `${formatNumber(candidateCount, 0)} candidate ` +
        `${candidateCount === 1 ? "readout needs" : "readouts need"} review`;
      elements.mambaResultBadge.textContent = "Review";
      elements.mambaResultBadge.className = "badge warning";
      elements.mambaCandidateEmpty.textContent =
        "No candidate readouts were included in this scan result.";
    } else {
      elements.mambaResultTitle.textContent =
        `No candidate readouts in ${formatNumber(readoutCount, 0)} ` +
        `${readoutCount === 1 ? "readout" : "readouts"}`;
      elements.mambaResultBadge.textContent = "Complete";
      elements.mambaResultBadge.className = "badge success";
      elements.mambaCandidateEmpty.textContent =
        "No candidate readouts were reported in this scan.";
    }
    const range = firstObserved || lastObserved
      ? `${formatUtc(firstObserved || lastObserved)} through ${formatUtc(lastObserved || firstObserved)}`
      : "Stored readout window";
    const sourceDaySummary = mambaSourceDaySummary(result);
    if (!hasScorableReadouts) {
      elements.mambaResultDetail.textContent =
        `${sourceDaySummary ? `${sourceDaySummary}. ` : ""}` +
        `Completed ${formatUtc(completedAt)}. No abnormality assessment was possible.`;
      setMambaLiveSummary(
        `Mamba check complete. No new readouts were available to assess.` +
        `${sourceDaySummary ? ` ${sourceDaySummary}.` : ""}`,
      );
    } else {
      elements.mambaResultDetail.textContent =
        `${range} · completed ${formatUtc(completedAt)}. ` +
        `${sourceDaySummary ? `${sourceDaySummary}. ` : ""}` +
        "Candidates are experimental signals, not confirmed disturbances.";
      setMambaLiveSummary(
        candidateCount > 0
          ? `Mamba check complete. ${formatNumber(candidateCount, 0)} candidate ` +
            `${candidateCount === 1 ? "readout needs" : "readouts need"} review.`
          : `Mamba check complete. No candidate readouts in ` +
            `${formatNumber(usableCount, 0)} assessed ` +
            `${usableCount === 1 ? "readout" : "readouts"}.`,
      );
    }
    renderMambaCandidates(candidates, candidateCount, { showEmpty: hasScorableReadouts });
    renderMambaProvenance(status, result);
    elements.mambaResult.hidden = false;
  }

  function renderMambaStatus(status) {
    mambaStatus = status;
    const state = normalizedMambaState(status);
    const model = mambaModel(status);
    const scan = mambaScan(status);
    const job = mambaActiveJob(status);
    const modelId = model?.model_version_id || model?.version_id || model?.id;
    const pendingCount = Number(scan.pending_readout_count ?? status?.pending_readout_count);
    const lastSuccessfulAt = scan.last_successful_at || mambaLastResult(status)?.completed_at;
    const region = status?.region || status?.baseline_region || model?.region;

    elements.mambaRegionDetail.textContent = formatMambaRegion(region);
    elements.mambaModelDetail.textContent = modelId || "Not initialized";
    const trainingStart = model?.training_start || model?.training_started_at;
    const trainingEnd = model?.training_end || model?.training_ended_at;
    const trainingCount = Number(model?.training_readout_count ?? model?.readout_count);
    if (trainingStart || trainingEnd) {
      elements.mambaTrainingDetail.textContent =
        `${formatUtc(trainingStart)} through ${formatUtc(trainingEnd)}` +
        (Number.isFinite(trainingCount) ? ` · ${formatNumber(trainingCount, 0)} readouts` : "");
    } else {
      const years = Number(model?.history_years ?? status?.history_years ?? 20);
      elements.mambaTrainingDetail.textContent =
        `${Number.isFinite(years) ? formatNumber(years, 0) : "20"} years requested`;
    }
    elements.mambaLastCheck.textContent = lastSuccessfulAt ? formatUtc(lastSuccessfulAt) : "Never";
    elements.mambaMonitorNote.classList.remove("is-error");

    if (state === "unavailable") {
      setMambaBadge("Unavailable", "danger");
      elements.mambaStatusTitle.textContent = "The Mamba pipeline is unavailable";
      elements.mambaStatusCopy.textContent = String(
        status?.reason || status?.message || "Start the data services, then this panel will reconnect automatically.",
      );
      setMambaButton(null, "Pipeline unavailable", { disabled: true });
      elements.mambaMonitorNote.textContent = "The source-map and forecast tools remain available.";
    } else if (state === "initializing") {
      setMambaBadge("Initializing", "warning");
      elements.mambaStatusTitle.textContent = "Building the historical baseline";
      const initializationMessage = String(
        job?.message || "Downloading, validating, and preparing historical readouts",
      ).trim();
      elements.mambaStatusCopy.textContent =
        `${initializationMessage}${/[.!?]$/.test(initializationMessage) ? "" : "."} ` +
        "The recurrence stays fixed; initialization fits its ridge predictor and anomaly calibration.";
      setMambaButton(null, "Initializing baseline…", { busy: true });
      elements.mambaMonitorNote.textContent =
        "Progress is saved by the pipeline; a successful baseline will be versioned before it becomes active.";
    } else if (state === "scanning") {
      setMambaBadge("Scanning", "warning");
      elements.mambaStatusTitle.textContent = "Checking new readouts";
      elements.mambaStatusCopy.textContent = String(
        job?.message || "Acquiring complete daily maps and comparing their readouts with the saved baseline.",
      );
      setMambaButton(null, "Checking readouts…", { busy: true });
      elements.mambaMonitorNote.textContent =
        "The successful-scan cursor advances only after every result in this window is saved.";
    } else if (state === "failed" || state === "interrupted") {
      const hasModel = Boolean(model);
      setMambaBadge(state === "failed" ? "Needs attention" : "Interrupted", "danger");
      elements.mambaStatusTitle.textContent = hasModel
        ? "The last readout check did not finish"
        : "Baseline initialization did not finish";
      elements.mambaStatusCopy.textContent = String(
        job?.error || job?.message || status?.error || "The saved progress can be retried without skipping readouts.",
      );
      setMambaButton(
        hasModel ? "scan" : "initialize",
        hasModel ? "Retry readout check" : "Resume initialization",
      );
      elements.mambaMonitorNote.textContent =
        "The last successful cursor is unchanged, so retrying will not create a gap.";
      elements.mambaMonitorNote.classList.add("is-error");
    } else if (state === "ready") {
      setMambaBadge("Ready", "success");
      elements.mambaStatusTitle.textContent = "Ready to check CODE for new readouts";
      const localPendingCopy = Number.isFinite(pendingCount)
        ? ` ${formatNumber(pendingCount, 0)} ${pendingCount === 1 ? "readout is" : "readouts are"} ` +
          "already stored locally beyond the successful cursor."
        : "";
      elements.mambaStatusCopy.textContent = lastSuccessfulAt
        ? "Download complete CODE days and assess every readout since the last successful scan." +
          localPendingCopy
        : "Run the first check from the model’s immutable training snapshot through the latest complete readout." +
          localPendingCopy;
      setMambaButton("scan", "Check new readouts");
      elements.mambaMonitorNote.textContent =
        "A successful check advances the saved cursor; failed checks leave it unchanged for a safe retry.";
    } else {
      setMambaBadge("Not initialized", "neutral");
      elements.mambaStatusTitle.textContent = "Initialize a 20-year regional baseline";
      elements.mambaStatusCopy.textContent =
        "The one-time job acquires historical CODE maps, then fits a ridge predictor and anomaly calibration over a fixed Mamba-inspired recurrence for the current regional crop.";
      setMambaButton("initialize", "Initialize 20-year baseline");
      elements.mambaMonitorNote.textContent =
        "This is a large, long-running local experiment. The pipeline reports durable progress here.";
    }

    renderMambaProgress(job, state === "initializing" || state === "scanning");
    renderMambaResult(status);
  }

  function renderMambaStatusError(error) {
    setMambaBadge("Unavailable", "danger");
    elements.mambaStatusTitle.textContent = "Cannot read the Mamba pipeline status";
    elements.mambaStatusCopy.textContent = error.message;
    elements.mambaModelDetail.textContent = "Status unavailable";
    setMambaButton(null, "Waiting to reconnect", { disabled: true });
    elements.mambaProgressWrap.hidden = true;
    elements.mambaMonitorNote.textContent =
      "OPHANIM will retry this panel automatically; the rest of the application can still be used.";
    elements.mambaMonitorNote.classList.add("is-error");
  }

  function mambaStatusIsActive(status) {
    const state = normalizedMambaState(status);
    return state === "initializing" || state === "scanning";
  }

  function scheduleMambaPoll(status = mambaStatus) {
    window.clearTimeout(mambaPollTimer);
    if (mambaStopped) return;
    let delay = mambaStatusIsActive(status) ? 2000 : 10000;
    if (document.hidden) delay = Math.max(delay, 30000);
    if (normalizedMambaState(status) === "unavailable") delay = 30000;
    mambaPollTimer = window.setTimeout(refreshMambaStatus, delay);
  }

  async function refreshMambaStatus({ announceError = false } = {}) {
    if (mambaPollInFlight || mambaStopped) return;
    mambaPollInFlight = true;
    try {
      const payload = await api("/api/mamba/status");
      const status = mambaStatusFromPayload(payload);
      renderMambaStatus(status);
      scheduleMambaPoll(status);
    } catch (error) {
      renderMambaStatusError(error);
      if (announceError) showNotice(`Mamba monitor unavailable: ${error.message}`, "error", 0);
      scheduleMambaPoll({ available: false });
    } finally {
      mambaPollInFlight = false;
    }
  }

  function spatialStatusFromPayload(payload) {
    if (!isPlainObject(payload)) {
      throw new Error("The spatial baseline status response was not an object.");
    }
    let nested = null;
    for (const key of ["spatial", "convlstm", "monitor", "status"]) {
      if (isPlainObject(payload[key])) {
        nested = payload[key];
        break;
      }
    }
    if (!nested) return payload;
    return {
      ...payload,
      ...nested,
      model: nested.model || payload.model || null,
      scan: nested.scan || payload.scan || {},
      active_job: nested.active_job || nested.job || payload.active_job || payload.job || null,
    };
  }

  function spatialModel(status) {
    return status?.model || status?.model_version || null;
  }

  function spatialScan(status) {
    return status?.scan || status?.scan_state || {};
  }

  function spatialLastResult(status) {
    const scan = spatialScan(status);
    return scan.last_result || scan.latest_result || status?.last_result || status?.last_scan || null;
  }

  function spatialActiveJob(status) {
    return status?.active_job || status?.job || null;
  }

  function setSpatialButton(action, label, { busy = false, disabled = false } = {}) {
    spatialAction = action;
    const labelElement = elements.spatialActionButton.querySelector(".button-label");
    labelElement.textContent = label;
    elements.spatialActionButton.disabled = disabled || busy || action === null;
    elements.spatialActionButton.classList.toggle("is-loading", busy);
    elements.spatialActionButton.setAttribute("aria-busy", String(busy));
  }

  function setSpatialBadge(label, kind) {
    elements.spatialStatusBadge.textContent = label;
    elements.spatialStatusBadge.className = `badge ${kind}`;
  }

  function renderSpatialProgress(job, active) {
    elements.spatialProgressWrap.hidden = !active;
    if (!active) return;

    const phase = job?.phase || job?.stage || job?.kind;
    elements.spatialProgressLabel.textContent = humanizeIdentifier(phase, "Preparing");
    const completed = Number(job?.completed_units ?? job?.completed ?? job?.processed_units);
    const total = Number(job?.total_units ?? job?.total);
    let fraction = Number(job?.progress_fraction ?? job?.progress);
    if (!Number.isFinite(fraction) && Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
      fraction = completed / total;
    }
    if (Number.isFinite(fraction) && fraction > 1 && fraction <= 100) fraction /= 100;
    if (Number.isFinite(fraction)) {
      fraction = Math.max(0, Math.min(1, fraction));
      elements.spatialProgress.value = fraction * 100;
      const numericCopy = Number.isFinite(completed) && Number.isFinite(total) && total > 0
        ? `${formatNumber(completed, 0)} of ${formatNumber(total, 0)} · `
        : "";
      elements.spatialProgressCount.textContent =
        `${numericCopy}${formatNumber(fraction * 100, 0)}% complete`;
    } else {
      elements.spatialProgress.removeAttribute("value");
      elements.spatialProgressCount.textContent = String(job?.message || "Job is running");
    }
  }

  function spatialCandidateLocation(candidate) {
    const peak = candidate?.peak || candidate?.peak_cell || candidate || {};
    const latitude = Number(
      peak.latitude_degrees ?? peak.latitude ?? candidate?.peak_latitude_degrees,
    );
    const longitude = Number(
      peak.longitude_degrees ?? peak.longitude ?? candidate?.peak_longitude_degrees,
    );
    if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
      return `${formatNumber(latitude, 2)}°, ${formatNumber(longitude, 2)}°`;
    }
    if (typeof candidate?.location === "string" && candidate.location.trim()) {
      return candidate.location;
    }
    return "Native-grid cluster";
  }

  function spatialMambaAssessment(candidate) {
    const raw = candidate?.mamba_candidate ?? candidate?.mamba_match ?? candidate?.comparison?.mamba_candidate;
    if (raw === true) return "Also flagged";
    if (raw === false) return "Spatial only";
    const label = candidate?.mamba_assessment || candidate?.comparison?.label;
    return typeof label === "string" && label.trim() ? label : "Not compared";
  }

  function renderSpatialCandidates(candidates, reportedCount, { showEmpty = true } = {}) {
    const rows = Array.isArray(candidates) ? candidates : [];
    const fragment = document.createDocumentFragment();
    rows.forEach((candidate) => {
      const peak = candidate?.peak || candidate?.peak_cell || candidate || {};
      const residual = candidate?.peak_residual_tecu ?? candidate?.residual_tecu ?? peak.residual_tecu;
      const values = [
        formatUtc(candidate?.observed_at || candidate?.first_observed_at),
        spatialCandidateLocation(candidate),
        formatNumber(candidate?.anomaly_score ?? candidate?.score ?? peak.anomaly_score, 3),
        Number.isFinite(Number(residual)) ? `${formatNumber(residual, 3)} TECU` : "—",
        formatNumber(
          candidate?.cluster_cell_count ?? candidate?.affected_cell_count ?? candidate?.cell_count,
          0,
        ),
        spatialMambaAssessment(candidate),
      ];
      const tableRow = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        tableRow.append(cell);
      });
      fragment.append(tableRow);
    });
    elements.spatialCandidateBody.replaceChildren(fragment);
    elements.spatialTableScroll.hidden = rows.length === 0;
    elements.spatialCandidateEmpty.hidden = rows.length > 0 || !showEmpty;
    if (rows.length > 0 && reportedCount > rows.length) {
      elements.spatialMonitorNote.textContent =
        `Showing ${formatNumber(rows.length, 0)} of ${formatNumber(reportedCount, 0)} ` +
        "residual-map candidates, highest-scoring first.";
    }
  }

  function renderSpatialComparison(result, status) {
    const comparison =
      result?.comparison ||
      result?.mamba_comparison ||
      status?.comparison ||
      status?.mamba_comparison;
    if (!isPlainObject(comparison)) {
      elements.spatialComparison.hidden = true;
      return;
    }
    const matched = optionalCount(
      comparison.both_candidate_count ?? comparison.matched_candidate_count ?? comparison.overlap_count,
    );
    const spatialOnly = optionalCount(comparison.spatial_only_count);
    const mambaOnly = optionalCount(comparison.mamba_only_count);
    const assessed = optionalCount(comparison.compared_readout_count ?? comparison.assessed_count);
    const unscored = optionalCount(comparison.unscored_mamba_readout_count);
    const overlapValue =
      comparison.candidate_set_overlap_fraction ??
      comparison.agreement_fraction ??
      comparison.agreement_rate;
    const agreement = overlapValue === null || overlapValue === undefined
      ? Number.NaN
      : Number(overlapValue);
    const parts = [];
    if (matched !== null) parts.push(`${formatNumber(matched, 0)} flagged by both`);
    if (spatialOnly !== null) parts.push(`${formatNumber(spatialOnly, 0)} spatial only`);
    if (mambaOnly !== null) parts.push(`${formatNumber(mambaOnly, 0)} Mamba only`);
    if (assessed !== null) parts.push(`${formatNumber(assessed, 0)} readouts compared`);
    if (comparison.complete === false && unscored !== null) {
      parts.push(`${formatNumber(unscored, 0)} Mamba readouts unscored`);
    }
    if (Number.isFinite(agreement)) {
      const percentage = agreement <= 1 ? agreement * 100 : agreement;
      parts.push(`${formatNumber(percentage, 1)}% candidate-set overlap`);
    }
    elements.spatialComparisonCopy.textContent = parts.length > 0
      ? `${parts.join(" · ")}. Candidate overlap is supporting evidence, not confirmation.`
      : String(comparison.summary || "Comparison data were saved for this check.");
    elements.spatialComparison.hidden = false;
  }

  function renderSpatialProvenance(status, result) {
    const model = spatialModel(status) || {};
    const scan = spatialScan(status);
    const nativeGrid = model.native_grid_shape || model.grid_shape;
    const gridCopy = Array.isArray(nativeGrid) ? nativeGrid.join(" × ") : nativeGrid;
    const rows = [
      ["Model version", model.model_version_id ?? model.version_id ?? model.id],
      ["Algorithm", model.algorithm ?? model.architecture ?? "Predictive ConvLSTM"],
      ["Input frames", model.input_frame_count ?? model.sequence_length, "integer"],
      ["Forecast horizon", model.forecast_horizon_hours ?? model.horizon_hours, "hours"],
      ["Native grid", gridCopy],
      ["Decision threshold", model.threshold ?? result.threshold, "number"],
      ["Model artifact", model.artifact_checksum_sha256 ?? model.checksum_sha256],
      ["Training data hash", model.training_data_hash],
      ["Training start", model.training_start ?? model.training_started_at, "date"],
      ["Training end", model.training_end ?? model.training_ended_at, "date"],
      ["Check run", result.scan_run_id ?? result.run_id ?? result.scan_id],
      ["Cursor start", result.cursor_start ?? result.start_cursor],
      ["Cursor end", result.cursor_end ?? result.end_cursor ?? scan.last_successful_cursor],
      ["Completed", result.completed_at ?? result.finished_at ?? scan.last_successful_at, "date"],
    ];
    const fragment = document.createDocumentFragment();
    rows.forEach(([term, rawValue, format]) => {
      const group = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      const hasValue = rawValue !== null && rawValue !== undefined && rawValue !== "";
      if (format === "date") {
        dd.textContent = hasValue ? formatUtc(rawValue) : "—";
      } else if (format === "number") {
        dd.textContent = hasValue ? formatNumber(rawValue, 3) : "—";
      } else if (format === "integer") {
        dd.textContent = hasValue ? formatNumber(rawValue, 0) : "—";
      } else if (format === "hours") {
        dd.textContent = hasValue ? `${formatNumber(rawValue, 1)} hours` : "—";
      } else {
        dd.textContent = shortCursor(rawValue);
      }
      group.append(dt, dd);
      fragment.append(group);
    });
    elements.spatialProvenanceList.replaceChildren(fragment);
  }

  function setSpatialLiveSummary(message) {
    if (elements.spatialLiveSummary.textContent !== message) {
      elements.spatialLiveSummary.textContent = message;
    }
  }

  function renderSpatialResult(status) {
    const result = spatialLastResult(status);
    if (!isPlainObject(result)) {
      elements.spatialResult.hidden = true;
      return;
    }
    const candidates =
      result.candidates || result.candidate_maps || result.residual_candidates || result.candidate_events || [];
    const mapCount = Number(result.map_count ?? result.maps_checked ?? result.readout_count ?? 0);
    const usableCount = Number(
      result.usable_map_count ?? result.scorable_map_count ?? result.usable_readout_count ?? mapCount,
    );
    const candidateCount = Number(
      result.candidate_map_count ?? result.candidate_readout_count ?? result.candidate_count ?? candidates.length,
    );
    const maximumScore = result.max_anomaly_score ?? result.maximum_score ?? result.max_score;
    const firstObserved = result.first_observed_at || result.observed_start;
    const lastObserved = result.last_observed_at || result.observed_end;
    const completedAt = result.completed_at || result.finished_at || spatialScan(status).last_successful_at;
    const hasScorableMaps = Number.isFinite(usableCount) && usableCount > 0;

    elements.spatialMapCount.textContent = formatNumber(mapCount, 0);
    elements.spatialUsableCount.textContent = formatNumber(usableCount, 0);
    elements.spatialCandidateCount.textContent = formatNumber(candidateCount, 0);
    elements.spatialMaxScore.textContent = hasScorableMaps ? formatNumber(maximumScore, 3) : "—";

    if (!hasScorableMaps) {
      elements.spatialResultTitle.textContent = "No new maps were available to assess";
      elements.spatialResultBadge.textContent = "No data";
      elements.spatialResultBadge.className = "badge neutral";
      elements.spatialCandidateEmpty.textContent = "No new native maps were available for a residual assessment.";
    } else if (candidateCount > 0) {
      elements.spatialResultTitle.textContent =
        `${formatNumber(candidateCount, 0)} residual-map ` +
        `${candidateCount === 1 ? "candidate needs" : "candidates need"} review`;
      elements.spatialResultBadge.textContent = "Review";
      elements.spatialResultBadge.className = "badge warning";
      elements.spatialCandidateEmpty.textContent = "No candidate details were included in this check result.";
    } else {
      elements.spatialResultTitle.textContent =
        `No residual-map candidates in ${formatNumber(mapCount, 0)} ` +
        `${mapCount === 1 ? "map" : "maps"}`;
      elements.spatialResultBadge.textContent = "Complete";
      elements.spatialResultBadge.className = "badge success";
      elements.spatialCandidateEmpty.textContent = "No residual-map candidates were reported in this check.";
    }

    const range = firstObserved || lastObserved
      ? `${formatUtc(firstObserved || lastObserved)} through ${formatUtc(lastObserved || firstObserved)}`
      : "Stored native-map window";
    if (hasScorableMaps) {
      elements.spatialResultDetail.textContent =
        `${range} · completed ${formatUtc(completedAt)}. ` +
        "Scores come from masked native-grid prediction residuals; candidates are not confirmed disturbances.";
      setSpatialLiveSummary(
        candidateCount > 0
          ? `Spatial check complete. ${formatNumber(candidateCount, 0)} residual-map ` +
            `${candidateCount === 1 ? "candidate needs" : "candidates need"} review.`
          : `Spatial check complete. No candidates in ${formatNumber(usableCount, 0)} scorable ` +
            `${usableCount === 1 ? "map" : "maps"}.`,
      );
    } else {
      elements.spatialResultDetail.textContent =
        `Completed ${formatUtc(completedAt)}. No spatial abnormality assessment was possible.`;
      setSpatialLiveSummary("Spatial check complete. No new maps were available to assess.");
    }
    renderSpatialCandidates(candidates, candidateCount, { showEmpty: hasScorableMaps });
    renderSpatialComparison(result, status);
    renderSpatialProvenance(status, result);
    elements.spatialResult.hidden = false;
  }

  function renderSpatialStatus(status) {
    spatialStatus = status;
    const state = normalizedMambaState(status);
    const model = spatialModel(status);
    const scan = spatialScan(status);
    const job = spatialActiveJob(status);
    const modelId = model?.model_version_id || model?.version_id || model?.id;
    const lastSuccessfulAt = scan.last_successful_at || spatialLastResult(status)?.completed_at;
    const region = status?.region || status?.baseline_region || model?.region;

    elements.spatialRegionDetail.textContent = formatMambaRegion(region);
    elements.spatialModelDetail.textContent = modelId || "Not trained";
    const trainingStart = model?.training_start || model?.training_started_at;
    const trainingEnd = model?.training_end || model?.training_ended_at;
    const trainingCount = Number(model?.training_map_count ?? model?.training_readout_count ?? model?.sample_count);
    if (trainingStart || trainingEnd) {
      elements.spatialTrainingDetail.textContent =
        `${formatUtc(trainingStart)} through ${formatUtc(trainingEnd)}` +
        (Number.isFinite(trainingCount) ? ` · ${formatNumber(trainingCount, 0)} maps` : "");
    } else {
      const years = Number(model?.history_years ?? status?.history_years ?? 20);
      elements.spatialTrainingDetail.textContent =
        `${Number.isFinite(years) ? formatNumber(years, 0) : "20"} years requested`;
    }
    elements.spatialLastCheck.textContent = lastSuccessfulAt ? formatUtc(lastSuccessfulAt) : "Never";
    elements.spatialMonitorNote.classList.remove("is-error");

    if (state === "unavailable") {
      setSpatialBadge("Unavailable", "danger");
      elements.spatialStatusTitle.textContent = "The spatial pipeline is unavailable";
      elements.spatialStatusCopy.textContent = String(
        status?.reason || status?.message || "Start the data services, then this panel will reconnect automatically.",
      );
      setSpatialButton(null, "Pipeline unavailable", { disabled: true });
      elements.spatialMonitorNote.textContent = "Mamba and the source-map tools remain available.";
    } else if (state === "initializing") {
      setSpatialBadge("Training", "warning");
      elements.spatialStatusTitle.textContent = "Training the predictive spatial baseline";
      elements.spatialStatusCopy.textContent = String(
        job?.message || "Preparing chronological native-grid windows and fitting the CNN–ConvLSTM predictor.",
      );
      setSpatialButton(null, "Training spatial baseline…", { busy: true });
      elements.spatialMonitorNote.textContent =
        "Only earlier frames feed each prediction; missing target cells are masked out of loss and calibration.";
    } else if (state === "scanning") {
      setSpatialBadge("Checking", "warning");
      elements.spatialStatusTitle.textContent = "Predicting and comparing new native maps";
      elements.spatialStatusCopy.textContent = String(
        job?.message || "Building residual maps and comparing their candidates with the Mamba result.",
      );
      setSpatialButton(null, "Checking & comparing…", { busy: true });
      elements.spatialMonitorNote.textContent =
        "The saved cursor advances only after the residual evidence and comparison are stored.";
    } else if (state === "failed" || state === "interrupted") {
      const hasModel = Boolean(model);
      setSpatialBadge(state === "failed" ? "Needs attention" : "Interrupted", "danger");
      elements.spatialStatusTitle.textContent = hasModel
        ? "The last spatial check did not finish"
        : "Spatial baseline training did not finish";
      elements.spatialStatusCopy.textContent = String(
        job?.error || job?.message || status?.error || "Saved progress can be retried without skipping maps.",
      );
      setSpatialButton(
        hasModel ? "scan" : "initialize",
        hasModel ? "Retry check & comparison" : "Resume spatial training",
      );
      elements.spatialMonitorNote.textContent =
        "The last successful cursor is unchanged, so retrying will not create a gap.";
      elements.spatialMonitorNote.classList.add("is-error");
    } else if (state === "ready") {
      setSpatialBadge("Ready", "success");
      elements.spatialStatusTitle.textContent = "Ready to check and compare new CODE maps";
      elements.spatialStatusCopy.textContent = lastSuccessfulAt
        ? "Predict every complete map since the last successful check and compare candidates with Mamba."
        : "Run the first causal next-map check from the immutable training snapshot through the latest complete map.";
      setSpatialButton("scan", "Check & compare");
      elements.spatialMonitorNote.textContent =
        "Residual maps retain spatial evidence; failed checks leave the successful cursor unchanged.";
    } else {
      setSpatialBadge("Not trained", "neutral");
      elements.spatialStatusTitle.textContent = "Train a 20-year predictive spatial baseline";
      elements.spatialStatusCopy.textContent =
        "The one-time job builds chronological native-grid sequences and trains a small CNN–ConvLSTM to predict the next two-hour CODE map.";
      setSpatialButton("initialize", "Train spatial baseline");
      elements.spatialMonitorNote.textContent =
        "This is a long-running local experiment. Durable progress and the eventual model version appear here.";
    }

    renderSpatialProgress(job, state === "initializing" || state === "scanning");
    renderSpatialResult(status);
  }

  function renderSpatialStatusError(error) {
    setSpatialBadge("Unavailable", "danger");
    elements.spatialStatusTitle.textContent = "Cannot read the spatial pipeline status";
    elements.spatialStatusCopy.textContent = error.message;
    elements.spatialModelDetail.textContent = "Status unavailable";
    setSpatialButton(null, "Waiting to reconnect", { disabled: true });
    elements.spatialProgressWrap.hidden = true;
    elements.spatialMonitorNote.textContent =
      "OPHANIM will retry this panel automatically; the other application tools remain available.";
    elements.spatialMonitorNote.classList.add("is-error");
  }

  function spatialStatusIsActive(status) {
    const state = normalizedMambaState(status);
    return state === "initializing" || state === "scanning";
  }

  function scheduleSpatialPoll(status = spatialStatus) {
    window.clearTimeout(spatialPollTimer);
    if (mambaStopped) return;
    let delay = spatialStatusIsActive(status) ? 2000 : 10000;
    if (document.hidden) delay = Math.max(delay, 30000);
    if (normalizedMambaState(status) === "unavailable") delay = 30000;
    spatialPollTimer = window.setTimeout(refreshSpatialStatus, delay);
  }

  async function refreshSpatialStatus({ announceError = false } = {}) {
    if (spatialPollInFlight || mambaStopped) return;
    spatialPollInFlight = true;
    try {
      const payload = await api("/api/spatial/status");
      const status = spatialStatusFromPayload(payload);
      renderSpatialStatus(status);
      scheduleSpatialPoll(status);
    } catch (error) {
      renderSpatialStatusError(error);
      if (announceError) showNotice(`Spatial monitor unavailable: ${error.message}`, "error", 0);
      scheduleSpatialPoll({ available: false });
    } finally {
      spatialPollInFlight = false;
    }
  }

  function resetRegionalReadout() {
    regionalReadout = null;
    elements.regionalResult.hidden = true;
    elements.regionalResult.classList.remove("is-stale");
    elements.regionalStaleBadge.hidden = true;
    elements.regionalTableBody.replaceChildren();
    elements.regionalCsvButton.disabled = true;
    elements.regionalGridStatus.classList.remove("is-error");
    elements.regionalGridStatus.textContent =
      `Select an epoch and generate a ` +
      `${formatGridStep(elements.regionalResolution.value)}° readout.`;
    setRegionalButtonLabel("Generate readout");
  }

  function configureRegionalReadout(loaded) {
    resetRegionalReadout();
    elements.regionalEpoch.replaceChildren();
    loaded.epochs.forEach((epoch) => {
      const option = document.createElement("option");
      option.value = epoch.observed_at;
      option.textContent = epochOptionLabel(epoch);
      elements.regionalEpoch.append(option);
    });
    if (
      loaded.default_origin &&
      Array.from(elements.regionalEpoch.options).some(
        (option) => option.value === loaded.default_origin,
      )
    ) {
      elements.regionalEpoch.value = loaded.default_origin;
    }
    const hasEpochs = elements.regionalEpoch.options.length > 0;
    elements.regionalEpoch.disabled = !hasEpochs;
    elements.regionalGridButton.disabled = !hasEpochs;
    if (!hasEpochs) {
      elements.regionalGridStatus.textContent =
        "This source does not contain an epoch available for interpolation.";
    }
    elements.regionalPanel.hidden = false;
  }

  function renderRegionalReadout(readout) {
    if (!readout || !Array.isArray(readout.rows)) {
      throw new Error("The regional readout response did not include grid rows.");
    }

    regionalReadout = readout;
    const fragment = document.createDocumentFragment();
    readout.rows.forEach((row) => {
      const tableRow = document.createElement("tr");
      const latitude = document.createElement("td");
      const longitude = document.createElement("td");
      const estimate = document.createElement("td");
      latitude.textContent = formatNumber(row.latitude_degrees, 4);
      longitude.textContent = formatNumber(row.longitude_degrees, 4);
      estimate.textContent = formatNumber(row.estimated_vtec_tecu, 3);
      tableRow.append(latitude, longitude, estimate);
      fragment.append(tableRow);
    });
    elements.regionalTableBody.replaceChildren(fragment);

    const cellCount = Number.isInteger(Number(readout.cell_count))
      ? Number(readout.cell_count)
      : readout.rows.length;
    const requested = readout.requested_grid || {};
    const nativeGrid = readout.native_grid || {};
    const bounds = readout.bounds || {};
    const pointWord = cellCount === 1 ? "point" : "points";
    const method = readout.method === "bilinear"
      ? "Bilinear interpolation"
      : String(readout.method || "Interpolation");
    const coreGrid = isCoreGrid(requested);

    elements.regionalResultTitle.textContent =
      `${formatNumber(cellCount, 0)} interpolated grid ${pointWord}`;
    elements.regionalNativeResolution.textContent = formatGridResolution(nativeGrid);
    elements.regionalInterpolationMethod.textContent = method;
    elements.regionalOutputLabel.textContent = coreGrid ? "Core output" : "Custom output";
    elements.regionalOutputResolution.textContent = formatGridResolution(requested);
    elements.regionalOutputKind.textContent =
      readout.value_kind === "interpolated_estimate"
        ? "Interpolated estimates"
        : String(readout.value_kind || "Estimated values").replaceAll("_", " ");
    elements.regionalResultDetail.textContent =
      `${formatUtc(readout.observed_at)} · ` +
      `${formatNumber(bounds.south, 3)}° to ${formatNumber(bounds.north, 3)}° lat, ` +
      `${formatNumber(bounds.west, 3)}° to ${formatNumber(bounds.east, 3)}° lon`;
    elements.regionalGridStatus.classList.remove("is-error");
    elements.regionalGridStatus.textContent =
      `${formatNumber(cellCount, 0)} interpolated ` +
      `${cellCount === 1 ? "estimate is" : "estimates are"} ready.`;
    elements.regionalCsvButton.disabled = readout.rows.length === 0;
    elements.regionalResult.classList.remove("is-stale");
    elements.regionalStaleBadge.hidden = true;
    elements.regionalResult.hidden = false;
    setRegionalButtonLabel("Refresh readout");
  }

  function markRegionalReadoutStale() {
    elements.regionalGridStatus.classList.remove("is-error");
    if (regionalReadout) {
      elements.regionalGridStatus.textContent =
        "The epoch or resolution changed. Refresh to update the table.";
      elements.regionalResult.classList.add("is-stale");
      elements.regionalStaleBadge.hidden = false;
      setRegionalButtonLabel("Refresh readout");
    } else {
      elements.regionalGridStatus.textContent =
        `Generate a ${formatGridStep(elements.regionalResolution.value)}° readout ` +
        "for this epoch.";
    }
  }

  function csvCell(value) {
    const text = String(value ?? "");
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }

  function downloadRegionalCsv() {
    if (!regionalReadout || !regionalReadout.rows.length) return;
    const lines = [
      "latitude_degrees,longitude_degrees,estimated_vtec_tecu",
      ...regionalReadout.rows.map((row) =>
        [
          row.latitude_degrees,
          row.longitude_degrees,
          row.estimated_vtec_tecu,
        ].map(csvCell).join(","),
      ),
    ];
    const blob = new Blob([`${lines.join("\r\n")}\r\n`], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const epoch = String(regionalReadout.observed_at || "epoch")
      .replaceAll(":", "-")
      .replaceAll("/", "-");
    const step = regionalReadout.requested_grid?.latitude_step_degrees || "grid";
    link.href = url;
    link.download = `ophanim-regional-${epoch}-${step}deg.csv`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    showNotice("Regional readout downloaded as CSV.", "success");
  }

  function setPanelControls(panel, enabled) {
    panel.querySelectorAll("input, select, textarea").forEach((control) => {
      control.disabled = !enabled;
    });
  }

  function updateEditionDetail() {
    elements.editionDetail.textContent =
      elements.gimEdition.value === "final"
        ? "A later, refined CODE solution for an older observation day."
        : "Usually available sooner after the observation day.";
  }

  function setSourceMode(mode, { focusTab = false } = {}) {
    sourceMode = mode === "local" ? "local" : "automatic";
    const automatic = sourceMode === "automatic";

    elements.automaticSourceTab.setAttribute(
      "aria-selected",
      String(automatic),
    );
    elements.localSourceTab.setAttribute("aria-selected", String(!automatic));
    elements.automaticSourceTab.tabIndex = automatic ? 0 : -1;
    elements.localSourceTab.tabIndex = automatic ? -1 : 0;
    elements.automaticSourcePanel.hidden = !automatic;
    elements.localSourcePanel.hidden = automatic;
    setPanelControls(elements.automaticSourcePanel, automatic);
    setPanelControls(elements.localSourcePanel, !automatic);

    elements.fileInput.required = !automatic;
    elements.revisionField.hidden = automatic;
    elements.revisionPriorityField.hidden = automatic;
    elements.revision.disabled = automatic;
    elements.revisionPriority.disabled = automatic;

    if (automatic) {
      setLoadButtonLabel("Download & inspect GIM");
      elements.loadNote.textContent = elements.gimDate.value
        ? `Ready to download the ${elements.gimEdition.value} CODE GIM for ` +
          `${elements.gimDate.value} UTC.`
        : "Latest uses the newest complete UTC day available.";
    } else {
      setLoadButtonLabel("Load & inspect file");
      updateFile(selectedFile);
    }

    if (focusTab) {
      (automatic
        ? elements.automaticSourceTab
        : elements.localSourceTab
      ).focus();
    }
  }

  function updateFile(file) {
    selectedFile = file || null;
    if (!selectedFile) {
      elements.fileTitle.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = "Choose an IONEX file";
      elements.fileTitle.append(strong);
      elements.fileDetail.textContent = "or drop one here · plain text, gzip, or legacy .Z";
      if (sourceMode === "local") {
        elements.loadNote.textContent = "Choose a local IONEX file to continue.";
      }
      return;
    }
    elements.fileTitle.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = selectedFile.name;
    elements.fileTitle.append(strong);
    if (selectedFile.size > maxUploadBytes) {
      elements.fileDetail.textContent =
        `${formatBytes(selectedFile.size)} · exceeds the 64 MiB limit`;
      elements.loadNote.textContent = "Choose a source map no larger than 64 MiB.";
    } else {
      elements.fileDetail.textContent =
        `${formatBytes(selectedFile.size)} · ready to inspect`;
      elements.loadNote.textContent = "Ready to store, parse, and aggregate this map.";
    }
  }

  function updateValidTime() {
    const origin = new Date(elements.origin.value);
    const horizon = Number(elements.horizon.value);
    if (Number.isNaN(origin.getTime()) || !Number.isFinite(horizon)) {
      elements.validTime.textContent = "Choose an epoch and horizon.";
      return;
    }
    const valid = new Date(origin.getTime() + horizon * 60 * 60 * 1000);
    elements.validTime.textContent = `Valid at ${formatUtc(valid.toISOString())}`;
  }

  function updateModeExplanation() {
    if (elements.forecastMode.value === "live") {
      elements.modeExplanation.textContent =
        "Uses only observations available when the forecast is issued.";
    } else {
      elements.modeExplanation.textContent =
        "Can use observations loaded after their epoch.";
    }
  }

  function showLoaded(loaded, { scroll = false } = {}) {
    loadedDataset = loaded;
    const loadedBounds = loaded.bounds;
    if (isPlainObject(loadedBounds)) {
      const controls = [
        [elements.south, loadedBounds.south ?? loadedBounds.south_latitude_degrees],
        [elements.west, loadedBounds.west ?? loadedBounds.west_longitude_degrees],
        [elements.north, loadedBounds.north ?? loadedBounds.north_latitude_degrees],
        [elements.east, loadedBounds.east ?? loadedBounds.east_longitude_degrees],
      ];
      if (controls.every(([, value]) => Number.isFinite(Number(value)))) {
        controls.forEach(([control, value]) => {
          control.value = String(value);
        });
        if (!mambaStatus || normalizedMambaState(mambaStatus) === "uninitialized") {
          elements.mambaRegionDetail.textContent = formatMambaRegion(null);
        }
        if (!spatialStatus || normalizedMambaState(spatialStatus) === "uninitialized") {
          elements.spatialRegionDetail.textContent = formatMambaRegion(null);
        }
      }
    }
    elements.sourcePanel.classList.add("source-complete");
    elements.loadedSummary.hidden = false;
    elements.loadedName.textContent = loaded.filename;

    const epochWord = loaded.epochs.length === 1 ? "epoch" : "epochs";
    const cellDetail =
      loaded.expected_cell_count === loaded.inferred_cell_count
        ? `${loaded.inferred_cell_count} regional cells`
        : `${loaded.inferred_cell_count} cells found · ${loaded.expected_cell_count} expected`;
    const globalEpochCount = Number(loaded.global_epoch_count);
    const globalMinimum = Number(loaded.global_cells_per_epoch_minimum);
    const globalMaximum = Number(loaded.global_cells_per_epoch_maximum);
    let globalDetail = "";
    if (
      Number.isInteger(globalEpochCount) &&
      globalEpochCount > 0 &&
      Number.isInteger(globalMinimum) &&
      globalMinimum > 0 &&
      Number.isInteger(globalMaximum) &&
      globalMaximum >= globalMinimum
    ) {
      const mapWord = globalEpochCount === 1 ? "map" : "maps";
      globalDetail =
        globalMinimum === globalMaximum
          ? `${globalEpochCount} global ${mapWord} · ` +
            `${formatNumber(globalMinimum, 0)} global cells/map · `
          : `${globalEpochCount} global ${mapWord} · ` +
            `${formatNumber(globalMinimum, 0)}–` +
            `${formatNumber(globalMaximum, 0)} global cells/map · `;
    }
    elements.loadedDetail.textContent =
      `${globalDetail}${loaded.epochs.length} regional ${epochWord} · ` +
      `${cellDetail} · ${loaded.provider}/${loaded.product}`;

    elements.origin.replaceChildren();
    loaded.epochs.forEach((epoch) => {
      const option = document.createElement("option");
      option.value = epoch.observed_at;
      option.textContent = epochOptionLabel(epoch);
      elements.origin.append(option);
    });
    if (
      loaded.default_origin &&
      Array.from(elements.origin.options).some(
        (option) => option.value === loaded.default_origin,
      )
    ) {
      elements.origin.value = loaded.default_origin;
    }

    elements.epochCount.textContent =
      `${loaded.epochs.length} regional ${epochWord} available from this source.`;
    configureRegionalReadout(loaded);
    elements.forecastPanel.hidden = false;
    elements.results.hidden = lastResult === null;
    updateValidTime();
    setStep(2);
    if (scroll) {
      elements.regionalPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function statusPresentation(status) {
    const presentations = {
      pending: ["Pending actual", "neutral"],
      scored: ["Scored", "success"],
      insufficient_data: ["Insufficient data", "warning"],
      failed: ["Failed", "danger"],
    };
    return presentations[status] || [status || "Unknown", "neutral"];
  }

  function renderProvenance(result) {
    const forecast = result.forecast || {};
    const decision = result.decision;
    const actual = result.actual;
    const source = result.source_observation;
    const rows = [
      ["Forecast", forecast.forecast_id],
      ["Model version", forecast.model_version_id],
      ["Detector version", decision?.detector_version_id || "Not scored yet"],
      ["Processing version", forecast.processing_version_id],
      ["Region version", forecast.region_version_id],
      ["Source observation", source?.observation_id || "Unavailable"],
      ["Actual observation", actual?.observation_id || "Awaiting observation"],
      ["Decision", decision?.decision_id || "Not created yet"],
      ["Inputs", (forecast.input_observation_ids || []).join(", ") || "None"],
    ];

    elements.provenanceList.replaceChildren();
    rows.forEach(([term, description]) => {
      const group = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = description || "—";
      group.append(dt, dd);
      elements.provenanceList.append(group);
    });
  }

  function showResult(result, { scroll = false } = {}) {
    if (!result || !result.forecast) return;
    lastResult = result;
    const forecast = result.forecast;
    const decision = result.decision;
    const actual = result.actual;
    const source = result.source_observation;

    elements.predictedValue.textContent = formatNumber(
      forecast.predicted_vtec_tecu,
    );
    elements.predictionCaption.textContent = source
      ? `${metricLabel(forecast.target_metric)} persisted from ` +
        formatUtc(source.observed_at)
      : `${metricLabel(forecast.target_metric)} persistence forecast`;
    elements.resultOrigin.textContent = formatUtc(forecast.origin_at);
    elements.resultValid.textContent = formatUtc(forecast.valid_at);

    const [statusLabel, statusClass] = statusPresentation(forecast.status);
    elements.forecastStatus.textContent = statusLabel;
    elements.forecastStatus.className = `badge ${statusClass}`;

    if (decision) {
      const actualValue = actual
        ? actual[`${forecast.target_metric}_tecu`]
        : null;
      elements.anomalyHeading.textContent = decision.is_forecast_anomaly
        ? "Forecast miss detected"
        : "Forecast within range";
      elements.anomalyCopy.textContent = actual
        ? `Observed ${metricLabel(forecast.target_metric).toLowerCase()} was ` +
          `${formatNumber(actualValue)} TECU at the valid time. ` +
          (decision.is_forecast_anomaly
            ? "The calibrated residual crossed the anomaly threshold."
            : "The calibrated residual stayed below the anomaly threshold.")
        : "A stored decision is available, but its source observation could not be displayed.";
      elements.residualValue.textContent =
        `${formatNumber(decision.residual_tecu)} TECU`;
      elements.anomalyScore.textContent = formatNumber(decision.anomaly_score);
      elements.anomalyThreshold.textContent = formatNumber(decision.threshold);
    } else {
      elements.anomalyHeading.textContent =
        forecast.status === "insufficient_data"
          ? "Not enough usable data"
          : "Awaiting actual";
      elements.anomalyCopy.textContent =
        forecast.status === "insufficient_data"
          ? "No usable actual became available within the scoring window."
          : "The forecast can be scored when an observation exists at its valid time.";
      elements.residualValue.textContent = "—";
      elements.anomalyScore.textContent = "—";
      elements.anomalyThreshold.textContent = "—";
    }

    const assessment = decision?.disturbance_assessment || "unknown";
    const assessmentCopy = {
      normal:
        "The available observation is consistent with the experiment’s normal band.",
      candidate:
        "The residual detector marked this as a candidate disturbance. It is not a confirmation.",
      confirmed:
        "This result carries a confirmed disturbance assessment in the stored decision.",
      unknown:
        "The available evidence does not support a physical disturbance assessment yet.",
    };
    elements.disturbanceHeading.textContent =
      assessment.charAt(0).toUpperCase() + assessment.slice(1);
    elements.disturbanceCopy.textContent = assessmentCopy[assessment] || assessmentCopy.unknown;

    renderProvenance(result);
    elements.results.hidden = false;
    setStep(3);
    if (scroll) {
      elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function queryFromLoadForm() {
    const values = new URLSearchParams();
    values.set("filename", selectedFile.name || "upload.ionex");
    values.set("provider", elements.provider.value.trim());
    values.set("product", elements.product.value.trim());
    values.set("south_latitude_degrees", elements.south.value);
    values.set("west_longitude_degrees", elements.west.value);
    values.set("north_latitude_degrees", elements.north.value);
    values.set("east_longitude_degrees", elements.east.value);
    values.set("expected_cell_count", elements.expectedCells.value);
    values.set("revision", elements.revision.value.trim());
    values.set(
      "revision_priority",
      elements.revisionPriority.value,
    );
    return values;
  }

  function automaticFetchPayload() {
    const expectedCells = elements.expectedCells.value.trim();
    return {
      edition: elements.gimEdition.value,
      date: elements.gimDate.value || null,
      south_latitude_degrees: Number(elements.south.value),
      west_longitude_degrees: Number(elements.west.value),
      north_latitude_degrees: Number(elements.north.value),
      east_longitude_degrees: Number(elements.east.value),
      expected_cell_count: expectedCells ? Number(expectedCells) : null,
    };
  }

  const sourceTabs = [
    elements.automaticSourceTab,
    elements.localSourceTab,
  ];

  elements.automaticSourceTab.addEventListener("click", () => {
    setSourceMode("automatic");
  });

  elements.localSourceTab.addEventListener("click", () => {
    setSourceMode("local");
  });

  sourceTabs.forEach((tab, index) => {
    tab.addEventListener("keydown", (event) => {
      let targetIndex = null;
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        targetIndex = (index - 1 + sourceTabs.length) % sourceTabs.length;
      } else if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        targetIndex = (index + 1) % sourceTabs.length;
      } else if (event.key === "Home") {
        targetIndex = 0;
      } else if (event.key === "End") {
        targetIndex = sourceTabs.length - 1;
      }
      if (targetIndex === null) return;
      event.preventDefault();
      setSourceMode(targetIndex === 0 ? "automatic" : "local", {
        focusTab: true,
      });
    });
  });

  elements.gimEdition.addEventListener("change", () => {
    updateEditionDetail();
    setSourceMode("automatic");
  });

  elements.gimDate.addEventListener("change", () => {
    setSourceMode("automatic");
  });

  elements.fileInput.addEventListener("change", () => {
    updateFile(elements.fileInput.files?.[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    elements.fileDrop.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.fileDrop.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    elements.fileDrop.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.fileDrop.classList.remove("is-dragging");
    });
  });

  elements.fileDrop.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    try {
      elements.fileInput.files = event.dataTransfer.files;
    } catch (_error) {
      // The selectedFile fallback supports browsers with a read-only FileList.
    }
    updateFile(file);
  });

  elements.loadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (sourceMode === "automatic") {
      if (!elements.loadForm.reportValidity()) return;

      setButtonBusy(elements.loadButton, true, "Downloading GIM…");
      elements.loadForm.setAttribute("aria-busy", "true");
      try {
        const payload = await api("/api/gim/fetch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(automaticFetchPayload()),
        });
        showLoaded(payload.loaded, { scroll: true });
        const mapCount = Number(payload.loaded.global_epoch_count);
        const mapCopy = Number.isInteger(mapCount) && mapCount > 0
          ? `${mapCount} global map${mapCount === 1 ? "" : "s"}`
          : `${payload.loaded.epochs.length} regional epochs`;
        const readyCount = Number.isInteger(mapCount) && mapCount > 0
          ? mapCount
          : payload.loaded.epochs.length;
        const verb = payload.loaded.already_present ? "Reused" : "Downloaded";
        showNotice(
          `${verb} ${payload.loaded.filename}; ${mapCopy} ` +
            `${readyCount === 1 ? "is" : "are"} ready.`,
          "success",
        );
      } catch (error) {
        showNotice(`CODE GIM download failed: ${error.message}`, "error", 0);
      } finally {
        setButtonBusy(elements.loadButton, false, "");
        elements.loadForm.removeAttribute("aria-busy");
      }
      return;
    }

    selectedFile = selectedFile || elements.fileInput.files?.[0];
    if (!selectedFile) {
      elements.fileInput.setCustomValidity("Choose an IONEX file first.");
      elements.fileInput.reportValidity();
      showNotice("Choose an IONEX file first.", "error");
      return;
    }
    if (selectedFile.size > maxUploadBytes) {
      showNotice("The selected IONEX file exceeds the 64 MiB limit.", "error");
      return;
    }
    elements.fileInput.setCustomValidity("");
    const formIsValid = elements.loadForm.reportValidity();
    if (!formIsValid) return;

    setButtonBusy(elements.loadButton, true, "Loading file…");
    elements.loadForm.setAttribute("aria-busy", "true");
    try {
      const query = queryFromLoadForm();
      const payload = await api(`/api/load?${query.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: selectedFile,
      });
      showLoaded(payload.loaded, { scroll: true });
      const verb = payload.loaded.already_present ? "Reused" : "Stored";
      showNotice(
        `${verb} ${payload.loaded.filename} and built ` +
          `${payload.loaded.epochs.length} regional epochs.`,
        "success",
      );
    } catch (error) {
      showNotice(error.message, "error", 0);
    } finally {
      setButtonBusy(elements.loadButton, false, "");
      elements.loadForm.removeAttribute("aria-busy");
    }
  });

  elements.replaceFile.addEventListener("click", () => {
    elements.sourcePanel.classList.remove("source-complete");
    elements.loadedSummary.hidden = true;
    elements.regionalPanel.hidden = true;
    elements.forecastPanel.hidden = true;
    elements.results.hidden = lastResult === null;
    loadedDataset = null;
    resetRegionalReadout();
    elements.fileInput.value = "";
    updateFile(null);
    setStep(1);
    if (sourceMode === "automatic") {
      elements.gimEdition.focus();
    } else {
      elements.fileInput.focus();
    }
  });

  elements.origin.addEventListener("change", updateValidTime);
  elements.horizon.addEventListener("change", updateValidTime);
  elements.forecastMode.addEventListener("change", updateModeExplanation);

  elements.regionalEpoch.addEventListener("change", markRegionalReadoutStale);
  elements.regionalResolution.addEventListener("change", markRegionalReadoutStale);
  elements.regionalCsvButton.addEventListener("click", downloadRegionalCsv);

  elements.regionalGridForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!elements.regionalGridForm.reportValidity() || !loadedDataset) return;

    setButtonBusy(elements.regionalGridButton, true, "Interpolating…");
    elements.regionalGridForm.setAttribute("aria-busy", "true");
    elements.regionalGridStatus.classList.remove("is-error");
    elements.regionalGridStatus.textContent = "Building the regional estimate grid…";
    try {
      const step = Number(elements.regionalResolution.value);
      const payload = await api("/api/regional-grid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: loadedDataset.dataset_id,
          observed_at: elements.regionalEpoch.value,
          latitude_step_degrees: step,
          longitude_step_degrees: step,
        }),
      });
      renderRegionalReadout(payload.readout);
    } catch (error) {
      elements.regionalGridStatus.classList.add("is-error");
      elements.regionalGridStatus.textContent = regionalReadout
        ? `Refresh failed; the table below is the previous result. ${error.message}`
        : error.message;
      if (regionalReadout) {
        elements.regionalResult.classList.add("is-stale");
        elements.regionalStaleBadge.hidden = false;
      }
      showNotice(`Regional readout failed: ${error.message}`, "error", 0);
    } finally {
      setButtonBusy(elements.regionalGridButton, false, "");
      elements.regionalGridForm.removeAttribute("aria-busy");
    }
  });

  elements.forecastForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!elements.forecastForm.reportValidity() || !loadedDataset) return;

    setButtonBusy(elements.forecastButton, true, "Running forecast…");
    elements.forecastForm.setAttribute("aria-busy", "true");
    try {
      const payload = await api("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: loadedDataset.dataset_id,
          forecast_origin: elements.origin.value,
          forecast_horizon_hours: Number(elements.horizon.value),
          target_metric: document.querySelector("#target-metric").value,
          mode: elements.forecastMode.value,
          calibration_residual_mean_tecu: Number(
            document.querySelector("#calibration-mean").value,
          ),
          calibration_residual_standard_deviation_tecu: Number(
            document.querySelector("#calibration-std").value,
          ),
          calibration_sample_count: Number(
            document.querySelector("#calibration-count").value,
          ),
          detector_threshold: Number(
            document.querySelector("#detector-threshold").value,
          ),
        }),
      });
      showResult(payload.result, { scroll: true });
      showNotice(
        payload.result.decision
          ? "Forecast issued and scored against the available actual."
          : "Forecast issued. It will remain pending until an actual is available.",
        "success",
      );
    } catch (error) {
      showNotice(error.message, "error", 0);
    } finally {
      setButtonBusy(elements.forecastButton, false, "");
      elements.forecastForm.removeAttribute("aria-busy");
    }
  });

  elements.reconcileButton.addEventListener("click", async () => {
    const restingLabel = elements.reconcileButton.textContent;
    elements.reconcileButton.disabled = true;
    elements.reconcileButton.textContent = "Checking…";
    try {
      const payload = await api("/api/reconcile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (payload.result) showResult(payload.result);
      const scored = payload.summary?.scored || 0;
      showNotice(
        scored
          ? `Scored ${scored} pending forecast${scored === 1 ? "" : "s"}.`
          : "Pending forecasts are up to date.",
        scored ? "success" : "info",
      );
    } catch (error) {
      showNotice(error.message, "error", 0);
    } finally {
      elements.reconcileButton.disabled = false;
      elements.reconcileButton.textContent = restingLabel;
    }
  });

  [elements.south, elements.west, elements.north, elements.east].forEach((control) => {
    control.addEventListener("input", () => {
      if (!mambaStatus || normalizedMambaState(mambaStatus) === "uninitialized") {
        elements.mambaRegionDetail.textContent = formatMambaRegion(null);
      }
      if (!spatialStatus || normalizedMambaState(spatialStatus) === "uninitialized") {
        elements.spatialRegionDetail.textContent = formatMambaRegion(null);
      }
    });
  });

  elements.mambaActionButton.addEventListener("click", async () => {
    const action = mambaAction;
    if (!action) return;
    let body = {};
    if (action === "initialize") {
      const region = currentMambaRegion();
      if (!region) return;
      body = { history_years: 20, region };
    }

    setMambaButton(
      action,
      action === "initialize" ? "Starting initialization…" : "Starting readout check…",
      { busy: true },
    );
    elements.mambaMonitorNote.classList.remove("is-error");
    elements.mambaMonitorNote.textContent =
      action === "initialize"
        ? "Creating a durable initialization job…"
        : "Taking a fixed snapshot of the new-readout window…";
    try {
      const payload = await api(
        action === "initialize" ? "/api/mamba/initialize" : "/api/mamba/scan",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      const status = mambaStatusFromPayload(payload);
      renderMambaStatus(status);
      showNotice(
        action === "initialize"
          ? "The 20-year baseline initialization job has started."
          : "The new-readout check has started.",
        "success",
      );
      scheduleMambaPoll(status);
      window.setTimeout(refreshMambaStatus, 250);
    } catch (error) {
      showNotice(
        `${action === "initialize" ? "Mamba initialization" : "Mamba readout check"} ` +
          `could not start: ${error.message}`,
        "error",
        0,
      );
      if (mambaStatus) renderMambaStatus(mambaStatus);
      else renderMambaStatusError(error);
      scheduleMambaPoll();
    }
  });

  elements.spatialActionButton.addEventListener("click", async () => {
    const action = spatialAction;
    if (!action) return;
    let body = {};
    if (action === "initialize") {
      const savedMambaRegion =
        mambaStatus?.region || mambaStatus?.baseline_region || mambaModel(mambaStatus)?.region;
      const region = isPlainObject(savedMambaRegion)
        ? savedMambaRegion
        : currentMambaRegion();
      if (!region) return;
      body = {
        history_years: 20,
        region,
        architecture: "predictive-convlstm",
        input_frame_count: 12,
        forecast_horizon_hours: 2,
      };
    }

    setSpatialButton(
      action,
      action === "initialize" ? "Starting spatial training…" : "Starting comparison…",
      { busy: true },
    );
    elements.spatialMonitorNote.classList.remove("is-error");
    elements.spatialMonitorNote.textContent = action === "initialize"
      ? "Creating a durable spatial training job…"
      : "Taking a fixed snapshot of the native-map comparison window…";
    try {
      const payload = await api(
        action === "initialize" ? "/api/spatial/initialize" : "/api/spatial/scan",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      const status = spatialStatusFromPayload(payload);
      renderSpatialStatus(status);
      showNotice(
        action === "initialize"
          ? "The predictive ConvLSTM training job has started."
          : "The spatial check and Mamba comparison have started.",
        "success",
      );
      scheduleSpatialPoll(status);
      window.setTimeout(refreshSpatialStatus, 250);
    } catch (error) {
      showNotice(
        `${action === "initialize" ? "Spatial baseline training" : "Spatial check"} ` +
          `could not start: ${error.message}`,
        "error",
        0,
      );
      if (spatialStatus) renderSpatialStatus(spatialStatus);
      else renderSpatialStatusError(error);
      scheduleSpatialPoll();
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden || mambaStopped) {
      scheduleMambaPoll();
      scheduleSpatialPoll();
      return;
    }
    window.clearTimeout(mambaPollTimer);
    window.clearTimeout(spatialPollTimer);
    refreshMambaStatus();
    refreshSpatialStatus();
  });

  elements.noticeClose.addEventListener("click", hideNotice);

  elements.stopButton.addEventListener("click", () => {
    if (typeof elements.stopDialog.showModal === "function") {
      elements.stopDialog.showModal();
    } else if (window.confirm("Stop the local OPHANIM application?")) {
      elements.confirmStop.click();
    }
  });

  elements.confirmStop.addEventListener("click", async (event) => {
    event.preventDefault();
    elements.confirmStop.disabled = true;
    try {
      await api("/api/shutdown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (elements.stopDialog.open) elements.stopDialog.close();
      mambaStopped = true;
      window.clearTimeout(mambaPollTimer);
      window.clearTimeout(spatialPollTimer);
      elements.stoppedScreen.hidden = false;
      setConnection(false, "Stopped");
    } catch (error) {
      elements.confirmStop.disabled = false;
      showNotice(error.message, "error", 0);
    }
  });

  async function initialise() {
    updateFile(null);
    updateEditionDetail();
    setSourceMode("automatic");
    updateModeExplanation();
    setStep(1);
    elements.mambaRegionDetail.textContent = formatMambaRegion(null);
    elements.spatialRegionDetail.textContent = formatMambaRegion(null);
    refreshMambaStatus();
    refreshSpatialStatus();
    try {
      const [health, state] = await Promise.all([
        api("/api/health"),
        api("/api/state"),
      ]);
      setConnection(true, `Running locally · v${health.version}`);
      if (state.loaded) {
        showLoaded(state.loaded);
      }
      if (state.result) {
        showResult(state.result);
      }
    } catch (error) {
      showNotice(error.message, "error", 0);
    }
  }

  initialise();
})();
