"use strict";

// Deliberately dependency-free: native source cells, safe DOM, explicit research boundaries.
(() => {
  const $ = (id) => document.getElementById(`work-${id}`);
  if (!$('discover')) return;
  const token = document.querySelector('meta[name="ophanim-token"]')?.content || "";
  const state = {data: null, scan: null, candidate: null, project: null, submitting: false,
    view: "discover", scanKey: "", projectKey: "", pinnedScan: null, seenJob: null, pendingAction: null,
    recipeCandidateId: null, inherited: null, cameraMode: null};
  let timer;
  let polling = false;
  const human = (value) => String(value ?? "unknown").replaceAll("_", " ");
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const number = (value, digits = 1) => finite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: digits}) : "—";
  const signed = (value) => finite(value) ? `${value > 0 ? "+" : ""}${number(value)}` : "—";
  function timeMillis(value) {
    if (!value) return NaN;
    const iso = /(?:Z|[+-]\d\d:\d\d)$/.test(String(value)) ? String(value) : `${value}Z`;
    return new Date(iso).valueOf();
  }
  function utc(value) {
    if (!value) return "Unknown time";
    const iso = /(?:Z|[+-]\d\d:\d\d)$/.test(String(value)) ? String(value) : `${value}Z`;
    const date = new Date(iso);
    return Number.isNaN(date.valueOf()) ? String(value) : `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
  }
  function age(value) {
    if (!value) return "age unknown";
    const iso = /(?:Z|[+-]\d\d:\d\d)$/.test(String(value)) ? value : `${value}Z`;
    const hours = (Date.now() - new Date(iso).valueOf()) / 3600000;
    if (!Number.isFinite(hours)) return "age unknown";
    if (hours < 0) return "future timestamp; verify source";
    return hours < 48 ? `${number(hours, 0)} hours old` : `${number(hours / 24, 1)} days old`;
  }
  function node(tag, text, className) {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  }
  function safeAsset(value) {
    return typeof value === "string" && /^\/workspace-assets\/[A-Za-z0-9_-]+\/[A-Za-z0-9_./-]+$/.test(value)
      && !value.includes("..") ? value : "";
  }
  function showError(error) {
    $('error').textContent = typeof error === "string" ? error : error.message || "The operation could not finish.";
    $('error').hidden = false;
  }
  function clearError() { $('error').hidden = true; $('error').textContent = ""; }

  const tabs = [...document.querySelectorAll('[data-work-tab]')];
  function navigate(name, focus = false) {
    if (!tabs.some((tab) => tab.dataset.workTab === name)) name = "discover";
    state.view = name;
    for (const tab of tabs) {
      const selected = tab.dataset.workTab === name;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      $(tab.dataset.workTab).hidden = !selected;
      if (selected && focus) tab.focus();
    }
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    if (name === "event" && state.candidate) renderEvent();
    window.scrollTo({top: 0, behavior: "instant"});
  }
  for (const tab of tabs) {
    tab.addEventListener("click", () => navigate(tab.dataset.workTab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const index = tabs.indexOf(tab);
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      navigate(tabs[next].dataset.workTab, true);
    });
  }
  for (const button of document.querySelectorAll('[data-work-go]'))
    button.addEventListener("click", () => navigate(button.dataset.workGo, true));
  window.addEventListener("hashchange", () => navigate(location.hash.slice(1)));

  async function api(path, body) {
    const response = await fetch(`/api/workspace${path}`, body === undefined ? {cache: "no-store"} : {
      method: "POST", headers: {"Content-Type": "application/json", "X-OPHANIM-Token": token},
      body: JSON.stringify(body),
    });
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error("The workspace returned an unreadable response. Check the local application."); }
    if (!response.ok || data.ok === false) throw new Error(typeof data.error === "string" ? data.error : "The request could not finish.");
    return data;
  }
  function busy() { return state.submitting || ["running", "queued", "cancelling"].includes(state.data?.job?.state); }
  function updateControls() {
    const working = busy();
    const available = state.data?.available !== false;
    for (const id of ["pull", "cached", "demo", "create"]) $(id).disabled = working || !available;
    $('artify').disabled = working || !state.candidate || !available;
    $('restyle').disabled = working || !state.project;
    const blender = state.data?.capabilities?.blender;
    const hasBlender = typeof blender === "object" ? Boolean(blender.available) : Boolean(blender);
    const dirty = visualSettingsDirty();
    $('render').disabled = working || !state.project || !hasBlender || dirty;
    $('animate').disabled = working || !state.project || !hasBlender || dirty;
    $('composite').disabled = working || !state.project?.render_url || !$('photo').files?.length;
    $('render-note').textContent = dirty ? "Camera or style settings changed. Apply style & camera before rendering or animating; existing outputs use the saved settings." : hasBlender
      ? "Blender rendering is optional and runs separately from the scientific analysis."
      : "Blender is not available to this application. The preview and exported scene remain usable.";
    const job = state.data?.job;
    $('job').hidden = !job || !["running", "queued", "cancelling", "failed", "interrupted", "cancelled"].includes(job.state);
    if (job) {
      $('job-title').textContent = `${human(job.kind || "Workspace")} · ${human(job.stage || job.state)}`;
      $('job-message').textContent = job.message || job.error || "";
      const value = typeof job.progress === "number" ? job.progress : job.progress?.fraction;
      if (finite(value)) { $('progress').max = value > 1 ? 100 : 1; $('progress').value = value; }
      else $('progress').removeAttribute("value");
      $('progress').hidden = !["running", "queued", "cancelling"].includes(job.state);
      $('cancel').hidden = !["running", "queued"].includes(job.state);
      $('cancel').disabled = state.submitting;
    }
  }
  async function submit(path, body, action) {
    if (busy()) return;
    clearError(); state.submitting = true; state.pendingAction = action || path;
    updateControls();
    try {
      const data = await api(path, body);
      state.submitting = false;
      if (data.job || data.last_scan || data.projects) show(data);
      await poll(true);
    } catch (error) { state.submitting = false; state.pendingAction = null; showError(error); }
    updateControls();
  }

  function sourceKind(scan = state.scan) {
    const source = scan?.source || {};
    return scan?.source_kind === "synthetic" || source.source_kind === "synthetic" || source.kind === "synthetic" || scan?.demo === true ? "synthetic" : "native";
  }
  function show(data) {
    state.data = {...state.data, ...data};
    if (!state.pinnedScan && data.last_scan) state.scan = data.last_scan;
    const scanKey = JSON.stringify(state.scan);
    if (scanKey !== state.scanKey) { state.scanKey = scanKey; renderScan(); }
    const historyValue = state.pinnedScan || "";
    const historyOptions = (state.data.scans || []).map((scan) => ({value: scan.scan_id,
      label: `${utc(scan.created_at)} · ${human(scan.status)} · ${scan.candidate_count || 0} hotspots`}));
    syncOptions($('history'), [{value: "", label: "Current scan"}, ...historyOptions], historyValue);
    const projects = state.data.projects || [];
    syncOptions($('projects'), [{value: "", label: "New imagined event"}, ...projects.map((project) => ({
      value: project.project_id, label: `${project.title || human(project.kind)} · ${project.active_view || (project.composite_url ? "photo" : project.animation_url ? "animation" : project.render_url ? "render" : project.style || "preview")} · ${project.project_id.slice(0, 6)}`}))], state.project?.project_id || "");
    const job = state.data.job;
    if (job?.state === "complete" && job.job_id !== state.seenJob) {
      state.seenJob = job.job_id;
      if (state.pendingAction && !["/scan", "scan", "/cancel"].includes(state.pendingAction) && projects.length) {
        state.project = projects[0]; loadRecipe(state.project); renderProject(); navigate("studio");
      }
      state.pendingAction = null;
    }
    if (state.project) {
      state.project = projects.find((project) => project.project_id === state.project.project_id) || state.project;
      renderProject();
    }
    if (data.available === false) showError(data.error || data.unavailable_reason || "The workspace's optional scientific dependencies are unavailable.");
    updateControls();
  }
  function syncOptions(select, options, selected) {
    const signature = JSON.stringify(options);
    if (select.dataset.signature !== signature) {
      select.replaceChildren(...options.map(({value, label}) => {
        const option = node("option", label); option.value = value; return option;
      }));
      select.dataset.signature = signature;
    }
    select.value = selected;
  }
  async function poll(immediate = false) {
    if (polling) return;
    polling = true; clearTimeout(timer);
    try { show(await api("")); }
    catch (error) { if (immediate || !state.data) showError(error); }
    finally { polling = false; timer = setTimeout(poll, busy() ? 1800 : 12000); }
  }

  function renderScan() {
    const scan = state.scan;
    const candidates = Array.isArray(scan?.candidates) ? scan.candidates.slice(0, 3) : [];
    const synthetic = sourceKind(scan) === "synthetic";
    $('count').textContent = `${candidates.length} hotspot${candidates.length === 1 ? "" : "s"}`;
    $('map-label').textContent = scan ? synthetic ? "SYNTHETIC DEMO" : scan.map?.field === "tec" ? "LATEST TEC" : human(scan.map?.field || "native TEC") : "Waiting for a scan";
    $('map-label').classList.toggle("is-synthetic", synthetic);
    let summary = "No scan yet. A clear result can contain no hotspots; missing evidence is never treated as quiet.";
    if (scan) {
      const status = String(scan.status || "complete");
      summary = status === "no_usable_data" ? "No usable native observations are available in this window. No hotspot or quiet-state claim can be made."
        : status.includes("insufficient") ? "Not enough supported history for this baseline. No hotspot claims are made. Pull more history or inspect the coverage."
        : candidates.length ? `${candidates.length} distinct candidate${candidates.length === 1 ? "" : "s"} passed the survey gates. Select one to inspect the evidence.`
          : "No candidates passed this survey's gates. This is not a guarantee that the ionosphere is undisturbed.";
      if (synthetic) summary = `Synthetic demonstration — not acquired observations. ${summary}`;
      const baseline = scan.baseline;
      if (baseline?.method) summary += ` Baseline: ${human(baseline.method)}.`;
    }
    $('scan-summary').textContent = summary;
    const source = scan?.source || {};
    const end = source.end || source.time_end || scan?.map?.epoch;
    $('freshness').textContent = scan ? `${synthetic ? "Synthetic timestamps" : "Latest observation"}: ${utc(end)}${synthetic ? "" : ` · ${age(end)}`}. ${finite(source.cadence_seconds) ? `${number(source.cadence_seconds / 60)} min source cadence. ` : ""}${finite(source.native_lon_spacing_deg) && finite(source.native_lat_spacing_deg) ? `${number(source.native_lon_spacing_deg)}° longitude × ${number(source.native_lat_spacing_deg)}° latitude. ` : ""}Interpolation adds no observations.`
      : "Daily CODE products, not a live sensor feed. Observation time and source support will appear here.";
    $('candidate-list').replaceChildren();
    candidates.forEach((candidate, index) => {
      const button = node("button", undefined, "work-candidate"); button.type = "button";
      const meta = node("span", undefined, "work-candidate-meta");
      meta.append(node("span", String(index + 1), "work-candidate-number"), node("span", human(candidate.evidence_strength || "candidate evidence")));
      button.append(meta, node("strong", candidate.title || `${human(candidate.polarity)} TEC change`),
        node("span", `${signed(candidate.peak_deviation_tecu)} TECU deviation · ${number(candidate.native_cell_count, 0)} native cells`),
        node("span", `${utc(candidate.peak_time || candidate.end_time)} →`));
      button.addEventListener("click", () => { state.candidate = candidate; renderEvent(); navigate("event", true); });
      $('candidate-list').append(button);
    });
    const warnings = (scan?.warnings || []).map((warning) => typeof warning === "string" ? warning : warning.message || human(warning.code));
    $('warnings').replaceChildren(...warnings.map((warning) => node("li", warning)));
    $('warnings').hidden = !warnings.length;
    if (state.candidate && !candidates.some((candidate) => candidate.candidate_id === state.candidate.candidate_id)) {
      state.candidate = null; $('event-content').hidden = true; $('event-empty').hidden = false;
    }
    drawMap(scan?.map, candidates);
  }

  const continents = [
    [[-168,66],[-150,70],[-130,55],[-125,40],[-114,30],[-98,18],[-82,8],[-78,10],[-88,22],[-80,26],[-81,34],[-65,47],[-55,52],[-65,62],[-95,73],[-125,72],[-168,66]],
    [[-81,12],[-67,10],[-50,0],[-35,-7],[-45,-23],[-54,-35],[-67,-55],[-75,-40],[-72,-15],[-81,0],[-81,12]],
    [[-17,36],[10,37],[33,31],[43,12],[50,10],[42,-13],[32,-31],[18,-35],[10,-20],[8,3],[-12,5],[-17,20],[-17,36]],
    [[-10,36],[-10,55],[10,60],[30,70],[60,72],[110,75],[170,63],[180,55],[150,48],[140,35],[120,25],[107,0],[95,6],[80,9],[70,25],[45,30],[32,42],[20,40],[-10,36]],
    [[113,-22],[130,-12],[143,-12],[154,-27],[145,-39],[130,-33],[115,-35],[113,-22]],
    [[-53,60],[-42,60],[-20,76],[-35,83],[-60,80],[-70,70],[-53,60]],
  ];
  function paintBase(context, width, height) {
    context.fillStyle = "#172c47"; context.fillRect(0, 0, width, height);
    context.lineWidth = 1; context.strokeStyle = "rgba(209,224,234,.16)";
    for (let longitude = -180; longitude <= 180; longitude += 30) {
      const x = (longitude + 180) / 360 * width; context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    }
    for (let latitude = -60; latitude <= 60; latitude += 30) {
      const y = (90 - latitude) / 180 * height; context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
    }
  }
  function wrappedRect(context, method, x, y, width, height, mapWidth) {
    context[method](x, y, width, height);
    if (x < 0) context[method](x + mapWidth, y, width, height);
    if (x + width > mapWidth) context[method](x - mapWidth, y, width, height);
  }
  function drawMap(map, candidates) {
    const canvas = $('map'), context = canvas.getContext("2d"); if (!context) return;
    const {width, height} = canvas;
    paintBase(context, width, height);
    const latitudes = map?.latitudes || [], longitudes = map?.longitudes || [], values = map?.values || [];
    const finiteValues = values.flat().filter(finite);
    const lower = finiteValues.length ? Math.min(...finiteValues) : 0;
    const upper = finiteValues.length ? Math.max(...finiteValues) : 1;
    const dy = latitudes.length > 1 ? Math.abs(latitudes[1] - latitudes[0]) : 2.5;
    const dx = longitudes.length > 1 ? Math.abs(longitudes[1] - longitudes[0]) : 5;
    for (let row = 0; row < latitudes.length; row++) for (let column = 0; column < longitudes.length; column++) {
      const value = values[row]?.[column]; if (!finite(value)) continue;
      const t = upper === lower ? .5 : Math.max(0, Math.min(1, (value - lower) / (upper - lower)));
      context.fillStyle = `rgb(${Math.round(34 + 201 * t)},${Math.round(71 + 96 * t)},${Math.round(113 - 47 * t)})`;
      wrappedRect(context, "fillRect", (longitudes[column] - dx / 2 + 180) / 360 * width, (90 - latitudes[row] - dy / 2) / 180 * height, dx / 360 * width + .5, dy / 180 * height + .5, width);
    }
    context.strokeStyle = "rgba(238,243,238,.65)"; context.lineWidth = 1.3;
    for (const outline of continents) {
      context.beginPath(); outline.forEach(([lon, lat], index) => context[index ? "lineTo" : "moveTo"]((lon + 180) / 360 * width, (90 - lat) / 180 * height)); context.stroke();
    }
    context.font = "12px sans-serif"; context.textAlign = "left"; context.fillStyle = "rgba(246,249,251,.8)";
    for (const latitude of [-60, 0, 60]) context.fillText(`${Math.abs(latitude)}°${latitude > 0 ? "N" : latitude < 0 ? "S" : ""}`, 6, (90 - latitude) / 180 * height - 6);
    candidates.forEach((candidate, index) => {
      const center = candidate.centroid || {};
      if (!finite(center.latitude) || !finite(center.longitude)) return;
      for (const cell of candidate.footprint || []) {
        if (!finite(cell.latitude) || !finite(cell.longitude)) continue;
        context.strokeStyle = "#fff7c5"; context.lineWidth = 2;
        wrappedRect(context, "strokeRect", (cell.longitude - dx / 2 + 180) / 360 * width, (90 - cell.latitude - dy / 2) / 180 * height, dx / 360 * width, dy / 180 * height, width);
      }
      const x = (center.longitude + 180) / 360 * width, y = (90 - center.latitude) / 180 * height;
      context.beginPath(); context.arc(x, y, 13, 0, Math.PI * 2); context.fillStyle = "#fff1d5"; context.fill();
      context.strokeStyle = "#17243c"; context.stroke(); context.fillStyle = "#17243c"; context.textAlign = "center"; context.font = "bold 13px sans-serif"; context.fillText(String(index + 1), x, y + 4);
    });
    if (!finiteValues.length) {
      context.fillStyle = "rgba(13,30,50,.77)"; context.fillRect(width / 2 - 170, height / 2 - 22, 340, 44);
      context.fillStyle = "#e8edf1"; context.font = "15px sans-serif"; context.textAlign = "center";
      context.fillText(map ? "No supported cells in this map" : "Pull data to begin a native-grid survey", width / 2, height / 2 + 5);
    }
    $('map-range').textContent = "Global latitude / longitude · coast outlines are approximate";
    $('map-legend').textContent = finiteValues.length ? `${number(lower)} → ${number(upper)} TECU · blue → amber` : "Blue: lower · amber: higher";
    canvas.setAttribute("aria-label", finiteValues.length ? `${human(map.field)} at ${utc(map.epoch)}; ${number(lower)} to ${number(upper)} TECU, ${candidates.length} labeled hotspots. Use candidate cards for details.` : "Global survey with approximate geographic outlines; no scan data loaded.");
  }

  function renderEvent() {
    const candidate = state.candidate; if (!candidate) return;
    $('event-empty').hidden = true; $('event-content').hidden = false;
    $('event-origin').textContent = sourceKind() === "synthetic" ? "SYNTHETIC DEMONSTRATION" : "OBSERVATIONAL CANDIDATE";
    $('event-origin').classList.toggle("is-synthetic", sourceKind() === "synthetic");
    $('event-title').textContent = candidate.title || `${human(candidate.polarity)} TEC change`;
    $('event-time').textContent = `${utc(candidate.start_time)} — ${utc(candidate.end_time)} · peak ${utc(candidate.peak_time)}`;
    $('event-metrics').replaceChildren(...[
      ["Peak deviation", `${signed(candidate.peak_deviation_tecu)} TECU`], ["Native cells", number(candidate.native_cell_count, 0)],
      ["Duration", `${number(candidate.duration_hours)} hours`], ["Evidence", human(candidate.evidence_strength)],
    ].map(([label, value]) => { const item = node("div"); item.append(node("dt", label), node("dd", value)); return item; }));
    const series = candidate.time_series || [];
    $('series-table').replaceChildren(...series.map((point) => {
      const row = node("tr"); for (const value of [utc(point.time), number(point.observed_tecu), number(point.baseline_tecu), signed(point.deviation_tecu)]) row.append(node("td", value)); return row;
    }));
    drawSeries(series);
    $('hypotheses').replaceChildren(...(candidate.hypotheses || []).map((hypothesis) => {
      const item = node("article", undefined, "work-hypothesis");
      item.append(node("strong", hypothesis.label || "Unknown explanation"), node("p", listText(hypothesis.evidence)), node("p", listText(hypothesis.limitations), "work-subtle")); return item;
    }));
    if (!candidate.hypotheses?.length) $('hypotheses').append(node("p", "No physical explanation is established from this TEC survey alone.", "work-subtle"));
    const source = state.scan?.source || {};
    const center = candidate.centroid || {};
    $('event-support').textContent = `Footprint center ${number(center.latitude, 2)}° latitude, ${number(center.longitude, 2)}° longitude. ${number(candidate.native_cell_count, 0)} native source cells. ${finite(source.cadence_seconds) ? `${number(source.cadence_seconds / 60)} minute cadence. ` : ""}${finite(source.native_lon_spacing_deg) ? `${number(source.native_lon_spacing_deg)}° longitude × ${number(source.native_lat_spacing_deg)}° latitude native spacing. ` : ""}Source RMS at peak: ${finite(candidate.source_rms_tecu_at_peak) ? `${number(candidate.source_rms_tecu_at_peak, 2)} TECU` : "unknown"}. ${listText(candidate.quality || candidate.source_quality || "Coverage and residual support are not a calibrated probability of a physical disturbance.")}`;
    $('event-provenance').textContent = JSON.stringify({scan_id: state.scan?.scan_id, candidate_id: candidate.candidate_id,
      source, baseline: state.scan?.baseline, bounds: candidate.bounds, evidence: candidate.evidence || candidate.provenance || null}, null, 2);
    drawFootprint(candidate);
    updateControls();
  }
  function listText(value) { return Array.isArray(value) ? value.join(" · ") : typeof value === "object" && value !== null ? JSON.stringify(value) : String(value || ""); }
  function drawSeries(series) {
    const canvas = $('series'), context = canvas.getContext("2d"); if (!context) return;
    const {width, height} = canvas, left = 52, right = width - 18, top = 18, bottom = height - 36;
    context.clearRect(0, 0, width, height);
    const points = series.map((point, index) => ({...point, index, timestamp: timeMillis(point.time)}));
    const pool = points.flatMap((point) => [point.observed_tecu, point.baseline_tecu]).filter(finite);
    if (!pool.length) { context.fillStyle = "#6f7b8e"; context.font = "15px sans-serif"; context.fillText("No supported time-series values", left, height / 2); return; }
    const min = Math.min(...pool), max = Math.max(...pool), padding = Math.max((max - min) * .12, 1);
    const first = points[0].timestamp, last = points.at(-1).timestamp;
    const realTime = Number.isFinite(first) && Number.isFinite(last) && last > first;
    const x = (point) => left + (realTime ? (point.timestamp - first) / (last - first) : point.index / Math.max(1, points.length - 1)) * (right - left);
    const y = (value) => bottom - (value - min + padding) / (max - min + padding * 2) * (bottom - top);
    context.font = "12px sans-serif"; context.textAlign = "right";
    for (let tick = 0; tick <= 4; tick++) {
      const value = min - padding + tick / 4 * (max - min + padding * 2);
      context.strokeStyle = "#e2e5e8"; context.beginPath(); context.moveTo(left, y(value)); context.lineTo(right, y(value)); context.stroke();
      context.fillStyle = "#6f7b8e"; context.fillText(number(value), left - 8, y(value) + 4);
    }
    for (const [field, color] of [["baseline_tecu", "#cc873c"], ["observed_tecu", "#183762"]]) {
      context.strokeStyle = color; context.lineWidth = 2.3; context.beginPath(); let start = true, previousTime = null;
      for (const point of points) {
        if (!finite(point[field]) || (realTime && !Number.isFinite(point.timestamp))) { start = true; previousTime = null; continue; }
        const cadence = state.scan?.source?.cadence_seconds;
        if (previousTime !== null && finite(cadence) && point.timestamp - previousTime > cadence * 1500) start = true;
        context[start ? "moveTo" : "lineTo"](x(point), y(point[field])); start = false; previousTime = point.timestamp;
      }
      context.stroke();
    }
    context.fillStyle = "#6f7b8e"; context.textAlign = "left"; context.fillText(utc(points[0].time), left, height - 10);
    context.textAlign = "right"; context.fillText(utc(points.at(-1).time), right, height - 10);
  }
  function drawFootprint(candidate) {
    const canvas = $('footprint'), context = canvas.getContext("2d"); if (!context) return;
    const {width, height} = canvas; context.clearRect(0, 0, width, height);
    const cells = (candidate.footprint || []).filter((cell) => finite(cell.latitude) && finite(cell.longitude));
    if (!cells.length) { context.fillStyle = "#6f7b8e"; context.font = "14px sans-serif"; context.fillText("No footprint coordinates supplied.", 20, 60); return; }
    const center = candidate.centroid?.longitude || 0;
    const wrap = (longitude) => center + ((longitude - center + 540) % 360 - 180);
    const lons = cells.map((cell) => wrap(cell.longitude)), lats = cells.map((cell) => cell.latitude);
    const west = Math.min(...lons) - 7.5, east = Math.max(...lons) + 7.5, south = Math.max(-90, Math.min(...lats) - 5), north = Math.min(90, Math.max(...lats) + 5);
    const x = (longitude) => 60 + (longitude - west) / (east - west) * (width - 90);
    const y = (latitude) => 15 + (north - latitude) / (north - south) * (height - 60);
    context.fillStyle = "#eff3f6"; context.fillRect(60, 15, width - 90, height - 60);
    context.strokeStyle = "#d2dbe2"; context.fillStyle = "#6f7b8e"; context.font = "12px sans-serif";
    for (let tick = 0; tick <= 4; tick++) {
      const lon = west + (east - west) * tick / 4, lat = south + (north - south) * tick / 4;
      context.beginPath(); context.moveTo(x(lon), 15); context.lineTo(x(lon), height - 45); context.stroke();
      context.textAlign = "center"; context.fillText(`${number(((lon + 540) % 360) - 180)}°`, x(lon), height - 23);
      context.beginPath(); context.moveTo(60, y(lat)); context.lineTo(width - 30, y(lat)); context.stroke();
      context.textAlign = "right"; context.fillText(`${number(lat)}°`, 52, y(lat) + 4);
    }
    context.fillStyle = "rgba(219,141,54,.65)"; context.strokeStyle = "#935a20";
    const dx = state.scan?.source?.native_lon_spacing_deg || 5, dy = state.scan?.source?.native_lat_spacing_deg || 2.5;
    for (const cell of cells) {
      const px = x(wrap(cell.longitude) - dx / 2), py = y(cell.latitude + dy / 2), w = dx / (east - west) * (width - 90), h = dy / (north - south) * (height - 60);
      context.fillRect(px, py, w, h); context.strokeRect(px, py, w, h);
    }
  }

  function scenarioParameters() {
    const fields = {amplitude_tecu: "amplitude", width_km: "width", speed_m_s: "speed", bearing_deg: "bearing", period_minutes: "period",
      duration_minutes: "duration", extent_km: "extent", spacing_km: "spacing", seed: "seed", center_latitude: "latitude", center_longitude: "longitude"};
    const values = {kind: $('kind').value, ...Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, Number($(id).value)]))};
    if (state.inherited?.candidateId === state.recipeCandidateId) {
      for (const [key, value] of Object.entries(state.inherited.parameters)) if (values[key] === value) delete values[key];
    }
    return values;
  }
  function is3D() { return $('model-mode').value === "imagined_3d"; }
  function imaginationParameters() {
    // A saved recipe can contain non-default resolution, altitude or duration
    // supplied through the research API. Compact controls must not erase them.
    const parameters = {...(state.project?.kind === "imagined_3d" ? state.project.parameters : {}),
      seed: Number($('imagine-seed').value)};
    if ($('imagine-kind').value !== "auto") parameters.kind = $('imagine-kind').value;
    else delete parameters.kind;
    for (const [key, id] of [["width_km", "width"], ["speed_m_s", "speed"], ["bearing_deg", "bearing"], ["period_minutes", "period"]]) {
      if ($(`imagine-${id}`).value !== "") parameters[key] = Number($(`imagine-${id}`).value);
      else delete parameters[key];
    }
    // Reuse estimates through their immutable evidence, not as new overrides.
    if (state.project?.kind === "imagined_3d" && state.project.candidate_id === state.recipeCandidateId) {
      for (const [key, value] of Object.entries(parameters))
        if (state.project.provenance?.[key]?.status === "estimated" && value === state.project.parameters?.[key]) delete parameters[key];
    }
    return parameters;
  }
  function modelingMode() {
    const volume = is3D();
    // Orbit and ground-observer pitch have different purposes. Never carry a
    // saved orbit's downward view into observation art (or vice versa).
    if (state.cameraMode !== $('model-mode').value) {
      for (const [id, value] of Object.entries({heading: volume ? -35 : 0, pitch: volume ? -18 : 90,
        roll: 0, fov: volume ? 55 : 70, latitude: "", longitude: "", altitude: 0})) $(`camera-${id}`).value = value;
      state.cameraMode = $('model-mode').value;
    }
    $('3d-fields').hidden = !volume; $('3d-fields').disabled = !volume;
    $('2d-fields').hidden = volume; $('2d-fields').disabled = volume;
    $('camera-auto-label').hidden = !volume;
    const automatic = volume && $('camera-auto').checked;
    for (const id of ["heading", "pitch", "roll", "fov"]) $(`camera-${id}`).disabled = automatic;
    for (const id of ["latitude", "longitude", "altitude"]) {
      $(`camera-${id}`).disabled = volume; $(`camera-${id}`).closest("label").hidden = volume;
    }
    $('camera-note').textContent = volume
      ? "3D uses an automatically distanced orbit, not a calibrated ground observer. Uncheck automatic framing to choose heading, pitch, roll and FOV."
      : "Leave coordinates blank to use the scene center. These are manual artistic alignment settings, not inferred observer positions.";
    $('create').textContent = volume ? "Imagine in 3D & preview" : "Create 2D scenario & preview";
    validateRecipe(); updateControls();
  }
  function loadRecipe(project) {
    state.inherited = null; state.recipeCandidateId = project.candidate_id || null;
    const volume = project.kind === "imagined_3d";
    $('model-mode').value = volume ? "imagined_3d" : "scenario";
    state.cameraMode = $('model-mode').value;
    $('camera-auto').checked = !project.camera;
    if (volume) {
      $('imagine-kind').value = project.parameters?.kind || "auto";
      $('imagine-seed').value = project.parameters?.seed ?? 42;
      for (const [key, id] of [["width_km", "width"], ["speed_m_s", "speed"], ["bearing_deg", "bearing"], ["period_minutes", "period"]])
        $(`imagine-${id}`).value = project.parameters?.[key] ?? "";
    }
    const fields = {kind: "kind", amplitude_tecu: "amplitude", width_km: "width", speed_m_s: "speed", bearing_deg: "bearing", period_minutes: "period",
      duration_minutes: "duration", extent_km: "extent", spacing_km: "spacing", seed: "seed", center_latitude: "latitude", center_longitude: "longitude"};
    for (const [key, id] of Object.entries(fields)) if (!volume && project.parameters?.[key] !== undefined) $(id).value = project.parameters[key];
    if (["ghost", "luminous", "quiet"].includes(project.style)) $('style').value = project.style;
    for (const [key, id, fallback] of [["heading_deg", "camera-heading", volume ? -35 : 0], ["pitch_deg", "camera-pitch", volume ? -18 : project.camera ? 45 : 90], ["roll_deg", "camera-roll", 0],
      ["horizontal_fov_deg", "camera-fov", volume ? 55 : 70], ["observer_latitude", "camera-latitude", ""], ["observer_longitude", "camera-longitude", ""], ["observer_altitude_km", "camera-altitude", 0]])
      $(id).value = finite(project.camera?.[key]) ? project.camera[key] : fallback;
    for (const [key, id, fallback] of [["opacity", "opacity", .65], ["horizon_y", "horizon", .5], ["horizon_fade", "fade", .6],
      ["saturation", "saturation", .85], ["exposure", "exposure", 1]])
      $(`photo-${id}`).value = finite(project.photo_settings?.[key]) ? project.photo_settings[key] : fallback;
    for (const [index, id] of ["red", "green", "blue"].entries())
      $(`photo-${id}`).value = finite(project.photo_settings?.grade_rgb?.[index]) ? project.photo_settings.grade_rgb[index] : 1;
    $('opacity-label').textContent = Number($('photo-opacity').value).toFixed(2);
    const synthetic = Boolean(project.scenario_id) || /synthetic|scenario/.test(project.kind || "");
    $('scenario-context').textContent = volume ? "IMAGINED 3D EVENT · Evidence-linked hypothesis, chosen altitude and motion, dimensionless synthetic material. Not a reconstruction or a confirmed diagnosis. Local simulation; IFM is reserved for v0.3." : synthetic
      ? "HYPOTHETICAL SCENARIO · Mathematical TEC fields, not a physical reconstruction. Parameter provenance below records the source link and explicitly chosen assumptions."
      : "OBSERVATION-DERIVED ART · This preview uses acquired evidence; colors, altitude and camera remain artistic. The separate recipe at left can create a hypothetical scenario, not refine the observations.";
    modelingMode();
  }
  function provenanceHints() {
    for (const [key, id] of [["amplitude_tecu", "amplitude"], ["center_latitude", "latitude"], ["center_longitude", "longitude"]]) {
      const inherited = state.inherited?.candidateId === state.recipeCandidateId && state.inherited?.parameters[key] === Number($(id).value);
      $(`${id}-provenance`).textContent = inherited ? "estimated from source" : "chosen";
    }
  }
  function validateRecipe() {
    if (is3D()) {
      $('budget').textContent = "Local 3D model · default 41 × 41 × 25 cells × 12 timestamps. Server validates memory and numerical-work budgets before queuing.";
      return;
    }
    const nx = Math.floor(Number($('extent').value) / Number($('spacing').value)) + 1;
    const nt = Math.floor(Number($('duration').value) / 5) + 1;
    const count = nx * nx * nt;
    const invalid = count > 200000 || nx < 3;
    $('extent').setCustomValidity(count > 200000 ? "This exceeds the 200,000-cell laptop budget. Increase spacing or reduce extent/duration." : nx < 3 ? "Use at least three grid points across the extent." : "");
    $('width').setCustomValidity(Number($('width').value) > Number($('extent').value) ? "Width cannot exceed the experiment extent." : "");
    $('amplitude').max = $('kind').value === "localized_depletion" ? "20" : "50";
    $('budget').textContent = `${number(count, 0)} cells · ${nx} × ${nx} × ${nt} frames · ${invalid ? "over the supported budget; adjust the recipe" : "within the 200,000-cell laptop budget"}.`;
    provenanceHints();
  }
  function camera() {
    if (is3D() && $('camera-auto').checked) return null;
    const fields = {heading_deg: "heading", pitch_deg: "pitch", roll_deg: "roll", horizontal_fov_deg: "fov",
      observer_latitude: "latitude", observer_longitude: "longitude", observer_altitude_km: "altitude"};
    return Object.fromEntries(Object.entries(fields).filter(([, id]) => !$(`camera-${id}`).disabled && $(`camera-${id}`).value !== "")
      .map(([key, id]) => [key, Number($(`camera-${id}`).value)]));
  }
  function visualSettingsDirty() {
    if (!state.project) return false;
    if ((state.project.kind === "imagined_3d") !== is3D()) return true;
    if (is3D()) {
      const current = camera(), saved = state.project.camera || null;
      if (!current || !saved) return $('style').value !== state.project.style || current !== saved;
      const defaults = {heading_deg: -35, pitch_deg: -18, roll_deg: 0, horizontal_fov_deg: 55};
      return $('style').value !== state.project.style || Object.keys(defaults).some((key) => (current[key] ?? defaults[key]) !== (saved[key] ?? defaults[key]));
    }
    const saved = {heading_deg: 0, pitch_deg: state.project.camera ? 45 : 90, roll_deg: 0,
      horizontal_fov_deg: 70, observer_altitude_km: 0, ...state.project.camera};
    const current = camera();
    return $('style').value !== state.project.style || [...new Set([...Object.keys(saved), ...Object.keys(current)])]
      .some((key) => saved[key] !== current[key]);
  }
  function validCamera() {
    for (const input of document.querySelectorAll('input[id^="work-camera-"]')) {
      if (input.checkValidity()) continue;
      const disclosure = input.closest("details"); if (disclosure) disclosure.open = true;
      input.reportValidity(); return false;
    }
    return true;
  }
  function renderProject() {
    const project = state.project; if (!project) return;
    const key = JSON.stringify(project); if (state.projectKey === key) return; state.projectKey = key;
    $('projects').value = project.project_id;
    const synthetic = Boolean(project.scenario_id) || /synthetic|scenario/.test(project.kind || "");
    const volume = project.kind === "imagined_3d";
    $('project-kind').textContent = volume ? "IMAGINED 3D / ART" : synthetic ? "SYNTHETIC SCENARIO / ART" : "OBSERVATION-DERIVED ART";
    $('project-kind').classList.toggle("is-synthetic", synthetic);
    $('preview-title').textContent = project.title || "Your scene";
    $('project-actions').hidden = false;
    const view = ["preview", "render", "animation", "composite"].includes(project.active_view) ? project.active_view
      : project.composite_url ? "composite" : project.animation_url ? "animation" : project.render_url ? "render" : "preview";
    const preview = safeAsset(project[`${view}_url`]) || safeAsset(project.preview_url);
    $('preview-figure').hidden = !preview; $('preview-empty').hidden = Boolean(preview);
    if (preview) $('preview-image').src = preview;
    else $('preview-image').removeAttribute("src");
    $('preview-caption').textContent = `${view === "composite" ? "Manual artistic photo overlay" : view === "animation" ? "Short artistic animation" : view === "render" ? "Blender render" : volume ? "Depth-integrated 3D preview · Blender adds geometric ribbons and filaments" : "Flat artistic material preview · not camera geometry"} · ${synthetic ? "Chosen synthetic details, not recovered fine-scale measurements." : "Colors, shell altitude and visual displacement are artistic choices."}`;
    const analysis = project.analysis_summary;
    $('analysis-note').textContent = volume ? `${human(project.parameters?.kind)} · ${project.parameters?.frame_count || 12} simulated timestamps. ${project.provenance?.kind?.reason || "Chosen scenario, not a diagnosis."} Evolving synthetic material with chosen advection, diffusion and forcing—not plasma electrodynamics.` : analysis
      ? `Selected-frame evidence · motion: ${human(analysis.flow_status)} · waves: ${human(analysis.wave_status)} · source uncertainty: ${human(analysis.source_uncertainty_status || "unknown")}. ${analysis.timeline_frames || 0} independently assessed animation timestamps. Confidence is heuristic, not a probability.`
      : "Legacy project: single-target evidence. Recreate the scenario or artify the event again for time-specific animation states.";
    $('project-links').replaceChildren();
    for (const [label, field] of [["Science report", "science_report_url"], ["Art report", "visual_report_url"], ["Download scene", "export_url"], ["Recipe JSON", "recipe_url"], ["Open render", "render_url"], ["Photo overlay", "composite_url"], ["Animation GIF", "animation_url"], ["Animation frames", "animation_export_url"]]) {
      const url = safeAsset(project[field]); if (!url) continue;
      const link = node("a", label); link.href = url; link.target = "_blank"; link.rel = "noopener";
      if (field === "export_url") link.download = "";
      $('project-links').append(link);
    }
    $('project-provenance').textContent = JSON.stringify({project_id: project.project_id, kind: project.kind, candidate_id: project.candidate_id,
      scenario_id: project.scenario_id, representative_frame: project.representative_frame,
      parameters: project.parameters, provenance: project.provenance, camera: project.camera,
      analysis_summary: project.analysis_summary, photo_settings: project.photo_settings,
      hypothesis_semantics: project.hypothesis_semantics, evidence_sha256: project.evidence_sha256, ifm: project.ifm}, null, 2);
  }

  $('pull').addEventListener("click", () => { state.pinnedScan = null; submit("/scan", {source: "latest", days: 8}, "scan"); });
  $('cached').addEventListener("click", () => { state.pinnedScan = null; submit("/scan", {source: "cached", days: 8}, "scan"); });
  $('demo').addEventListener("click", () => { state.pinnedScan = null; submit("/scan", {source: "demo", days: 8}, "scan"); });
  $('cancel').addEventListener("click", async () => {
    if (!state.data?.job?.job_id || state.submitting) return;
    state.submitting = true; updateControls();
    try { const data = await api("/cancel", {job_id: state.data.job.job_id}); state.submitting = false; show(data); await poll(true); }
    catch (error) { state.submitting = false; showError(error); updateControls(); }
  });
  $('history').addEventListener("change", async () => {
    const id = $('history').value; clearError(); state.pinnedScan = id || null;
    if (!id) { state.scan = state.data?.last_scan || null; state.scanKey = JSON.stringify(state.scan); renderScan(); return; }
    try { const data = await api(`/scans/${encodeURIComponent(id)}`); state.scan = data.scan; state.scanKey = JSON.stringify(state.scan); renderScan(); }
    catch (error) { showError(error); }
  });
  $('explore').addEventListener("click", () => {
    if (!state.candidate) return;
    state.project = null; state.projectKey = ""; $('projects').value = "";
    $('scenario-form').reset();
    state.cameraMode = null;
    modelingMode();
    $('preview-figure').hidden = true; $('preview-empty').hidden = false; $('project-actions').hidden = true;
    $('preview-title').textContent = "Your scene"; $('project-kind').textContent = "Not generated";
    const center = state.candidate.centroid || {};
    const inherited = {};
    if (finite(center.latitude)) $('latitude').value = Number(Math.max(-90, Math.min(90, center.latitude)).toPrecision(8));
    if (finite(center.longitude)) $('longitude').value = Number(center.longitude.toPrecision(8));
    if (finite(center.latitude)) inherited.center_latitude = Number($('latitude').value);
    if (finite(center.longitude)) inherited.center_longitude = Number($('longitude').value);
    const deviation = state.candidate.peak_deviation_tecu;
    if (finite(deviation)) {
      const depletion = deviation < 0;
      $('kind').value = depletion ? "localized_depletion" : "wave_packet";
      $('amplitude').value = Number(Math.min(Math.abs(deviation), depletion ? 20 : 50).toPrecision(8));
      inherited.amplitude_tecu = Number($('amplitude').value);
    }
    state.recipeCandidateId = state.candidate.candidate_id;
    state.inherited = {candidateId: state.recipeCandidateId, parameters: inherited};
    validateRecipe();
    $('scenario-context').textContent = `IMAGINED 3D HYPOTHESIS · Inspired by “${state.candidate.title || "selected hotspot"}”. Coarse evidence seeds a proposal, not a diagnosis. We choose a possible altitude, motion and structure, then turn that imagined event into art.`;
    navigate("studio", true);
  });
  $('artify').addEventListener("click", () => {
    if (!state.candidate) return;
    navigate("studio", true);
    $('model-mode').value = "scenario"; modelingMode();
    if (!validCamera()) return;
    $('scenario-context').textContent = "OBSERVATION-DERIVED ART · The selected footprint supplies source evidence. Colors, camera, altitude and displacement are artistic choices; this is not a physical view of the ionosphere.";
    submit("/artify", {candidate_id: state.candidate.candidate_id, style: $('style').value, camera: camera()}, "artify");
  });
  $('scenario-form').addEventListener("submit", (event) => {
    event.preventDefault(); if (!$('scenario-form').reportValidity()) return;
    const payload = {parameters: is3D() ? imaginationParameters() : scenarioParameters(), style: $('style').value, camera: camera()};
    if (state.recipeCandidateId) payload.candidate_id = state.recipeCandidateId;
    if (is3D() && state.recipeCandidateId) {
      const originalSource = state.project?.kind === "imagined_3d" && state.project?.candidate_id === state.recipeCandidateId ? state.project.source_project_id : null;
      const source = (state.data?.projects || []).find((project) => project.kind === "observations" && project.candidate_id === state.recipeCandidateId);
      if (originalSource || source) payload.source_project_id = originalSource || source.project_id;
    }
    submit(is3D() ? "/imagine" : "/scenario", payload, is3D() ? "imagine" : "scenario");
  });
  $('model-mode').addEventListener("change", modelingMode);
  $('camera-auto').addEventListener("change", modelingMode);
  for (const id of ["kind", "amplitude", "width", "extent", "spacing", "duration", "latitude", "longitude"]) $(id).addEventListener("input", validateRecipe);
  for (const input of document.querySelectorAll('input[id^="work-camera-"], #work-style')) input.addEventListener("input", updateControls);
  $('restyle').addEventListener("click", () => {
    if (!state.project || !validCamera()) return;
    if ((state.project.kind === "imagined_3d") !== is3D()) { showError("Create a new project to change modeling space; restyling keeps the existing model."); return; }
    submit("/style", {project_id: state.project.project_id, style: $('style').value, camera: camera()}, "style");
  });
  $('render').addEventListener("click", () => {
    if (state.project && !visualSettingsDirty()) submit("/render", {project_id: state.project.project_id}, "render");
  });
  $('animate').addEventListener("click", () => {
    if (!state.project || visualSettingsDirty() || !$('animation-frames').reportValidity()) return;
    submit("/animation", {project_id: state.project.project_id, frame_count: Number($('animation-frames').value)}, "animation");
  });
  $('photo').addEventListener("change", () => {
    clearError();
    const file = $('photo').files?.[0];
    if (file && file.size > 8 * 1024 * 1024) { showError("Choose a PNG or JPEG photograph no larger than 8 MiB."); $('photo').value = ""; }
    $('photo-note').textContent = file && file.size <= 8 * 1024 * 1024 ? `Selected ${file.name}. It has not been uploaded. Create photo overlay to store it locally and composite the rendered frame.` : "Requires a rendered project. Choose only photographs you have permission to use.";
    updateControls();
  });
  $('photo-opacity').addEventListener("input", () => { $('opacity-label').textContent = Number($('photo-opacity').value).toFixed(2); });
  $('composite').addEventListener("click", async () => {
    const file = $('photo').files?.[0];
    const controls = {opacity: "opacity", horizon_y: "horizon", horizon_fade: "fade", saturation: "saturation", exposure: "exposure"};
    const gradeIds = ["red", "green", "blue"];
    if (busy() || !state.project?.render_url || !file ||
        [...Object.values(controls), ...gradeIds].some((id) => !$(`photo-${id}`).reportValidity())) return;
    if (file.size > 8 * 1024 * 1024) { showError("The photograph exceeds the 8 MiB upload limit."); return; }
    // Freeze the target and all inputs before any upload yields to UI changes.
    const targetProjectId = state.project.project_id;
    const photoSettings = {...Object.fromEntries(Object.entries(controls).map(([key, id]) => [key, Number($(`photo-${id}`).value)])),
      grade_rgb: gradeIds.map((id) => Number($(`photo-${id}`).value))};
    const masks = Object.fromEntries(["foreground", "cloud"].map((kind) => [kind, $(`${kind}-mask`).files?.[0]]));
    clearError(); state.submitting = true; updateControls();
    $('photo-note').textContent = "Storing the selected photograph in the local workspace…";
    try {
      const response = await fetch("/api/workspace/photo", {method: "POST", headers: {
        "Content-Type": file.type || "application/octet-stream", "X-OPHANIM-Token": token}, body: file});
      const data = await response.json();
      if (!response.ok || !data.ok || !data.photo?.photo_id) throw new Error(data.error || "The photograph could not be stored.");
      $('photo-note').textContent = `Local photograph stored · ${data.photo.width} × ${data.photo.height}. Preparing artistic overlay…`;
      const payload = {project_id: targetProjectId, photo_id: data.photo.photo_id, ...photoSettings};
      for (const kind of ["foreground", "cloud"]) {
        const mask = masks[kind];
        if (!mask) continue;
        if (mask.size > 8 * 1024 * 1024) throw new Error("Each mask must be no larger than 8 MiB.");
        const maskResponse = await fetch("/api/workspace/mask", {method: "POST", headers: {
          "Content-Type": mask.type || "application/octet-stream", "X-OPHANIM-Token": token}, body: mask});
        const saved = await maskResponse.json();
        if (!maskResponse.ok || !saved.ok || !saved.mask?.mask_id) throw new Error(saved.error || "Mask upload failed.");
        if (saved.mask.width !== data.photo.width || saved.mask.height !== data.photo.height)
          throw new Error("Masks must match the original oriented photograph's dimensions.");
        payload[`${kind}_mask_id`] = saved.mask.mask_id;
      }
      state.submitting = false;
      await submit("/composite", payload, "composite");
    } catch (error) { state.submitting = false; showError(error); updateControls(); }
  });
  $('projects').addEventListener("change", () => {
    state.project = (state.data?.projects || []).find((project) => project.project_id === $('projects').value) || null;
    state.projectKey = "";
    if (state.project) { loadRecipe(state.project); renderProject(); }
    else { state.inherited = null; state.recipeCandidateId = null; $('scenario-form').reset(); state.cameraMode = null; modelingMode();
      $('preview-figure').hidden = true; $('preview-empty').hidden = false; $('project-actions').hidden = true; $('preview-title').textContent = "Your scene"; $('project-kind').textContent = "Not generated";
      $('scenario-context').textContent = "IMAGINED 3D EVENT · Evidence inspires a hypothesis; assumed mechanics create the structure. Not a recovered view of the ionosphere. IFM comes in v0.3."; }
    updateControls();
  });

  const initial = location.hash.slice(1);
  navigate(initial === "shawtynet-panel" ? "advanced" : initial);
  drawMap(null, []); modelingMode(); poll();
})();
