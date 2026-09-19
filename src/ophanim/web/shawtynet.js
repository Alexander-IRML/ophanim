"use strict";

(() => {
  const panel = document.querySelector("#shawtynet-panel");
  if (!panel) return;
  const token = document.querySelector('meta[name="ophanim-token"]')?.content || "";
  const element = (name) => document.querySelector(`#shawtynet-${name}`);
  let timer;
  let submitting = false;
  let lastPreview = "";
  const human = (value) => String(value || "unknown").replaceAll("_", " ");
  const safeAsset = (value) => typeof value === "string" &&
    /^\/shawtynet-assets\/[0-9a-f]{24}\//.test(value) &&
    !value.includes("..") && !value.includes("\\") ? value : "";

  function show(status) {
    const busy = submitting || status.state === "running";
    for (const name of ["demo", "acquired"]) element(name).disabled = busy || !status.available;
    element("status").textContent = status.available ? status.message : status.unavailable_reason;
    element("error").hidden = !status.error;
    element("error").textContent = status.error || "";
    const result = status.result;
    element("result").hidden = !result;
    if (!result) return;
    element("result-origin").textContent = result.source_kind === "synthetic" ? "SYNTHETIC EXPERIMENT" : "ACQUIRED TEC PRODUCT";
    element("result-title").textContent = human(result.event?.display_class || result.event?.dominant_label || "analysis ready");
    element("result-time").textContent = result.target_time
      ? `Retrospective analysis at ${result.target_time}. Source window: ${result.time_start || ""} — ${result.time_end || ""} UTC.`
      : `${result.time_start || ""} — ${result.time_end || ""} UTC`;
    const channels = result.event?.channel_status || {};
    const flowState = typeof channels.flow === "string" ? channels.flow : channels.flow?.status;
    const waveState = typeof channels.wave === "string" ? channels.wave : channels.wave?.status;
    element("flow-state").textContent = human(flowState || result.capabilities?.flow?.status);
    element("wave-state").textContent = human(waveState || result.capabilities?.wave?.status);
    for (const [name, key] of [["science-link", "science_report_url"], ["visual-link", "visual_report_url"]]) {
      const url = safeAsset(result[key]);
      element(name).hidden = !url;
      if (url) element(name).href = url;
      else element(name).removeAttribute("href");
    }
    const preview = safeAsset(result.preview_url);
    element("preview").closest("figure").hidden = !preview;
    if (preview && preview !== lastPreview) {
      element("preview").src = preview;
      lastPreview = preview;
    }
  }

  async function poll() {
    clearTimeout(timer);
    try {
      const response = await fetch("/api/shawtynet/status", {cache: "no-store"});
      const status = await response.json();
      if (!response.ok || !status.ok) throw new Error(status.error || "Research status is unavailable.");
      show(status);
      timer = setTimeout(poll, status.state === "running" ? 2000 : 12000);
    } catch (error) {
      element("status").textContent = error.message;
      timer = setTimeout(poll, 12000);
    }
  }

  async function start(source) {
    if (submitting) return;
    submitting = true;
    element("error").hidden = true;
    element("status").textContent = "Starting research analysis…";
    for (const name of ["demo", "acquired"]) element(name).disabled = true;
    try {
      const body = {source, case: element("case").value};
      if (source === "desktop") {
        body.bounds = Object.fromEntries(["south", "north", "west", "east"].map((name) => [name, Number(element(name).value)]));
      }
      const response = await fetch("/api/shawtynet/analyze", {
        method: "POST", headers: {"Content-Type": "application/json", "X-OPHANIM-Token": token},
        body: JSON.stringify(body),
      });
      const status = await response.json();
      if (!response.ok || !status.ok) throw new Error(status.error || "The analysis could not start.");
      submitting = false;
      show(status);
      clearTimeout(timer);
      timer = setTimeout(poll, 1000);
    } catch (error) {
      submitting = false;
      element("error").textContent = error.message;
      element("error").hidden = false;
      for (const name of ["demo", "acquired"]) element(name).disabled = false;
    }
  }
  element("demo").addEventListener("click", () => start("synthetic"));
  element("acquired").addEventListener("click", () => start("desktop"));
  poll();
})();
