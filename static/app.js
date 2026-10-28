const $ = (id) => document.getElementById(id);
const countEl = $("count"), statusEl = $("status"), msgEl = $("message"),
      earEl = $("ear"), noticeEl = $("notice"), resetBtn = $("reset"), video = $("video");

function setStatus(text, level) {
  statusEl.textContent = text;
  statusEl.className = "status status-" + level;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    countEl.textContent = d.count;
    earEl.textContent = d.camera_ok && d.face_detected ? d.ear.toFixed(2) : "--";
    msgEl.textContent = d.message || "";
    const isError = d.camera_ok === false || /error/i.test(d.status);
    const isOk = d.camera_ok === true && d.face_detected;
    setStatus(d.status, isError ? "err" : isOk ? "ok" : "warn");
  } catch (e) {
    setStatus("Backend Error", "err");
    msgEl.textContent = "Cannot reach the server. Retrying...";
  }
}

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  noticeEl.textContent = "";
  try {
    const res = await fetch("/api/reset", { method: "POST" });
    const d = await res.json();
    if (!res.ok || !d.success) throw new Error(d.error || "Reset failed");
    countEl.textContent = d.count;          // immediate update, no page reload
  } catch (e) {
    noticeEl.textContent = "Reset failed: " + e.message;
  } finally {
    resetBtn.disabled = false;
  }
});

// If the MJPEG stream drops, reconnect it without reloading the page
video.addEventListener("error", () => {
  setTimeout(() => { video.src = "/video_feed?t=" + Date.now(); }, 2000);
});

pollStatus();
setInterval(pollStatus, 250);
