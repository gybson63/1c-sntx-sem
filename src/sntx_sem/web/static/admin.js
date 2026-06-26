const statusContent = document.getElementById("status-content");
const embeddingContent = document.getElementById("embedding-content");
const testEmbeddingBtn = document.getElementById("test-embedding");
const embeddingTestResult = document.getElementById("embedding-test-result");
const runIngestBtn = document.getElementById("run-ingest");
const runIndexBtn = document.getElementById("run-index");
const runIngestBspBtn = document.getElementById("run-ingest-bsp");
const jobPanel = document.getElementById("job-panel");
const jobMeta = document.getElementById("job-meta");
const jobLog = document.getElementById("job-log");

let activeJobId = null;
let logOffset = 0;
let pollTimer = null;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  if (response.status === 204 || !response.headers.get("content-type")?.includes("json")) {
    return null;
  }
  return response.json();
}

function renderStatus(data) {
  const idx = data.index || {};
  const cfg = data.config || {};
  const emb = cfg.embedding || {};
  statusContent.innerHTML = `
    <p><strong>Ready:</strong> ${data.ready ? "да" : "нет"}</p>
    <p><strong>Chunks:</strong> ${idx.indexed_chunks || 0} / ${idx.export_chunks || 0}</p>
    <p><strong>Platform:</strong> ${idx.platform_version || "—"}</p>
    <p><strong>Embedding:</strong> ${emb.provider || "—"} / ${emb.model || "—"}</p>
    <p><strong>Mismatch:</strong> ${idx.embedding_mismatch ? "да — нужен rebuild" : "нет"}</p>
  `;
}

async function loadEmbeddingSettings() {
  const settings = await api("/settings/embedding");
  embeddingContent.innerHTML = `
    <p><strong>Provider:</strong> ${settings.provider}</p>
    <p><strong>Model:</strong> ${settings.model}</p>
    <p><strong>API key:</strong> ${settings.api_key_set ? "задан" : "не задан"}</p>
    <p><strong>Mismatch:</strong> ${settings.embedding_mismatch ? "да" : "нет"}</p>
  `;
}

async function loadStatus() {
  const data = await api("/status");
  renderStatus(data);
  await loadEmbeddingSettings();
}

async function pollJob() {
  if (!activeJobId) return;
  const job = await api(`/jobs/${activeJobId}?since_log=${logOffset}`);
  logOffset = job.log_offset || logOffset;
  jobMeta.textContent = `${job.type} · ${job.status}`;
  if (job.logs?.length) {
    jobLog.textContent += job.logs.join("\n") + "\n";
    jobLog.scrollTop = jobLog.scrollHeight;
  }
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    if (job.error) {
      jobLog.textContent += `\nERROR: ${job.error}\n`;
    }
    await loadStatus();
  }
}

async function startJob(path, body = {}) {
  jobPanel.hidden = false;
  jobLog.textContent = "";
  logOffset = 0;
  const job = await api(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
  activeJobId = job.id;
  jobMeta.textContent = `${job.type} · ${job.status}`;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 1500);
  await pollJob();
}

runIngestBtn.addEventListener("click", () => startJob("/jobs/ingest"));
runIndexBtn.addEventListener("click", () => startJob("/jobs/index", { rebuild: true }));
runIngestBspBtn.addEventListener("click", () => startJob("/jobs/ingest-bsp"));

testEmbeddingBtn.addEventListener("click", async () => {
  embeddingTestResult.hidden = false;
  embeddingTestResult.textContent = "Тест…";
  try {
    const result = await api("/settings/embedding/test", {
      method: "POST",
      body: JSON.stringify({ text: "тестовый запрос" }),
    });
    embeddingTestResult.textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    embeddingTestResult.textContent = `Ошибка: ${err.message}`;
  }
});

loadStatus().catch((err) => {
  statusContent.textContent = `Ошибка: ${err.message}`;
});

api("/status")
  .then((data) => {
    const bspPath = data.config?.bsp?.path;
    if (bspPath) {
      runIngestBspBtn.hidden = false;
    }
  })
  .catch(() => {});
