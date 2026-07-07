const statusContent = document.getElementById("status-content");
const adminAlerts = document.getElementById("admin-alerts");
const embeddingForm = document.getElementById("embedding-form");
const embProvider = document.getElementById("emb-provider");
const embModel = document.getElementById("emb-model");
const embDevice = document.getElementById("emb-device");
const embBaseUrl = document.getElementById("emb-base-url");
const embApiKey = document.getElementById("emb-api-key");
const embApiKeyEnv = document.getElementById("emb-api-key-env");
const embQueryPrefix = document.getElementById("emb-query-prefix");
const embPassagePrefix = document.getElementById("emb-passage-prefix");
const embeddingStatus = document.getElementById("embedding-status");
const saveEmbeddingBtn = document.getElementById("save-embedding");
const testEmbeddingBtn = document.getElementById("test-embedding");
const embeddingTestResult = document.getElementById("embedding-test-result");
const runIngestBtn = document.getElementById("run-ingest");
const runIndexBtn = document.getElementById("run-index");
const runIngestBspBtn = document.getElementById("run-ingest-bsp");
const jobPanel = document.getElementById("job-panel");
const jobMeta = document.getElementById("job-meta");
const jobError = document.getElementById("job-error");
const jobProgress = document.getElementById("job-progress");
const jobLog = document.getElementById("job-log");

const JOB_TYPE_LABELS = {
  ingest: "Ingest HBK + Index",
  index: "Rebuild Index",
  ingest_bsp: "Ingest BSP + Index",
};

const STATUS_LABELS = {
  pending: "ожидание",
  running: "выполняется",
  completed: "завершено",
  failed: "ошибка",
};

const PROVIDER_LABELS = {
  sentence_transformers: "Локально (sentence-transformers / E5)",
  openai_compatible: "OpenAI-compatible API",
  ollama: "Ollama",
};

const DEFAULT_MODELS = {
  sentence_transformers: "intfloat/multilingual-e5-base",
  openai_compatible: "text-embedding-3-small",
  ollama: "nomic-embed-text",
};

const OPENAI_MODELS = new Set([
  "text-embedding-3-small",
  "text-embedding-3-large",
  "text-embedding-ada-002",
]);

const LOCAL_MODELS = new Set([
  "intfloat/multilingual-e5-small",
  "intfloat/multilingual-e5-base",
  "intfloat/multilingual-e5-large",
]);

let activeJobId = null;
let logOffset = 0;
let pollTimer = null;
let pollFailCount = 0;
const POLL_MAX_FAILURES = 30;

function parseErrorMessage(err) {
  const raw = String(err?.message || err || "Неизвестная ошибка");
  const lower = raw.toLowerCase();
  if (
    lower === "failed to fetch" ||
    lower.includes("networkerror") ||
    lower.includes("load failed")
  ) {
    return (
      "Сервер не ответил вовремя — при индексации API может быть временно недоступен. " +
      "Подождите 1–2 минуты и обновите страницу; задача может продолжаться в фоне."
    );
  }
  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      const payload = JSON.parse(jsonMatch[0]);
      if (payload.detail) {
        return typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail);
      }
    } catch {
      // keep raw message
    }
  }
  const statusMatch = raw.match(/^\d+:\s*(.+)$/s);
  return statusMatch ? statusMatch[1].trim() : raw;
}

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

function renderIssues(issues) {
  if (!issues?.length) {
    adminAlerts.hidden = true;
    adminAlerts.innerHTML = "";
    return;
  }
  adminAlerts.hidden = false;
  adminAlerts.innerHTML = issues
    .map((issue) => {
      const severity = issue.severity === "warning" ? "warning" : "error";
      const title = severity === "warning" ? "Предупреждение" : "Ошибка";
      return `<div class="alert alert--${severity}"><strong>${title}</strong>${issue.message}</div>`;
    })
    .join("");
}

function showJobError(message) {
  if (!message) {
    jobError.hidden = true;
    jobError.textContent = "";
    jobLog.classList.remove("has-error");
    return;
  }
  jobError.hidden = false;
  jobError.textContent = message;
  jobLog.classList.add("has-error");
}

function renderStatus(data) {
  const idx = data.index || {};
  const cfg = data.config || {};
  const emb = cfg.embedding || {};
  renderIssues(data.issues);
  statusContent.innerHTML = `
    <p><strong>Ready:</strong> ${data.ready ? "да" : "нет"}</p>
    <p><strong>Chunks:</strong> ${idx.indexed_chunks || 0} / ${idx.export_chunks || 0}</p>
    <p><strong>Platform:</strong> ${idx.platform_version || "—"}</p>
    <p><strong>Embedding:</strong> ${emb.provider || "—"} / ${emb.model || "—"}</p>
    <p><strong>Mismatch:</strong> ${idx.embedding_mismatch ? "да — нужен rebuild" : "нет"}</p>
    <p><strong>LanceDB:</strong> ${idx.lance_present ? "есть" : "нет"}</p>
  `;
}

function suggestedModel(provider, currentModel) {
  const defaults = { ...DEFAULT_MODELS, ...(window.__defaultModels || {}) };
  const fallback = defaults[provider] || "";
  const model = (currentModel || "").trim();
  if (!model) {
    return fallback;
  }
  if (provider === "sentence_transformers" && OPENAI_MODELS.has(model)) {
    return fallback;
  }
  if (
    (provider === "openai_compatible" || provider === "ollama") &&
    LOCAL_MODELS.has(model)
  ) {
    return defaults[provider] || fallback;
  }
  return model;
}

function updateProviderFields() {
  const provider = embProvider.value;
  embeddingForm.querySelectorAll("[data-provider]").forEach((el) => {
    const allowed = el.dataset.provider.split(/\s+/);
    el.hidden = !allowed.includes(provider);
  });
  const modelInput = embModel;
  const defaults = { ...DEFAULT_MODELS, ...(window.__defaultModels || {}) };
  modelInput.placeholder = defaults[provider] || "";
  modelInput.title =
    provider === "sentence_transformers"
      ? "Локальная модель HuggingFace, например intfloat/multilingual-e5-base"
      : "Модель API провайдера, например text-embedding-3-small";
}

function fillProviderOptions(providers) {
  embProvider.innerHTML = "";
  for (const value of providers) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = PROVIDER_LABELS[value] || value;
    embProvider.appendChild(option);
  }
}

function renderEmbeddingStatus(settings) {
  const parts = [];
  if (settings.api_key_set) {
    parts.push("API key задан");
  } else if (settings.provider !== "sentence_transformers") {
    parts.push("API key не задан");
  }
  if (settings.index_embedding_model) {
    parts.push(`индекс: ${settings.index_embedding_provider || "—"} / ${settings.index_embedding_model}`);
  }
  if (settings.embedding_mismatch) {
    parts.push("настройки не совпадают с индексом — выполните Rebuild Index");
  }
  if (!settings.config_writable) {
    parts.push("config.yaml недоступен для записи");
    saveEmbeddingBtn.disabled = true;
  } else {
    saveEmbeddingBtn.disabled = false;
  }
  embeddingStatus.textContent = parts.join(" · ");
  embeddingStatus.classList.toggle("error", Boolean(settings.embedding_mismatch));
}

async function loadEmbeddingSettings() {
  const settings = await api("/settings/embedding");
  window.__defaultModels = settings.default_models || DEFAULT_MODELS;
  fillProviderOptions(settings.providers || ["sentence_transformers", "openai_compatible", "ollama"]);
  embProvider.value = settings.provider;
  embModel.value = settings.model || "";
  embDevice.value = settings.device || "cpu";
  embBaseUrl.value = settings.base_url || "";
  embApiKey.value = "";
  embApiKeyEnv.value = settings.api_key_env || "OPENAI_API_KEY";
  embQueryPrefix.value = settings.query_prefix ?? "";
  embPassagePrefix.value = settings.passage_prefix ?? "";
  updateProviderFields();
  renderEmbeddingStatus(settings);
}

async function loadStatus() {
  const data = await api("/status");
  renderStatus(data);
  await loadEmbeddingSettings();
}

function renderJobMeta(job) {
  const type = JOB_TYPE_LABELS[job.type] || job.type;
  const status = STATUS_LABELS[job.status] || job.status;
  const parts = [type, status];
  if (job.phase_label && job.status === "running") {
    parts.push(job.phase_label);
    if (job.progress?.total) {
      parts.push(`${job.progress.current}/${job.progress.total} (${job.progress.percent}%)`);
    }
  }
  jobMeta.textContent = parts.join(" · ");
  jobMeta.classList.toggle("job-meta--failed", job.status === "failed");

  if (job.status === "running" && job.progress?.total) {
    jobProgress.hidden = false;
    jobProgress.value = job.progress.percent || 0;
  } else if (job.status === "completed") {
    jobProgress.hidden = false;
    jobProgress.value = 100;
    showJobError("");
  } else if (job.status === "failed") {
    jobProgress.hidden = true;
    jobProgress.value = 0;
    showJobError(job.error || "Задача завершилась с ошибкой");
  } else {
    jobProgress.hidden = true;
    jobProgress.value = 0;
  }
}

async function pollJob() {
  if (!activeJobId) return;
  try {
    const job = await api(`/jobs/${activeJobId}?since_log=${logOffset}`);
    pollFailCount = 0;
    showJobError("");
    logOffset = job.log_offset || logOffset;
    renderJobMeta(job);
    if (job.logs?.length) {
      jobLog.textContent += job.logs.join("\n") + "\n";
      jobLog.scrollTop = jobLog.scrollHeight;
    }
    if (job.status === "completed" || job.status === "failed") {
      clearInterval(pollTimer);
      pollTimer = null;
      renderJobMeta(job);
      if (job.error) {
        jobLog.textContent += `\nERROR: ${job.error}\n`;
        jobLog.scrollTop = jobLog.scrollHeight;
      }
      await loadStatus();
    }
  } catch (err) {
    pollFailCount += 1;
    const message = parseErrorMessage(err);
    if (pollFailCount < POLL_MAX_FAILURES) {
      jobMeta.textContent = `Задача · выполняется · ожидание ответа сервера (${pollFailCount}/${POLL_MAX_FAILURES})`;
      showJobError(message);
      return;
    }
    clearInterval(pollTimer);
    pollTimer = null;
    showJobError(message);
    jobMeta.textContent = "Задача · нет связи с сервером";
    jobMeta.classList.add("job-meta--failed");
    adminAlerts.hidden = false;
    adminAlerts.innerHTML = `<div class="alert alert--warning"><strong>Связь прервана</strong>${message}</div>`;
  }
}

async function startJob(path, body = {}) {
  jobPanel.hidden = false;
  jobLog.textContent = "";
  logOffset = 0;
  jobProgress.hidden = true;
  jobProgress.value = 0;
  pollFailCount = 0;
  showJobError("");
  jobMeta.classList.remove("job-meta--failed");
  try {
    const job = await api(path, {
      method: "POST",
      body: JSON.stringify(body),
    });
    activeJobId = job.id;
    renderJobMeta(job);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollJob, 1500);
    await pollJob();
  } catch (err) {
    const message = parseErrorMessage(err);
    showJobError(message);
    jobMeta.textContent = "Задача · не запущена";
    jobMeta.classList.add("job-meta--failed");
    adminAlerts.hidden = false;
    adminAlerts.innerHTML = `<div class="alert alert--error"><strong>Ошибка</strong>${message}</div>`;
  }
}

embProvider.addEventListener("change", () => {
  embModel.value = suggestedModel(embProvider.value, embModel.value);
  updateProviderFields();
});

embeddingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  embeddingStatus.textContent = "Сохранение…";
  embeddingStatus.classList.remove("error");
  const payload = {
    provider: embProvider.value,
    model: embModel.value.trim(),
    device: embDevice.value,
    base_url: embBaseUrl.value.trim(),
    api_key_env: embApiKeyEnv.value.trim(),
    query_prefix: embQueryPrefix.value,
    passage_prefix: embPassagePrefix.value,
  };
  const apiKey = embApiKey.value.trim();
  if (apiKey) {
    payload.api_key = apiKey;
  }
  try {
    const result = await api("/settings/embedding", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    embApiKey.value = "";
    renderEmbeddingStatus(result);
    let statusText = "Сохранено";
    if (result.model_adjustment) {
      embModel.value = result.model || embModel.value;
      statusText = result.model_adjustment;
    }
    if (result.rebuild_required) {
      statusText += " · нужен Rebuild Index";
    }
    embeddingStatus.textContent = statusText;
    await api("/status").then(renderStatus);
  } catch (err) {
    const message = parseErrorMessage(err);
    embeddingStatus.textContent = `Ошибка: ${message}`;
    embeddingStatus.classList.add("error");
  }
});

runIngestBtn.addEventListener("click", () => startJob("/jobs/ingest"));
runIndexBtn.addEventListener("click", () => startJob("/jobs/index", { rebuild: true }));
runIngestBspBtn.addEventListener("click", () => startJob("/jobs/ingest-bsp"));

testEmbeddingBtn.addEventListener("click", async () => {
  embeddingTestResult.hidden = false;
  embeddingTestResult.textContent = "Тест…";
  embeddingTestResult.classList.remove("error");
  try {
    const result = await api("/settings/embedding/test", {
      method: "POST",
      body: JSON.stringify({ text: "тестовый запрос" }),
    });
    embeddingTestResult.textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    const message = parseErrorMessage(err);
    embeddingTestResult.textContent = message;
    embeddingTestResult.classList.add("error");
  }
});

loadStatus().catch((err) => {
  const message = parseErrorMessage(err);
  statusContent.innerHTML = `<p class="error"><strong>Не удалось загрузить статус:</strong> ${message}</p>`;
  adminAlerts.hidden = false;
  adminAlerts.innerHTML = `<div class="alert alert--error"><strong>API недоступен</strong>${message}</div>`;
});

api("/status")
  .then((data) => {
    const bspPath = data.config?.bsp?.path;
    if (bspPath) {
      runIngestBspBtn.hidden = false;
    }
  })
  .catch(() => {});
