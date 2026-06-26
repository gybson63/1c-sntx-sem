const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const domainSelect = document.getElementById("domain");
const statusLine = document.getElementById("status-line");
const resultsPanel = document.getElementById("results-panel");
const resultsList = document.getElementById("results");
const topicPanel = document.getElementById("topic-panel");
const topicTitle = document.getElementById("topic-title");
const topicMeta = document.getElementById("topic-meta");
const topicBody = document.getElementById("topic-body");
const topicExamples = document.getElementById("topic-examples");
const examplesList = document.getElementById("examples-list");
const backBtn = document.getElementById("back-to-results");

let lastResults = [];

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

function showSearch() {
  topicPanel.hidden = true;
  resultsPanel.hidden = lastResults.length === 0;
}

function showTopic() {
  resultsPanel.hidden = true;
  topicPanel.hidden = false;
}

function renderResults(items) {
  lastResults = items;
  resultsList.innerHTML = "";
  if (!items.length) {
    statusLine.textContent = "Ничего не найдено.";
    resultsPanel.hidden = true;
    return;
  }
  statusLine.textContent = `Найдено: ${items.length}`;
  resultsPanel.hidden = false;
  for (const item of items) {
    const li = document.createElement("li");
    li.className = "result-item";
    li.innerHTML = `
      <div><a href="#" data-topic-id="${encodeURIComponent(item.id)}">${escapeHtml(item.title)}</a></div>
      <div class="result-score">[${escapeHtml(item.domain)}] score=${item.score}</div>
      <div>${escapeHtml(item.excerpt || "")}</div>
    `;
    li.querySelector("a").addEventListener("click", (e) => {
      e.preventDefault();
      openTopic(item.id);
    });
    resultsList.appendChild(li);
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function openTopic(topicId) {
  statusLine.textContent = "Загрузка топика…";
  try {
    const topic = await api(`/topic/${encodeURIComponent(topicId)}`);
    topicTitle.textContent = topic.title || topic.id;
    topicMeta.textContent = `[${topic.domain || ""}] ${topic.id || topicId}`;
    topicBody.textContent = topic.text || topic.content || JSON.stringify(topic, null, 2);
    examplesList.innerHTML = "";
    const examples = topic.examples || [];
    if (examples.length) {
      topicExamples.hidden = false;
      for (const ex of examples) {
        const li = document.createElement("li");
        li.textContent = ex.title || ex.id || ex.code?.slice(0, 80) || "example";
        examplesList.appendChild(li);
      }
    } else {
      topicExamples.hidden = true;
    }
    showTopic();
    statusLine.textContent = "";
  } catch (err) {
    statusLine.textContent = `Ошибка: ${err.message}`;
    statusLine.className = "meta error";
  }
}

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  statusLine.textContent = "Поиск…";
  statusLine.className = "meta";
  showSearch();
  try {
    const items = await api("/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        domain: domainSelect.value,
        limit: 10,
      }),
    });
    renderResults(items);
  } catch (err) {
    statusLine.textContent = `Ошибка: ${err.message}`;
    statusLine.className = "meta error";
  }
});

backBtn.addEventListener("click", () => {
  showSearch();
  if (lastResults.length) {
    resultsPanel.hidden = false;
  }
});

api("/health")
  .then((data) => {
    const ready = data.ready ? "индекс готов" : "индекс не готов";
    statusLine.textContent = `${ready} · ${data.embedding?.model || ""}`;
  })
  .catch(() => {
    statusLine.textContent = "API недоступен";
  });
