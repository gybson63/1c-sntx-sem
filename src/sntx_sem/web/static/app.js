const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const domainSelect = document.getElementById("domain");
const searchSubmitBtn = document.getElementById("search-submit");
const statusLine = document.getElementById("status-line");
const searchState = document.getElementById("search-state");
const searchStateTitle = document.getElementById("search-state-title");
const searchStateDetail = document.getElementById("search-state-detail");
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
let lastQuery = "";
let searchInFlight = false;
const topicCache = new Map();
const queryTokenRegex = /[a-zа-яё0-9]+/giu;

function parseErrorMessage(err) {
  const raw = String(err?.message || err || "Неизвестная ошибка");
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

function showSearch() {
  topicPanel.hidden = true;
  if (!searchInFlight) {
    resultsPanel.hidden = lastResults.length === 0 && searchState.hidden;
  }
}

function showTopic() {
  resultsPanel.hidden = true;
  searchState.hidden = true;
  topicPanel.hidden = false;
}

function setSearchLoading(query) {
  searchInFlight = true;
  searchState.hidden = false;
  searchState.className = "search-state search-state--loading card";
  searchStateTitle.textContent = `Поиск «${query}»…`;
  searchStateDetail.textContent =
    "Запрос отправлен. Первый поиск после запуска сервера может занять до минуты (загрузка модели E5).";
  resultsPanel.hidden = false;
  resultsList.innerHTML = "";
  statusLine.textContent = "Выполняется поиск…";
  statusLine.className = "meta";
  searchSubmitBtn.disabled = true;
  searchSubmitBtn.textContent = "Поиск…";
}

function setSearchError(message) {
  searchInFlight = false;
  searchState.hidden = false;
  searchState.className = "search-state search-state--error card";
  searchStateTitle.textContent = "Поиск не выполнен";
  searchStateDetail.textContent = message;
  resultsPanel.hidden = false;
  resultsList.innerHTML = "";
  statusLine.textContent = message;
  statusLine.className = "meta error";
  searchSubmitBtn.disabled = false;
  searchSubmitBtn.textContent = "Найти";
}

function clearSearchState() {
  searchInFlight = false;
  searchState.hidden = true;
  searchSubmitBtn.disabled = false;
  searchSubmitBtn.textContent = "Найти";
}

function tokenizeTerms(text) {
  if (!text) return [];
  const tokens = text.toLowerCase().match(queryTokenRegex) || [];
  return [...new Set(tokens)];
}

function getHighlightTerms(item) {
  const terms = Array.isArray(item?.highlight_terms) ? item.highlight_terms : [];
  if (terms.length) {
    return [...new Set(terms.map((term) => String(term).toLowerCase()).filter(Boolean))];
  }
  return tokenizeTerms(lastQuery);
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightIntoElement(element, text, terms) {
  element.textContent = "";
  if (!text) {
    return;
  }
  const uniqueTerms = [...new Set((terms || []).filter(Boolean))];
  if (!uniqueTerms.length) {
    element.textContent = text;
    return;
  }
  const pattern = uniqueTerms
    .slice()
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp)
    .join("|");
  if (!pattern) {
    element.textContent = text;
    return;
  }
  const regex = new RegExp(pattern, "giu");
  let cursor = 0;
  let match = regex.exec(text);
  while (match) {
    if (match.index > cursor) {
      element.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    }
    const marked = document.createElement("mark");
    marked.className = "search-highlight";
    marked.textContent = match[0];
    element.appendChild(marked);
    cursor = match.index + match[0].length;
    match = regex.exec(text);
  }
  if (cursor < text.length) {
    element.appendChild(document.createTextNode(text.slice(cursor)));
  }
}

async function fetchTopic(topicId) {
  if (topicCache.has(topicId)) {
    return topicCache.get(topicId);
  }
  const topic = await api(`/topic/${encodeURIComponent(topicId)}`);
  topicCache.set(topicId, topic);
  return topic;
}

function renderResults(items) {
  lastResults = items;
  resultsList.innerHTML = "";
  clearSearchState();
  if (!items.length) {
    searchState.hidden = false;
    searchState.className = "search-state search-state--empty card";
    searchStateTitle.textContent = "Ничего не найдено";
    searchStateDetail.textContent =
      "Попробуйте другой запрос, домен Platform API или более общую формулировку.";
    resultsPanel.hidden = false;
    statusLine.textContent = "Ничего не найдено.";
    statusLine.className = "meta";
    return;
  }
  searchState.hidden = true;
  statusLine.textContent = `Найдено: ${items.length}`;
  statusLine.className = "meta";
  resultsPanel.hidden = false;
  for (const item of items) {
    const highlightTerms = getHighlightTerms(item);
    const li = document.createElement("li");
    li.className = "result-item";
    const titleRow = document.createElement("div");
    const topicLink = document.createElement("a");
    topicLink.href = "#";
    topicLink.dataset.topicId = encodeURIComponent(item.id);
    topicLink.textContent = item.title || item.id;
    titleRow.appendChild(topicLink);

    const scoreRow = document.createElement("div");
    scoreRow.className = "result-score";
    scoreRow.textContent = `[${item.domain || ""}] score=${item.score}`;

    const excerptRow = document.createElement("div");
    excerptRow.className = "result-excerpt";
    highlightIntoElement(excerptRow, item.excerpt || "", highlightTerms);

    const actionsRow = document.createElement("div");
    actionsRow.className = "result-actions";
    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.className = "link-btn inline-action-btn";
    toggleBtn.textContent = "Показать чанк";
    actionsRow.appendChild(toggleBtn);

    const fullChunk = document.createElement("div");
    fullChunk.className = "result-full-text";
    fullChunk.hidden = true;
    fullChunk.textContent = "";

    topicLink.addEventListener("click", (e) => {
      e.preventDefault();
      openTopic(item.id, highlightTerms);
    });

    toggleBtn.addEventListener("click", async () => {
      const isOpening = fullChunk.hidden;
      if (!isOpening) {
        fullChunk.hidden = true;
        toggleBtn.textContent = "Показать чанк";
        return;
      }
      toggleBtn.disabled = true;
      toggleBtn.textContent = "Загрузка…";
      try {
        const topic = await fetchTopic(item.id);
        const fullText = topic.text || topic.content || "";
        highlightIntoElement(fullChunk, fullText, highlightTerms);
        fullChunk.hidden = false;
        toggleBtn.textContent = "Скрыть чанк";
      } catch (err) {
        setSearchError(parseErrorMessage(err));
        toggleBtn.textContent = "Показать чанк";
      } finally {
        toggleBtn.disabled = false;
      }
    });

    li.appendChild(titleRow);
    li.appendChild(scoreRow);
    li.appendChild(excerptRow);
    li.appendChild(actionsRow);
    li.appendChild(fullChunk);
    resultsList.appendChild(li);
  }
}

async function openTopic(topicId, highlightTerms = tokenizeTerms(lastQuery)) {
  statusLine.textContent = "Загрузка топика…";
  try {
    const topic = await fetchTopic(topicId);
    topicTitle.textContent = topic.title || topic.id;
    topicMeta.textContent = `[${topic.domain || ""}] ${topic.id || topicId}`;
    const fullText = topic.text || topic.content || JSON.stringify(topic, null, 2);
    highlightIntoElement(topicBody, fullText, highlightTerms);
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
    setSearchError(parseErrorMessage(err));
  }
}

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query || searchInFlight) return;
  lastQuery = query;
  topicCache.clear();
  showSearch();
  setSearchLoading(query);
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
    setSearchError(parseErrorMessage(err));
  }
});

backBtn.addEventListener("click", () => {
  showSearch();
  if (lastResults.length) {
    resultsPanel.hidden = false;
  }
});

function renderInitialStatus(health, status) {
  const idx = status.index || {};
  const issues = status.issues?.length ? status.issues : health.issues || [];
  const parts = [];
  if (health.ready) {
    parts.push("индекс готов");
  } else {
    parts.push("индекс не готов — соберите базу в /admin");
  }
  if (idx.embedding_model) {
    parts.push(`модель индекса: ${idx.embedding_model}`);
  }
  if (idx.indexed_chunks) {
    parts.push(`${idx.indexed_chunks} чанков`);
  }
  statusLine.textContent = parts.join(" · ");
  statusLine.className = "meta";

  const errorIssues = issues.filter((item) => item.severity !== "warning");
  const warningIssues = issues.filter((item) => item.severity === "warning");

  if (!health.ready || errorIssues.length) {
    statusLine.className = "meta error";
    searchState.hidden = false;
    searchState.className = "search-state search-state--error card";
    searchStateTitle.textContent = "Поиск недоступен";
    const detailParts = errorIssues.map((item) => item.message);
    if (!detailParts.length) {
      detailParts.push(
        "Индекс не готов. Откройте /admin → Ingest HBK + Index (или Rebuild Index, если export уже есть)."
      );
    }
    searchStateDetail.textContent = detailParts.join(" ");
    resultsPanel.hidden = false;
    return;
  }

  if (warningIssues.length || health.embedding_in_sync === false) {
    const warnText = warningIssues.map((item) => item.message).join(" ");
    statusLine.textContent += warnText
      ? ` · ${warnText}`
      : " · config.yaml не совпадает с индексом (поиск использует модель индекса)";
    statusLine.className = "meta error";
    searchState.hidden = false;
    searchState.className = "search-state search-state--error card";
    searchStateTitle.textContent = "Поиск доступен с ограничениями";
    searchStateDetail.textContent = warnText || statusLine.textContent;
    resultsPanel.hidden = false;
  }
}

api("/health")
  .then((health) => api("/status").then((status) => ({ health, status })))
  .then(({ health, status }) => renderInitialStatus(health, status))
  .catch((err) => {
    statusLine.textContent = `API недоступен: ${parseErrorMessage(err)}`;
    statusLine.className = "meta error";
    searchState.hidden = false;
    searchState.className = "search-state search-state--error card";
    searchStateTitle.textContent = "Сервер недоступен";
    searchStateDetail.textContent = parseErrorMessage(err);
    resultsPanel.hidden = false;
  });
