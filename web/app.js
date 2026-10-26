const $ = (selector) => document.querySelector(selector);
const MODES = {
  attribute: "По атрибутам",
  baseline: "Базовая модель",
  bm25: "BM25",
};
const STATUS = {
  match: { label: "Совпало", icon: "check" },
  conflict: { label: "Расхождение", icon: "conflict" },
  unknown: { label: "Неопределено", icon: "unknown" },
};
const ATTRIBUTES = {
  color: "Цвет",
  colour: "Цвет",
  brand: "Бренд",
  size: "Размер",
  material: "Материал",
  gender: "Для кого",
  age: "Возраст",
  category: "Категория",
  type: "Тип",
  capacity: "Объём",
  storage: "Объём памяти",
  power: "Мощность",
  length: "Длина / размер",
  count: "Количество",
  pack_size: "В упаковке",
  waterproof: "Водостойкость",
  wireless: "Беспроводное подключение",
  compatibility: "Совместимость",
  model: "Модель",
};
const METRIC_LABELS = {
  ndcg_at_10: "nDCG@10",
  ndcg_10: "nDCG@10",
  "ndcg@10": "nDCG@10",
  ndcg: "nDCG",
  map: "MAP",
  mrr: "MRR",
  recall_at_10: "Recall@10",
  "recall@10": "Recall@10",
  precision_at_10: "Precision@10",
  "precision@10": "Precision@10",
  attribute: "Модель с атрибутами",
  baseline: "Базовая модель",
  bm25: "BM25",
  tfidf: "TF–IDF",
  rule: "Базовая модель + правила",
  no_product_evidence: "Атрибуты запроса (абляция)",
  test_queries: "Запросов в оценке",
  queries: "Запросов",
  catalog_size: "Товаров в каталоге",
  latency_ms: "Задержка, мс",
};
const REASONS = {
  "no comparable explicit specification":
    "В товаре нет сопоставимого явно указанного значения",
  "equal canonical values": "Значения совпадают после приведения единиц",
  "same identifier": "Совпадает обозначение модели или бренда",
  "different identifiers in the same family or role":
    "Указано другое обозначение в той же группе",
  "different explicit values in the same semantic role":
    "Для одного свойства явно указаны разные значения",
  "nearby capacity may reflect binary/decimal conventions or rounding":
    "Близкие значения памяти могут различаться из-за единиц или округления",
  "nearby dimension may reflect nominal rounding":
    "Близкие размеры могут быть округлены",
  "nominal wattages alone do not establish compatibility":
    "Одной мощности недостаточно, чтобы подтвердить совместимость",
  "query lies in an explicitly supported range":
    "Значение из запроса входит в указанный диапазон",
  "query lies outside the explicitly supported range":
    "Значение из запроса вне указанного диапазона",
  "supported range only partly covers the query":
    "Указанный диапазон покрывает запрос только частично",
  "disjoint requirement and specification intervals":
    "Диапазоны требования и товара не пересекаются",
  "strict requirement boundary is not met": "Граница требования не соблюдена",
  "product explicitly excludes the boundary value":
    "Пограничное значение явно исключено в описании",
  "entire product interval satisfies the requirement":
    "Весь указанный диапазон соответствует требованию",
  "overlapping intervals do not establish satisfaction":
    "Пересечения диапазонов недостаточно для подтверждения",
  "product text contains contradictory or multiple variant specifications":
    "В тексте есть противоречивые значения или несколько вариантов товара",
  "description-only disagreement is insufficient for a hard conflict":
    "Данных из описания недостаточно, чтобы уверенно установить расхождение",
  "ambiguous specification": "Свойство указано неоднозначно",
  "bundle, dimensions or alternatives have unresolved scope":
    "Неясно, к чему относятся значения: комплекту, размерам или вариантам",
  "alternative requirements are not conjunctive constraints":
    "В запросе перечислены альтернативы",
  "elided model list is not fully parsed":
    "Сокращённый список моделей не удалось разобрать полностью",
  "negated range is not a positive requirement interval":
    "Исключённый диапазон не задаёт допустимые значения",
  "a conjunction of values is not necessarily a range":
    "Перечисленные значения не обязательно образуют диапазон",
  "range combines different dimensions": "В диапазоне смешаны разные величины",
  "descending range": "Границы диапазона указаны в обратном порядке",
  "explicit excluded value differs":
    "Товар отличается от явно исключённого значения",
  "both explicitly exclude the same value":
    "Одинаковое значение явно исключено",
  "value is explicitly excluded": "Значение явно исключено в запросе",
  "negation does not identify a value":
    "Исключение не задаёт конкретное нужное значение",
  "same explicit exclusion": "Совпадает явно указанное исключение",
  "identifier is explicitly excluded": "Это обозначение явно исключено",
  "different non-excluded identifier": "Обозначение отличается от исключённого",
  "negation does not identify the product":
    "Исключение не определяет конкретный товар",
};
const state = {
  meta: null,
  query: "",
  mode: "attribute",
  compare: false,
  limit: 12,
  responses: {},
  loading: false,
  sequence: 0,
  controller: null,
  detailController: null,
  selected: null,
};
const welcomeContent = $("#results").firstElementChild.cloneNode(true);
const integer = new Intl.NumberFormat("ru-RU");
const decimal = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 });

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function icon(name, className = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#icon-${name}`);
  svg.append(use);
  return svg;
}

function asText(value, fallback = "") {
  if (value === null || value === undefined || value === "") return fallback;
  if (Array.isArray(value))
    return value
      .map((item) => asText(item))
      .filter(Boolean)
      .join(", ");
  if (typeof value === "object")
    return asText(value.value ?? value.name ?? value.label, fallback);
  return String(value)
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;|&#160;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function metricLabel(value) {
  return asText(value, "Метрика")
    .split(/\s*·\s*/)
    .map((part) => METRIC_LABELS[part] || part.replaceAll("_", " "))
    .join(" · ");
}

function safeUrl(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value, location.origin);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function setLink(element, value, label) {
  const url = safeUrl(value);
  if (!url) return false;
  element.href = url;
  if (label) element.textContent = label;
  if (new URL(url).origin !== location.origin) {
    element.target = "_blank";
    element.rel = "noopener noreferrer";
  }
  element.hidden = false;
  return true;
}

async function request(path, signal) {
  const response = await fetch(path, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let message = "";
    try {
      const error = await response.json();
      if (typeof error.detail === "string") message = error.detail;
      else if (typeof error.message === "string") message = error.message;
    } catch {
      /* The HTTP status remains sufficient to explain a failed request. */
    }
    const failure = new Error(message || `HTTP ${response.status}`);
    failure.status = response.status;
    throw failure;
  }
  return response.json();
}

function announce(text) {
  $("#announcer").textContent = text;
}

function flattenMetrics(metrics, prefix = "", depth = 0) {
  if (!metrics || depth > 3) return [];
  if (Array.isArray(metrics))
    return metrics.flatMap((metric) => {
      if (metric && typeof metric === "object" && metric.value !== undefined)
        return [
          {
            label: asText(
              metric.label ?? metric.name ?? metric.metric,
              "Метрика",
            ),
            value: metric.value,
            detail: asText(metric.detail ?? metric.description),
            unit: asText(metric.unit),
          },
        ];
      return flattenMetrics(metric, prefix, depth + 1);
    });
  if (typeof metrics !== "object") return [];
  return Object.entries(metrics).flatMap(([key, value]) => {
    const label = [prefix, METRIC_LABELS[key] || key.replaceAll("_", " ")]
      .filter(Boolean)
      .join(" · ");
    if (typeof value === "number" && Number.isFinite(value))
      return [{ label, value }];
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      value.value !== undefined
    )
      return [
        {
          label: asText(value.label, label),
          value: value.value,
          detail: asText(value.detail ?? value.description),
          unit: asText(value.unit),
        },
      ];
    if (value && typeof value === "object")
      return flattenMetrics(value, label, depth + 1);
    return [];
  });
}

function renderMeta(meta) {
  state.meta = meta;
  const size = Number(meta.catalog_size);
  $("#catalog-count").textContent = Number.isFinite(size)
    ? `${integer.format(size)} товаров`
    : "Каталог подключён";
  $("#catalog-status .status-dot").className = "status-dot";
  $("#connection-notice").hidden = true;

  const examples = Array.isArray(meta.examples) ? meta.examples : [];
  const container = $("#examples");
  container.replaceChildren();
  const seen = new Set();
  examples.slice(0, 6).forEach((example) => {
    const query = asText(
      typeof example === "string"
        ? example
        : (example?.query ?? example?.q ?? example?.text),
    );
    if (!query || seen.has(query)) return;
    seen.add(query);
    const button = node("button", "example-button", query);
    button.type = "button";
    button.addEventListener("click", () => {
      $("#search-input").value = query;
      performSearch({ query });
    });
    container.append(button);
  });
  $("#examples-row").hidden = !container.childElementCount;
  if (seen.size)
    $("#search-input").placeholder = `Например, ${seen.values().next().value}`;

  if (meta.model_version !== undefined) {
    $("#model-version").textContent = asText(meta.model_version);
    $("#version-fact").hidden = false;
  }
  const source = meta.dataset || meta.source;
  const sourceUrl =
    meta.source_url ||
    meta.dataset_url ||
    (source && typeof source === "object" ? source.url : null);
  const sourceName = asText(
    meta.dataset_name ||
      meta.source_name ||
      (source && typeof source === "object" ? source.name : source),
  );
  if (
    sourceUrl &&
    setLink(
      $("#source-link"),
      sourceUrl,
      sourceName || new URL(safeUrl(sourceUrl) || location.origin).hostname,
    )
  )
    $("#source-fact").hidden = false;
  else if (sourceName) {
    $("#source-link").textContent = sourceName;
    $("#source-fact").hidden = false;
  }
  if (meta.description)
    $("#meta-description").textContent = asText(meta.description);
  if (meta.report_url) setLink($("#report-link"), meta.report_url);

  const metrics = flattenMetrics(meta.metrics);
  const grid = $("#metrics-grid");
  grid.replaceChildren();
  metrics.forEach((metric) => {
    const card = node("div", "metric-card");
    card.append(node("div", "metric-label", metricLabel(metric.label)));
    const value =
      typeof metric.value === "number"
        ? decimal.format(metric.value)
        : asText(metric.value);
    card.append(
      node(
        "div",
        "metric-value",
        `${value}${metric.unit ? ` ${metric.unit}` : ""}`,
      ),
    );
    if (metric.detail) card.append(node("p", "metric-detail", metric.detail));
    grid.append(card);
  });
  $("#metrics-block").hidden = !metrics.length;
  $("#metrics-note").textContent = asText(
    meta.evaluation_note || meta.metrics_note || meta.evaluation?.description,
  );
}

async function loadMeta() {
  try {
    renderMeta(await request("/api/meta"));
  } catch {
    $("#catalog-count").textContent = "Сведения о каталоге недоступны";
    $("#catalog-status .status-dot").className = "status-dot is-error";
    $("#connection-notice").hidden = false;
  }
}

function evidenceOf(product) {
  return Array.isArray(product.evidence)
    ? product.evidence
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          ...item,
          attribute: item.attribute ?? item.kind,
          query_value:
            item.query_value ?? item.query?.text ?? item.query?.value,
          product_value:
            item.product_value ?? item.product?.text ?? item.product?.value,
        }))
    : [];
}
function attributeName(value) {
  const raw = asText(value, "Свойство");
  return ATTRIBUTES[raw.toLowerCase()] || raw.replaceAll("_", " ");
}
function evidenceStatus(item) {
  return Object.hasOwn(STATUS, item.status) ? item.status : "unknown";
}

function evidenceChip(item) {
  const status = evidenceStatus(item);
  const text = `${attributeName(item.attribute)}: ${asText(item.product_value, status === "unknown" ? "не указано" : asText(item.query_value, "—"))}`;
  const chip = node("span", `evidence-chip ${status}`);
  chip.title = `${STATUS[status].label}. Из запроса: ${asText(item.query_value, "не указано")}; в товаре: ${asText(item.product_value, "не указано")}`;
  chip.append(icon(STATUS[status].icon), node("span", "chip-label", text));
  const accessible = node("span", "sr-only", ` — ${STATUS[status].label}`);
  chip.append(accessible);
  return chip;
}

function productCard(product, index, mode) {
  const card = node("article", "product-card");
  card.append(
    node(
      "span",
      "product-rank",
      String(product.rank ?? index + 1).padStart(2, "0"),
    ),
  );
  const content = node("div", "product-main");
  const meta = node("div", "product-meta");
  const brand = asText(product.brand);
  const category = asText(product.category);
  if (brand) meta.append(node("span", null, brand));
  if (brand && category) meta.append(node("span", "meta-separator"));
  if (category) meta.append(node("span", null, category));
  if (meta.childElementCount) content.append(meta);
  const title = asText(product.title, "Товар без названия");
  const titleButton = node("button", "product-title", title);
  titleButton.type = "button";
  titleButton.addEventListener("click", () => openProduct(product, mode));
  const heading = node("h3", "product-heading");
  heading.append(titleButton);
  content.append(heading);
  const description = asText(product.description);
  if (description)
    content.append(node("p", "product-description", description));
  const evidence = evidenceOf(product);
  if (evidence.length) {
    const chips = node("div", "product-evidence");
    evidence.slice(0, 4).forEach((item) => chips.append(evidenceChip(item)));
    if (evidence.length > 4)
      chips.append(node("span", "evidence-more", `+${evidence.length - 4}`));
    content.append(chips);
  } else
    content.append(
      node("p", "product-no-evidence", "Откройте описание и разбор результата"),
    );
  card.append(content);
  const open = node("button", "product-open");
  open.type = "button";
  open.setAttribute("aria-label", `Разобрать товар: ${title}`);
  open.append(icon("chevron"));
  open.addEventListener("click", () => openProduct(product, mode));
  card.append(open);
  return card;
}

function renderSkeleton() {
  const loading = node("div", "loading-state");
  loading.setAttribute("aria-label", "Ищем товары");
  for (let index = 0; index < 3; index++) {
    const card = node("div", "skeleton-card");
    card.setAttribute("aria-hidden", "true");
    card.append(node("div", "skeleton-text skeleton-rank"));
    const copy = node("div");
    ["brand", "title", "description", "evidence"].forEach((kind) =>
      copy.append(node("div", `skeleton-text skeleton-${kind}`)),
    );
    card.append(copy);
    loading.append(card);
  }
  return loading;
}

function stateCard(kind, title, message, retry) {
  const card = node("div", `state-card ${kind}`);
  const illustration = node("div", "state-icon");
  illustration.append(icon(kind === "error" ? "info" : "search"));
  card.append(illustration);
  const copy = node("div");
  copy.append(node("h3", null, title), node("p", null, message));
  if (retry) {
    const button = node("button", "outline-button", "Попробовать ещё раз");
    button.type = "button";
    button.append(icon("arrow", "icon icon-small"));
    button.addEventListener("click", retry);
    copy.append(button);
  }
  card.append(copy);
  return card;
}

function responseList(response, mode) {
  const list = node("div", "product-list");
  if (!response.results.length) {
    list.append(
      stateCard(
        "empty",
        "Пока нет совпадений",
        "Попробуйте более короткий запрос или другой режим. Каталог содержит товары на английском.",
      ),
    );
    return list;
  }
  response.results.forEach((product, index) =>
    list.append(productCard(product, index, mode)),
  );
  return list;
}

function candidateCount(response) {
  const value = Number(response?.total_candidates);
  return Number.isFinite(value) && value >= 0 ? value : null;
}
function latency(response) {
  const value = Number(response?.latency_ms);
  return Number.isFinite(value) && value >= 0
    ? `${decimal.format(value)} мс`
    : null;
}

function renderResults() {
  const container = $("#results");
  container.replaceChildren();
  $("#results-eyebrow").textContent = state.compare
    ? "Два взгляда на один запрос"
    : MODES[state.mode];
  $("#results-title").textContent = `«${state.query}»`;
  const stats = $("#results-statistics");
  stats.replaceChildren();
  if (state.compare) {
    const caption = node(
      "p",
      "compare-caption",
      "Ранжирование с атрибутами и базовая модель. Откройте любой результат, чтобы проверить его свойства.",
    );
    container.append(caption);
    const grid = node("div", "comparison-grid");
    for (const mode of ["attribute", "baseline"]) {
      const response = state.responses[mode];
      const column = node("section", "comparison-column");
      const heading = node("div", "comparison-heading");
      heading.append(
        node("h3", null, MODES[mode]),
        node(
          "span",
          null,
          `${integer.format(response.results.length)} товаров${latency(response) ? ` · ${latency(response)}` : ""}`,
        ),
      );
      column.append(heading, responseList(response, mode));
      grid.append(column);
    }
    container.append(grid);
    const total = candidateCount(state.responses.attribute);
    if (total !== null)
      stats.append(node("span", null, `${integer.format(total)} кандидатов`));
  } else {
    const response = state.responses[state.mode];
    container.append(responseList(response, state.mode));
    stats.append(
      node(
        "span",
        null,
        `${integer.format(response.results.length)} товаров в выдаче`,
      ),
    );
    const total = candidateCount(response);
    if (total !== null)
      stats.append(
        node(
          "span",
          null,
          `${integer.format(total)} кандидатов${latency(response) ? ` · ${latency(response)}` : ""}`,
        ),
      );
    else if (latency(response))
      stats.append(node("span", null, latency(response)));
  }
  const responses = Object.values(state.responses);
  $("#more-container").hidden =
    state.limit >= 40 ||
    !responses.some(
      (response) =>
        response.results.length >= state.limit &&
        (candidateCount(response) === null ||
          candidateCount(response) > state.limit),
    );
  announce(
    `Поиск завершён. ${state.compare ? "Два режима для сравнения." : `${state.responses[state.mode].results.length} товаров в выдаче.`}`,
  );
}

function syncControls() {
  const radio = $(`input[name="mode"][value="${state.mode}"]`);
  if (radio) radio.checked = true;
  $("#compare-toggle").setAttribute("aria-pressed", String(state.compare));
  document.body.classList.toggle("is-comparing", state.compare);
}

function syncLocation(replace = false) {
  const url = new URL(location.href);
  url.searchParams.set("q", state.query);
  url.searchParams.set("mode", state.mode);
  if (state.compare) url.searchParams.set("compare", "1");
  else url.searchParams.delete("compare");
  if (url.href !== location.href)
    history[replace ? "replaceState" : "pushState"]({}, "", url);
}

async function performSearch({
  query = $("#search-input").value,
  preserveLimit = false,
  updateLocation = true,
} = {}) {
  query = query.trim();
  if (!query) {
    $("#search-input").setAttribute("aria-invalid", "true");
    $("#search-input").focus();
    announce("Введите запрос на английском.");
    return;
  }
  $("#search-input").removeAttribute("aria-invalid");
  $("#search-input").value = query;
  if (!preserveLimit || state.query !== query) state.limit = 12;
  state.query = query;
  state.responses = {};
  state.loading = true;
  state.controller?.abort();
  state.controller = new AbortController();
  const sequence = ++state.sequence;
  syncControls();
  if (updateLocation) syncLocation();
  $("#results-title").textContent = `«${query}»`;
  $("#results-eyebrow").textContent = "Ищем в каталоге";
  $("#results-statistics").replaceChildren();
  $("#results").setAttribute("aria-busy", "true");
  $("#results").replaceChildren(renderSkeleton());
  $("#more-container").hidden = true;
  $("#load-more").disabled = true;
  announce("Ищем товары.");
  const modes = state.compare ? ["attribute", "baseline"] : [state.mode];
  try {
    const responses = await Promise.all(
      modes.map(async (mode) => {
        const params = new URLSearchParams({
          q: query,
          mode,
          limit: String(state.limit),
        });
        const response = await request(
          `/api/search?${params}`,
          state.controller.signal,
        );
        if (!response || !Array.isArray(response.results))
          throw new Error("Некорректный ответ поиска");
        return [mode, response];
      }),
    );
    if (sequence !== state.sequence) return;
    state.responses = Object.fromEntries(responses);
    renderResults();
  } catch (error) {
    if (error.name === "AbortError" || sequence !== state.sequence) return;
    $("#results-eyebrow").textContent = "Поиск недоступен";
    const message =
      error.status === 429
        ? "Сейчас слишком много запросов. Подождите немного и повторите поиск."
        : error.status === 422 || error.status === 400
          ? "Проверьте запрос и попробуйте ещё раз."
          : "Не удалось получить товары из каталога. Проверьте соединение и повторите поиск.";
    $("#results").replaceChildren(
      stateCard("error", "Не получилось загрузить выдачу", message, () =>
        performSearch(),
      ),
    );
    announce("Ошибка поиска. Попробуйте ещё раз.");
  } finally {
    if (sequence === state.sequence) {
      state.loading = false;
      $("#results").setAttribute("aria-busy", "false");
      $("#load-more").disabled = false;
    }
  }
}

function renderDrawer(product, { loading = false, failed = false } = {}) {
  const root = $("#drawer-content");
  root.replaceChildren();
  const selected = state.selected;
  if (!selected) return;
  if (loading)
    root.append(node("div", "drawer-loading", "Загружаем полные сведения"));
  if (failed)
    root.append(
      node(
        "p",
        "drawer-notice",
        "Полные сведения о товаре недоступны. Ниже показаны данные из поисковой выдачи.",
      ),
    );
  const meta = [asText(product.brand), asText(product.category)]
    .filter(Boolean)
    .join(" / ");
  if (meta) root.append(node("div", "drawer-meta", meta));
  const title = node(
    "h2",
    "drawer-title",
    asText(product.title, "Товар без названия"),
  );
  title.id = "drawer-title";
  root.append(title);
  const score = node("div", "drawer-score");
  const mode = node("span");
  mode.append(node("strong", null, MODES[selected.mode] || selected.mode));
  score.append(mode);
  if (selected.product.rank !== undefined)
    score.append(node("span", null, `Место ${selected.product.rank}`));
  if (
    typeof selected.product.score === "number" &&
    Number.isFinite(selected.product.score)
  )
    score.append(
      node(
        "span",
        null,
        `Оценка ранжирования ${decimal.format(selected.product.score)}`,
      ),
    );
  root.append(score);
  root.append(node("h3", "drawer-section-title", "Запрос и данные товара"));
  const evidence = evidenceOf(selected.product);
  if (evidence.length) {
    const table = node("div", "evidence-table");
    const head = node("div", "evidence-table-head");
    head.append(
      node("span", null, "Из запроса"),
      node("span", null, "В каталоге"),
      node("span", "sr-only", "Результат"),
    );
    table.append(head);
    evidence.forEach((item) => {
      const status = evidenceStatus(item);
      const row = node("div", `evidence-row ${status}`);
      row.append(
        node("div", "evidence-attribute", attributeName(item.attribute)),
        node("div", "evidence-query", asText(item.query_value, "Не выделено")),
        node(
          "div",
          "evidence-product",
          asText(item.product_value, "Не указано"),
        ),
      );
      const mark = node("span", `evidence-status-icon ${status}`);
      mark.append(icon(STATUS[status].icon));
      mark.setAttribute("aria-label", STATUS[status].label);
      row.append(
        mark,
        node(
          "div",
          "evidence-row-status",
          `${STATUS[status].label}${item.reason ? ` · ${REASONS[item.reason] || asText(item.reason)}` : ""}`,
        ),
      );
      table.append(row);
    });
    root.append(table);
  } else
    root.append(
      node(
        "p",
        "drawer-empty",
        "Для этого результата нет выделенных атрибутов. Сопоставьте запрос с названием и описанием товара.",
      ),
    );
  const description = asText(product.description);
  if (description)
    root.append(
      node("h3", "drawer-section-title", "Описание из каталога"),
      node("p", "drawer-description", description),
    );
  const bulletPoints = asText(
    product.bullet_points ??
      (typeof product.attributes === "string" ? product.attributes : ""),
  );
  if (bulletPoints && bulletPoints !== description)
    root.append(
      node("h3", "drawer-section-title", "Характеристики из каталога"),
      node("p", "drawer-description", bulletPoints),
    );
  const attributes = product.attributes;
  if (
    attributes &&
    typeof attributes === "object" &&
    !Array.isArray(attributes)
  ) {
    const facts = node("dl", "drawer-facts");
    Object.entries(attributes).forEach(([key, value]) => {
      const text = asText(value);
      if (!text) return;
      const row = node("div");
      row.append(node("dt", null, attributeName(key)), node("dd", null, text));
      facts.append(row);
    });
    if (facts.childElementCount)
      root.append(
        node("h3", "drawer-section-title", "Дополнительные свойства"),
        facts,
      );
  }
  const facts = node("dl", "drawer-facts");
  const id = node("div");
  id.append(
    node("dt", null, "ID товара"),
    node("dd", null, asText(product.id, asText(selected.product.id))),
  );
  facts.append(id);
  root.append(node("h3", "drawer-section-title", "Источник результата"), facts);
  const external = node(
    "a",
    "inline-link drawer-external",
    "Открыть источник товара",
  );
  if (
    setLink(external, product.url || product.product_url || product.source_url)
  ) {
    external.append(icon("external", "icon icon-small"));
    root.append(external);
  }
}

async function openProduct(product, mode) {
  state.detailController?.abort();
  state.detailController = new AbortController();
  const controller = state.detailController;
  state.selected = { product, mode };
  renderDrawer(product, { loading: true });
  const dialog = $("#product-dialog");
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
  try {
    if (product.id === undefined || product.id === null)
      throw new Error("Отсутствует ID товара");
    const data = await request(
      `/api/product/${encodeURIComponent(product.id)}`,
      controller.signal,
    );
    if (controller.signal.aborted) return;
    const full =
      data.product && typeof data.product === "object" ? data.product : data;
    renderDrawer({ ...product, ...full });
  } catch (error) {
    if (error.name !== "AbortError") renderDrawer(product, { failed: true });
  }
}

$("#search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  performSearch();
});
$("#search-input").addEventListener("input", () =>
  $("#search-input").removeAttribute("aria-invalid"),
);
document.querySelectorAll('input[name="mode"]').forEach((radio) =>
  radio.addEventListener("change", () => {
    state.mode = radio.value;
    if (state.compare) state.compare = false;
    syncControls();
    if (state.query) performSearch();
  }),
);
$("#compare-toggle").addEventListener("click", () => {
  state.compare = !state.compare;
  syncControls();
  if (state.query) performSearch();
  else
    announce(
      state.compare
        ? "Сравнение включено. Введите запрос."
        : "Сравнение выключено.",
    );
});
$("#load-more").addEventListener("click", () => {
  state.limit = Math.min(state.limit + 12, 40);
  performSearch({ preserveLimit: true });
});
$("#meta-retry").addEventListener("click", loadMeta);
$("#drawer-close").addEventListener("click", () =>
  $("#product-dialog").close(),
);
$("#product-dialog").addEventListener("close", () => {
  state.detailController?.abort();
  state.selected = null;
});
$("#product-dialog").addEventListener("click", (event) => {
  if (event.target === $("#product-dialog")) $("#product-dialog").close();
});
document.addEventListener("keydown", (event) => {
  const target = event.target;
  const typing =
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
  if (
    !$("#product-dialog").open &&
    ((event.key === "/" &&
      !typing &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey) ||
      ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k"))
  ) {
    event.preventDefault();
    $("#search-input").focus();
    $("#search-input").select();
  }
});

function readLocation() {
  const params = new URLSearchParams(location.search);
  const mode = params.get("mode");
  state.mode = Object.hasOwn(MODES, mode) ? mode : "attribute";
  state.compare = params.get("compare") === "1";
  const query = (params.get("q") || "").slice(0, 256).trim();
  $("#search-input").value = query;
  syncControls();
  if (query) performSearch({ query, updateLocation: false });
  else {
    state.controller?.abort();
    ++state.sequence;
    state.query = "";
    state.responses = {};
    state.loading = false;
    state.limit = 12;
    $("#results-title").textContent = "Начните с детали.";
    $("#results-eyebrow").textContent = "Ваш следующий запрос";
    $("#results-statistics").replaceChildren();
    $("#results").setAttribute("aria-busy", "false");
    $("#results").replaceChildren(welcomeContent.cloneNode(true));
    $("#more-container").hidden = true;
  }
}

window.addEventListener("popstate", readLocation);
loadMeta();
readLocation();
