const $ = (selector) => document.querySelector(selector);
const format = new Intl.NumberFormat("ru-RU", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 4,
});
const integer = new Intl.NumberFormat("ru-RU");
const METHODS = [
  "baseline",
  "attribute",
  "bm25",
  "tfidf",
  "rule",
  "no_product_evidence",
];
const LABELS = {
  baseline: "Базовая модель",
  attribute: "Модель с атрибутами",
  bm25: "BM25",
  tfidf: "TF–IDF",
  rule: "Базовая модель + правила",
  no_product_evidence: "Атрибуты запроса · абляция",
};
const PARTITIONS = {
  train: "Обучение",
  dev: "Development",
  test: "Финальный test",
};
const LIMITATIONS = {
  "Reranking judged pools; no full-catalogue recall or purchase/profit effect measured":
    "Измерено ранжирование размеченных кандидатов. Полнота по всему каталогу, покупки и экономический эффект не оценивались.",
  "Challenge-selected queries are not representative traffic":
    "Запросы отобраны для сложного benchmark и не представляют весь поисковый трафик.",
  "Current released test is public, with an internally frozen selection protocol":
    "Test опубликован. Внутренний протокол выбора модели зафиксирован до оценки его качества.",
  "WANDS Partial is not ESCI Substitute; transfer gain/ambiguity scenarios are descriptive":
    "Partial в WANDS не равен Substitute в ESCI. Варианты весов и неоднозначных labels — проверки чувствительности.",
  "Attribute evidence is conservative and soft, not certified physical compatibility":
    "Разбор атрибутов — признаки из текста. Он не подтверждает физическую совместимость товаров.",
  "No new independent human attribute labels are claimed by this experiment":
    "Новая независимая человеческая разметка атрибутов для этого эксперимента не собиралась.",
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
function number(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function valueText(value) {
  const result = number(value);
  return result === null ? "—" : format.format(result);
}
function signed(value) {
  const result = number(value);
  return result === null
    ? "—"
    : `${result > 0 ? "+" : ""}${format.format(result)}`;
}
function ndcg(metrics) {
  return number(metrics?.ndcg_at_10) ?? number(metrics?.ndcg10);
}
function methodsIn(models) {
  return [
    ...METHODS.filter((name) => Object.hasOwn(models, name)),
    ...Object.keys(models).filter((name) => !METHODS.includes(name)),
  ];
}
function dateText(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? String(value || "—")
    : new Intl.DateTimeFormat("ru-RU", {
        day: "numeric",
        month: "long",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

function sampleFact(label, value) {
  const element = node("div");
  element.append(
    node("strong", null, number(value) === null ? "—" : integer.format(value)),
    node("span", null, label),
  );
  return element;
}

function renderComparison(comparison, target, primary = false) {
  const difference = number(comparison?.difference);
  const lower = number(comparison?.ci_low);
  const upper = number(comparison?.ci_high);
  const container = $(target);
  container.replaceChildren();
  if (primary) {
    $("#primary-value").textContent = signed(difference);
    const verdict =
      lower === null || upper === null
        ? "Доверительный интервал не указан в отчёте."
        : lower > 0
          ? "На этой выборке весь интервал разницы выше нуля."
          : upper < 0
            ? "На этой выборке весь интервал разницы ниже нуля."
            : "Интервал включает ноль. Устойчивый рост качества не подтверждён.";
    $("#primary-verdict").textContent = verdict;
    $("#primary-sample").replaceChildren(
      sampleFact("запросов", comparison?.queries),
      sampleFact("семейств", comparison?.families),
    );
    $("#primary-bootstrap-note").textContent = comparison?.bootstrap_repeats
      ? `95% парный bootstrap по семействам запросов · ${integer.format(comparison.bootstrap_repeats)} повторов. Интервал относится к этой выборке и протоколу.`
      : "Парный интервал по семействам запросов. Подробности — в исходном отчёте.";
  }
  if (lower === null || upper === null || difference === null) {
    container.append(
      node(
        "p",
        "report-footnote",
        "Интервальная оценка в этом отчёте отсутствует.",
      ),
    );
    return;
  }
  container.append(
    node(
      "div",
      "confidence-caption",
      `95% интервал: ${signed(lower)} … ${signed(upper)}`,
    ),
  );
  const extent =
    Math.max(Math.abs(lower), Math.abs(upper), Math.abs(difference), 0.0001) *
    1.3;
  const position = (value) => `${50 + (value / extent) * 50}%`;
  const track = node("div", "confidence-track");
  track.setAttribute("role", "img");
  track.setAttribute(
    "aria-label",
    `Разница ${signed(difference)}, 95% интервал от ${signed(lower)} до ${signed(upper)}`,
  );
  track.append(node("span", "confidence-zero"));
  const interval = node("span", "confidence-interval");
  interval.style.left = position(lower);
  interval.style.width = `${((upper - lower) / (2 * extent)) * 100}%`;
  const point = node("span", "confidence-point");
  point.style.left = position(difference);
  track.append(interval, point);
  container.append(track);
  const scale = node("div", "confidence-scale");
  scale.append(
    node("span", null, signed(-extent)),
    node("span", null, "0"),
    node("span", null, signed(extent)),
  );
  container.append(scale);
}

function renderBars(models) {
  const container = $("#method-bars");
  container.replaceChildren();
  methodsIn(models).forEach((name) => {
    const value = ndcg(models[name]);
    const row = node(
      "div",
      `method-bar-row ${name === "attribute" ? "is-primary" : ""} ${name === "no_product_evidence" ? "is-ablation" : ""}`,
    );
    row.append(node("div", "method-bar-label", LABELS[name] || name));
    const track = node("div", "method-bar-track");
    const bar = node("span", "method-bar-fill");
    bar.style.width = `${value === null ? 0 : Math.max(0, Math.min(value, 1)) * 100}%`;
    track.append(bar);
    row.append(track, node("div", "method-bar-value", valueText(value)));
    container.append(row);
  });
}

function renderSplits(report) {
  const rows = $("#split-rows");
  rows.replaceChildren();
  (Array.isArray(report.splits) ? report.splits : []).forEach((split) => {
    const row = node("tr");
    row.append(
      node("th", null, PARTITIONS[split.partition] || split.partition),
    );
    row.firstElementChild.scope = "row";
    for (const key of ["queries", "families", "pairs"])
      row.append(
        node(
          "td",
          null,
          number(split[key]) === null ? "—" : integer.format(split[key]),
        ),
      );
    rows.append(row);
  });
  const dataset = report.dataset || {};
  const notes = [];
  if (number(dataset.excluded_official_train_queries) !== null)
    notes.push(
      `Из official train удалено ${integer.format(dataset.excluded_official_train_queries)} запросов с семействами, встречающимися в test.`,
    );
  if (number(dataset.test_products_seen_in_train) !== null)
    notes.push(
      `${integer.format(dataset.test_products_seen_in_train)} товаров test встречались в обучающих запросах: это проверка новых семейств запросов, а не новых товаров.`,
    );
  $("#data-overlap").textContent = notes.join(" ");
  const gains = $("#gain-map");
  gains.replaceChildren(node("span", "gain-label", "Веса ESCI"));
  Object.entries(report.gains || {}).forEach(([label, gain]) =>
    gains.append(
      node(
        "span",
        "gain-item",
        `${label} = ${number(gain) === null ? String(gain) : new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 }).format(gain)}`,
      ),
    ),
  );
  gains.hidden = !Object.keys(report.gains || {}).length;
}

function metricCells(row, values) {
  values.forEach((value) => row.append(node("td", null, valueText(value))));
}

function renderWands(wands) {
  const main = wands?.scenarios?.consensus_main?.exact_binary;
  $("#wands-section").hidden = !main;
  if (!main) return;
  const summary = $("#wands-summary");
  summary.replaceChildren();
  summary.append(
    sampleFact("запросов", main.queries),
    sampleFact("размеченных пар", main.pairs),
  );
  const comparison = node("div", "transfer-difference");
  comparison.append(
    node("span", null, "Атрибуты − baseline"),
    node("strong", null, signed(main.primary_difference?.difference)),
    node(
      "small",
      null,
      `95% интервал ${signed(main.primary_difference?.ci_low)} … ${signed(main.primary_difference?.ci_high)}`,
    ),
  );
  summary.append(comparison);
  const rows = $("#wands-rows");
  rows.replaceChildren();
  methodsIn(main.models || {}).forEach((name) => {
    const row = node("tr", name === "attribute" ? "highlight-row" : "");
    const label = node("th", null, LABELS[name] || name);
    label.scope = "row";
    row.append(label);
    metricCells(row, [
      ndcg(main.models[name]),
      number(main.models[name]?.exact_recall10),
      number(main.models[name]?.mrr),
    ]);
    rows.append(row);
  });
  const sensitivity = $("#sensitivity-rows");
  sensitivity.replaceChildren();
  const scenarioLabels = {
    consensus_main: "Без спорных пар",
    ambiguous_min: "Спорные: нижняя label",
    ambiguous_max: "Спорные: верхняя label",
  };
  const gainLabels = {
    exact_binary: "0 · Exact binary",
    "partial_0.1": "0,1",
    "partial_0.5": "0,5",
  };
  for (const [scenario, maps] of Object.entries(wands.scenarios || {})) {
    for (const [map, value] of Object.entries(maps)) {
      const row = node("tr");
      row.append(
        node("th", null, scenarioLabels[scenario] || scenario),
        node("td", null, gainLabels[map] || map),
      );
      metricCells(row, [
        ndcg(value.models?.baseline),
        ndcg(value.models?.attribute),
      ]);
      row.append(
        node("td", null, signed(value.primary_difference?.difference)),
      );
      sensitivity.append(row);
    }
  }
  $("#sensitivity-details").hidden = !sensitivity.childElementCount;
}

function renderProvenance(report) {
  const container = $("#provenance");
  container.replaceChildren();
  const entries = [
    ["Завершение", dateText(report.completed_at_utc)],
    ["Выбор настроек", dateText(report.selection?.selected_at_utc)],
    ["Протокол SHA256", report.provenance?.protocol_sha256],
    ["Конфигурация SHA256", report.provenance?.config_sha256],
    ["Выбор модели SHA256", report.provenance?.selection_sha256],
    ["ESCI revision", report.dataset?.source_revision],
    [
      "Время эксперимента",
      number(report.runtime_seconds) === null
        ? null
        : `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(report.runtime_seconds)} с`,
    ],
  ];
  entries.forEach(([label, value]) => {
    if (!value) return;
    const row = node("div");
    row.append(node("dt", null, label), node("dd", null, value));
    container.append(row);
  });
  const limitations = $("#limitations");
  limitations.replaceChildren();
  (Array.isArray(report.limitations) ? report.limitations : []).forEach(
    (limitation) =>
      limitations.append(
        node("li", null, LIMITATIONS[limitation] || limitation),
      ),
  );
}

async function loadReport() {
  $("#report-error").hidden = true;
  $("#report-loading").hidden = false;
  try {
    const response = await fetch("/api/report", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok)
      throw new Error(
        response.status === 404
          ? "Эксперимент ещё не опубликовал итоговый отчёт. Можно вернуться к поиску и проверить страницу позже."
          : "Не удалось получить сохранённый отчёт. Повторите попытку.",
      );
    const report = await response.json();
    if (
      !report ||
      typeof report.esci !== "object" ||
      !report.primary_difference
    )
      throw new Error("В отчёте нет результатов основной проверки.");
    renderComparison(report.primary_difference, "#primary-ci", true);
    renderBars(report.esci);
    renderSplits(report);
    renderWands(report.wands);
    renderProvenance(report);
    $("#report-completion").replaceChildren(
      node("span", "status-dot"),
      node(
        "span",
        null,
        report.status === "completed"
          ? `Оценка завершена · ${dateText(report.completed_at_utc)}`
          : `Состояние отчёта: ${report.status || "не указано"}`,
      ),
    );
    $("#report-content").hidden = false;
  } catch (error) {
    $("#report-error-message").textContent = error.message;
    $("#report-error").hidden = false;
    $("#report-completion").replaceChildren(
      node("span", "status-dot is-error"),
      node("span", null, "Результаты недоступны"),
    );
  } finally {
    $("#report-loading").hidden = true;
  }
}

$("#report-retry").addEventListener("click", loadReport);
loadReport();
