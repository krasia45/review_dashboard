// ============================================================
// 대화형 HTML 대시보드 — 카테고리/제품 필터 + Chart.js 렌더링
// (Python이 __ALL_REVIEWS_JSON__ / __THRESHOLD__ 자리를 실제 값으로 치환해서 삽입한다)
// ============================================================
// Chart.js 전역 기본값 — 실제 기업 대시보드(Stripe/Linear/Mixpanel 계열)에서 흔한
// 톤: 옅은 그리드선, 절제된 폰트 크기, 툴팁 스타일을 전체 차트에 일관되게 적용한다.
if (typeof Chart !== "undefined") {
  Chart.defaults.font.family = "'Pretendard Variable','Pretendard',-apple-system,BlinkMacSystemFont,sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#64748B";
  Chart.defaults.borderColor = "#F1F5F9";
  Chart.defaults.plugins.tooltip.backgroundColor = "#0F172A";
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.titleFont = { weight: "600" };
  Chart.defaults.plugins.tooltip.bodySpacing = 4;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.boxHeight = 8;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.scale.grid.color = "#F1F5F9";
  Chart.defaults.scale.ticks.color = "#94A3B8";
}

const ALL_REVIEWS = __ALL_REVIEWS_JSON__;
const TREND_MA_WINDOW = 3; // 시간별 감정 추이: N일 이동평균

// 감정 점수 3단계 (부정/중립/긍정) — src/utils.py의 SENTIMENT_GRADES와 동일한 팔레트
const GRADE_INFO = [
  { score: 1, label: "부정", color: "#DC2626" },
  { score: 2, label: "중립", color: "#94A3B8" },
  { score: 3, label: "긍정", color: "#16A34A" },
];
const SENT_COLORS = { positive: "#16A34A", neutral: "#94A3B8", negative: "#DC2626" };
const SENT_LABEL = { positive: "긍정", neutral: "중립", negative: "부정" };
// 보너스: 다국어 지원 — 한국어/영어/중국어 3개 언어 라벨
const LANG_LABEL = { ko: "한국어", en: "영어", zh: "중국어" };

// Python의 utils.sentiment_grade() 와 동일한 로직 (긍정->3/중립->2/부정->1)
function sentimentGrade(sentiment) {
  if (sentiment === "positive") return GRADE_INFO[2];
  if (sentiment === "negative") return GRADE_INFO[0];
  return GRADE_INFO[1];
}

// N일 트레일링 이동평균 (앞쪽 구간은 있는 만큼만으로 평균내서 배열 길이를 유지한다)
function movingAverage(values, window) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

// 카테고리 -> 그 안에 있는 제품 목록
const CATEGORY_PRODUCTS = {};
ALL_REVIEWS.forEach((r) => {
  if (!r.category) return;
  if (!CATEGORY_PRODUCTS[r.category]) CATEGORY_PRODUCTS[r.category] = new Set();
  if (r.product) CATEGORY_PRODUCTS[r.category].add(r.product);
});

let charts = {};

function populateFilters() {
  const catSel = document.getElementById("catFilter");
  Object.keys(CATEGORY_PRODUCTS).sort().forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    catSel.appendChild(opt);
  });
  updateProductOptions();
}

function updateProductOptions() {
  const cat = document.getElementById("catFilter").value;
  const prodSel = document.getElementById("prodFilter");
  const prevValue = prodSel.value;
  prodSel.innerHTML = '<option value="__all__">전체 제품</option>';
  let products;
  if (cat === "__all__") {
    products = [...new Set(ALL_REVIEWS.map((r) => r.product).filter(Boolean))];
  } else {
    products = [...(CATEGORY_PRODUCTS[cat] || [])];
  }
  products.sort().forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    prodSel.appendChild(opt);
  });
  // 이전에 고른 제품이 새 카테고리 안에도 있으면 유지, 없으면 "전체 제품"으로
  if (products.includes(prevValue)) prodSel.value = prevValue;
}

function getFiltered() {
  const cat = document.getElementById("catFilter").value;
  const prod = document.getElementById("prodFilter").value;
  return ALL_REVIEWS.filter((r) => {
    if (cat !== "__all__" && r.category !== cat) return false;
    if (prod !== "__all__" && r.product !== prod) return false;
    return true;
  });
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

function renderAll() {
  const rows = getFiltered();
  const prodSelected = document.getElementById("prodFilter").value !== "__all__";

  renderKPIs(rows);
  renderDonut(rows);
  renderTrend(rows);
  renderGrade(rows);
  renderLanguage(rows);
  renderAspects(rows);
  toggleComparisonCharts(!prodSelected);
  if (!prodSelected) {
    renderProductComparison(rows);
    renderProductBreakdown(rows);
  }
  updateFilterLabel(rows.length);
  updateEmptyNote(rows.length);
}

function toggleComparisonCharts(show) {
  document.getElementById("cardProductComparison").style.display = show ? "" : "none";
  document.getElementById("cardProductBreakdown").style.display = show ? "" : "none";
  document.getElementById("compareHiddenNote").style.display = show ? "none" : "block";
}

function updateFilterLabel(count) {
  const cat = document.getElementById("catFilter").value;
  const prod = document.getElementById("prodFilter").value;
  let label = "전체";
  if (cat !== "__all__") label = cat;
  if (prod !== "__all__") label = prod;
  document.getElementById("filterLabel").textContent = `${label} (${count}건)`;
}

function updateEmptyNote(count) {
  document.getElementById("emptyNote").style.display = count === 0 ? "block" : "none";
}

function resetFilters() {
  document.getElementById("catFilter").value = "__all__";
  updateProductOptions();
  document.getElementById("prodFilter").value = "__all__";
  renderAll();
}

function renderKPIs(rows) {
  const total = rows.length;
  const analyzed = rows.filter((r) => r.sentiment).length;
  const positive = rows.filter((r) => r.sentiment === "positive").length;
  const posRatio = analyzed ? (positive / analyzed) * 100 : 0;
  const completion = total ? (analyzed / total) * 100 : 0;
  const ratings = rows.filter((r) => r.rating).map((r) => r.rating);
  const avgRating = ratings.length ? ratings.reduce((a, b) => a + b, 0) / ratings.length : 0;
  const confidences = rows.filter((r) => r.sentiment && r.confidence != null).map((r) => r.confidence);
  const avgConf = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0;
  const grades = rows.filter((r) => r.sentiment).map((r) => sentimentGrade(r.sentiment).score);
  const avgGrade = grades.length ? grades.reduce((a, b) => a + b, 0) / grades.length : 0;

  document.getElementById("kpiTotal").textContent = total + "건";
  document.getElementById("kpiRate").textContent = completion.toFixed(1) + "%";
  document.getElementById("kpiPos").textContent = posRatio.toFixed(1) + "%";
  document.getElementById("kpiRating").textContent = avgRating.toFixed(2);
  document.getElementById("kpiGrade").textContent = avgGrade.toFixed(2);
  document.getElementById("kpiConf").textContent = avgConf.toFixed(2);
}

function renderDonut(rows) {
  destroyChart("donut");
  const pos = rows.filter((r) => r.sentiment === "positive").length;
  const neu = rows.filter((r) => r.sentiment === "neutral").length;
  const neg = rows.filter((r) => r.sentiment === "negative").length;
  charts.donut = new Chart(document.getElementById("chartDonut"), {
    type: "doughnut",
    data: {
      labels: ["긍정", "중립", "부정"],
      datasets: [{ data: [pos, neu, neg], backgroundColor: [SENT_COLORS.positive, SENT_COLORS.neutral, SENT_COLORS.negative], borderWidth: 2, borderColor: "#fff" }],
    },
    options: { cutout: "70%", plugins: { legend: { position: "bottom" } }, animation: { duration: 300 } },
  });
}

function renderTrend(rows) {
  destroyChart("trend");
  const byDate = {};
  rows.forEach((r) => {
    if (!r.date || !r.sentiment) return;
    if (!byDate[r.date]) byDate[r.date] = { positive: 0, neutral: 0, negative: 0 };
    byDate[r.date][r.sentiment]++;
  });
  const dates = Object.keys(byDate).sort();
  // 날짜별 건수를 그대로 찍으면 하루 등락에 선이 너무 들쭉날쭉해서, N일 이동평균으로
  // 부드럽게 다듬은 선을 그린다 (급격한 하루짜리 이상치보다 추세 자체가 잘 보이도록).
  charts.trend = new Chart(document.getElementById("chartTrend"), {
    type: "line",
    data: {
      labels: dates,
      datasets: ["positive", "neutral", "negative"].map((k) => ({
        label: `${SENT_LABEL[k]} (${TREND_MA_WINDOW}일 이동평균)`,
        data: movingAverage(dates.map((d) => byDate[d][k]), TREND_MA_WINDOW),
        borderColor: SENT_COLORS[k], backgroundColor: SENT_COLORS[k] + "1A",
        tension: 0.35, pointRadius: 0, pointHoverRadius: 4, borderWidth: 2.25,
        fill: true,
      })),
    },
    options: { plugins: { legend: { position: "top" } }, scales: { y: { beginAtZero: true } }, animation: { duration: 300 } },
  });
}

function renderGrade(rows) {
  destroyChart("grade");
  const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  rows.forEach((r) => { if (r.sentiment) counts[sentimentGrade(r.sentiment).score]++; });
  charts.grade = new Chart(document.getElementById("chartGrade"), {
    type: "bar",
    data: {
      labels: GRADE_INFO.map((g) => g.score + "점 " + g.label),
      datasets: [{ data: GRADE_INFO.map((g) => counts[g.score]), backgroundColor: GRADE_INFO.map((g) => g.color), borderRadius: 6, borderSkipped: false, maxBarThickness: 28 }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, ticks: { precision: 0 } } }, animation: { duration: 300 } },
  });
}

function renderLanguage(rows) {
  destroyChart("lang");
  const byLang = {};
  rows.forEach((r) => {
    if (!r.language) return;
    if (!byLang[r.language]) byLang[r.language] = { count: 0, positive: 0, analyzed: 0 };
    byLang[r.language].count++;
    if (r.sentiment) {
      byLang[r.language].analyzed++;
      if (r.sentiment === "positive") byLang[r.language].positive++;
    }
  });
  const langs = Object.keys(byLang).sort((a, b) => byLang[b].count - byLang[a].count);
  charts.lang = new Chart(document.getElementById("chartLanguage"), {
    type: "bar",
    data: { labels: langs.map((l) => LANG_LABEL[l] || l), datasets: [{ label: "리뷰 수", data: langs.map((l) => byLang[l].count), backgroundColor: "#1E293B", borderRadius: 6, borderSkipped: false, maxBarThickness: 28 }] },
    options: { indexAxis: "y", plugins: { legend: { position: "top" } }, scales: { x: { beginAtZero: true, ticks: { precision: 0 } } }, animation: { duration: 300 } },
  });
}

// [사용자 요청 추가] 배송/상품/응대 측면 만족도를 5점 만점으로 집계해서 보여준다.
// positive=5, neutral=3, negative=1, not_mentioned은 평균에서 제외(0점 취급하면 왜곡됨).
const ASPECT_IDS_JS = ["product", "delivery", "service"];
const ASPECT_LABEL_JS = { product: "상품", delivery: "배송", service: "응대" };
const ASPECT_SCORE_MAP_JS = { positive: 5, neutral: 3, negative: 1, not_mentioned: null };

function renderAspects(rows) {
  destroyChart("aspects");
  const analyzed = rows.filter((r) => r.sentiment && r.aspects);
  const avgScores = ASPECT_IDS_JS.map((aid) => {
    const scores = analyzed
      .map((r) => ASPECT_SCORE_MAP_JS[r.aspects[aid] || "not_mentioned"])
      .filter((s) => s !== null && s !== undefined);
    return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  });
  const mentionedCounts = ASPECT_IDS_JS.map(
    (aid) => analyzed.filter((r) => (r.aspects[aid] || "not_mentioned") !== "not_mentioned").length
  );
  charts.aspects = new Chart(document.getElementById("chartAspects"), {
    type: "bar",
    data: {
      labels: ASPECT_IDS_JS.map((aid, i) => `${ASPECT_LABEL_JS[aid]} 만족도 (${mentionedCounts[i]}건 언급)`),
      datasets: [{ label: "평균 점수(5점 만점)", data: avgScores.map((s) => Math.round(s * 100) / 100),
        backgroundColor: avgScores.map((s) => (s >= 4 ? SENT_COLORS.positive : s >= 2.5 ? "#D97706" : SENT_COLORS.negative)),
        borderRadius: 6, borderSkipped: false, maxBarThickness: 28 }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, max: 5 } }, animation: { duration: 300 } },
  });
}

// 제품이 많아져도(12개 등) 막대가 잘리지 않도록, 캔버스를 감싼 래퍼 div의 높이를
// 제품 수에 비례해서 늘린다. Chart.js는 responsive:true + maintainAspectRatio:false일 때
// "캔버스를 감싼 부모 요소"의 실제 렌더링 크기를 기준으로 그리므로, 캔버스 자체가 아니라
// 래퍼에 높이를 줘야 한다 (캔버스에 직접 주면 Chart.js 내부 리사이즈 로직이 다시 덮어써서
// 의도한 것보다 훨씬 크게(또는 작게) 렌더링되는 문제가 있었다).
function _sizeWrapForCategories(wrapId, count) {
  const height = Math.max(220, count * 34 + 60);
  document.getElementById(wrapId).style.height = height + "px";
}

function renderProductComparison(rows) {
  destroyChart("prodComp");
  const byProd = {};
  rows.forEach((r) => {
    if (!r.product) return;
    if (!byProd[r.product]) byProd[r.product] = { positive: 0, neutral: 0, negative: 0, ratings: [] };
    if (r.sentiment) byProd[r.product][r.sentiment]++;
    if (r.rating) byProd[r.product].ratings.push(r.rating);
  });
  const products = Object.keys(byProd);
  const posRatio = (p) => {
    const d = byProd[p];
    const t = d.positive + d.neutral + d.negative;
    return t ? (d.positive / t) * 100 : 0;
  };
  products.sort((a, b) => posRatio(a) - posRatio(b));
  const posRatios = products.map(posRatio);
  _sizeWrapForCategories("wrapProductComparison", products.length);
  charts.prodComp = new Chart(document.getElementById("chartProductComparison"), {
    type: "bar",
    data: { labels: products, datasets: [{ label: "긍정비율(%)", data: posRatios, backgroundColor: posRatios.map((v) => (v >= 40 ? SENT_COLORS.positive : v >= 25 ? "#D97706" : SENT_COLORS.negative)), borderRadius: 5, borderSkipped: false, maxBarThickness: 24 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, max: 100 } }, animation: { duration: 300 } },
  });
}

function renderProductBreakdown(rows) {
  destroyChart("prodBreak");
  const byProd = {};
  rows.forEach((r) => {
    if (!r.product || !r.sentiment) return;
    if (!byProd[r.product]) byProd[r.product] = { positive: 0, neutral: 0, negative: 0 };
    byProd[r.product][r.sentiment]++;
  });
  const products = Object.keys(byProd);
  const total = (p) => byProd[p].positive + byProd[p].neutral + byProd[p].negative;
  products.sort((a, b) => byProd[a].positive / (total(a) || 1) - byProd[b].positive / (total(b) || 1));
  _sizeWrapForCategories("wrapProductBreakdown", products.length);
  charts.prodBreak = new Chart(document.getElementById("chartProductBreakdown"), {
    type: "bar",
    data: {
      labels: products,
      datasets: ["negative", "neutral", "positive"].map((k) => ({ label: SENT_LABEL[k], data: products.map((p) => byProd[p][k]), backgroundColor: SENT_COLORS[k], maxBarThickness: 24 })),
    },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "top" } }, scales: { x: { stacked: true, beginAtZero: true, ticks: { precision: 0 } }, y: { stacked: true } }, animation: { duration: 300 } },
  });
}

document.getElementById("catFilter").addEventListener("change", () => {
  updateProductOptions();
  renderAll();
});
document.getElementById("prodFilter").addEventListener("change", renderAll);
document.getElementById("resetFilterBtn").addEventListener("click", resetFilters);

populateFilters();
renderAll();
