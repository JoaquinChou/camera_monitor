/* AIS 时空统计分析前端逻辑：数据拉取 + 联动图表渲染（无框架依赖）。
 *
 * 与 app.js 解耦：本文件只负责 AIS 视图，通过 window.AIS.init() 暴露给
 * 视图切换逻辑调用。数据来自 /api/ais（后端 ais_data.py）。
 */
window.AIS = (function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------- 常量 ----------
  const HOURS = [
    "00:00–04:00", "04:00–08:00", "08:00–12:00",
    "12:00–16:00", "16:00–20:00", "20:00–24:00",
  ];
  const HOUR_STARTS = [0, 4, 8, 12, 16, 20];

  // 深色主题下「低 → 高」的 7 级青色渐变（历史图 7 级、预测图 6 级）。
  const TEAL = ["#17323a", "#1f5248", "#2a7d66", "#3aa488", "#51c3a4", "#74dcc0", "#a8f2dd"];
  const FORECAST_PALETTE = TEAL.slice(1);
  const FORECAST_BINS = [1, 5, 20, 50, 100];

  const SVG = {
    left: 56, top: 17, cell: 7.2, w: 576, h: 403.2,
    plot: "#0d1626", grid: "#22344c", axis: "#8aa0ba", ring: "#f2b84b",
  };

  const METHODS = [
    { key: "yesterday", label: "沿用昨天同一时段", tag: "" },
    { key: "mean_7d", label: "过去 7 天同一时段均值", tag: "热点输出" },
    { key: "poisson_gbdt", label: "泊松梯度提升树", tag: "" },
  ];

  const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 });
  const fmt = (v) => number.format(v);
  const pct = (v) => (v * 100).toFixed(1) + "%";

  let bundle = null;
  let cells = new Map();      // "y,x" -> {y, x, lat, lon, label}
  let historyState = { start: 0, end: 26, slot: -1, grid: "" };
  let fcHour = 8;

  // ---------- 坐标 / 图元 ----------
  function scaleX(lon) { return SVG.left + (lon + 123.8) * 10 * SVG.cell; }
  function scaleY(lat) { return SVG.top + (47.1 - lat) * 10 * SVG.cell; }
  // 以网格中心定位：西边缘 = lon-0.05，北边缘 = lat+0.05。
  function tileXY(lat, lon) { return { x: scaleX(lon - 0.05), y: scaleY(lat + 0.05) }; }

  function gridLinesSvg() {
    let s = "";
    for (let lon = -123; lon <= -116; lon++) {
      const x = scaleX(lon).toFixed(1);
      s += `<path d="M${x},${SVG.top} v${SVG.h}"/><text x="${x}" y="438">${lon}°</text>`;
    }
    for (let lat = 42; lat <= 47; lat++) {
      const y = scaleY(lat).toFixed(1);
      s += `<path d="M${SVG.left},${y} h${SVG.w}"/><text class="y-label" x="48" y="${(+y + 4).toFixed(1)}">${lat}°</text>`;
    }
    return s;
  }

  function axisCaptions() {
    return (
      '<text class="axis-caption" x="344" y="455">经度</text>' +
      '<text class="axis-caption" transform="translate(14,219) rotate(-90)">纬度</text>'
    );
  }

  const sum6 = (counts) => counts.slice(0, 6).reduce((a, b) => a + b, 0);
  const isWeekend = (date) => [0, 6].includes(new Date(date + "T00:00:00Z").getUTCDay());

  // ---------- 数据加载 ----------
  async function init() {
    if (bundle) return;
    try {
      const res = await fetch("/api/ais");
      if (!res.ok) throw new Error("/api/ais -> " + res.status);
      bundle = await res.json();
    } catch (err) {
      console.error("[ais] 数据加载失败:", err);
      const el = $("#ais-summary");
      if (el) el.innerHTML =
        '<p class="ais-error">AIS 数据加载失败，请确认服务已启动且 AIS 数据文件存在。</p>';
      return;
    }
    buildCells();
    render();
  }

  function buildCells() {
    cells = new Map();
    for (const row of bundle.history.rows) {
      const y = row[1], x = row[2];
      const key = y + "," + x;
      if (!cells.has(key)) {
        const lat = (y + 0.5) / 10, lon = (x + 0.5) / 10;
        cells.set(key, { y, x, lat, lon, label: lat.toFixed(2) + ", " + lon.toFixed(2) });
      }
    }
  }

  // ---------- 一次性渲染 ----------
  function render() {
    renderSummary();
    renderHistoryDesc();
    populateSelects();
    renderBacktest();
    renderForecast();
    bindEvents();
    renderHistory();
  }

  // ---------- 摘要卡片 ----------
  function renderSummary() {
    const s = bundle.summary;
    const t = bundle.metrics.test;
    const m7 = t.mean_7d;

    const dayMax = Math.max(s.day_count, s.night_count) || 1;
    const bars = (day, night) => `
      <div class="ais-summary-bar"><span>白天</span><span class="ais-summary-track" aria-hidden="true"><i style="width:${(day / dayMax * 100).toFixed(1)}%"></i></span><strong>${fmt(day)}</strong></div>
      <div class="ais-summary-bar is-highlight"><span>夜间</span><span class="ais-summary-track" aria-hidden="true"><i style="width:${(night / dayMax * 100).toFixed(1)}%"></i></span><strong>${fmt(night)}</strong></div>`;

    const capMax = Math.max(t.yesterday.top10_count_capture, m7.top10_count_capture, t.poisson_gbdt.top10_count_capture) || 1;
    const methodBar = (label, v, highlight) => `
      <div class="ais-summary-bar ${highlight ? "is-highlight" : ""}"><span>${label}</span><span class="ais-summary-track" aria-hidden="true"><i style="width:${(v / capMax * 100).toFixed(1)}%"></i></span><strong>${pct(v)}</strong></div>`;
    const methodBars =
      methodBar("昨日同期", t.yesterday.top10_count_capture, false) +
      methodBar("7 日同期均值", m7.top10_count_capture, true) +
      methodBar("泊松提升树", t.poisson_gbdt.top10_count_capture, false);

    $("#ais-summary").innerHTML = `
      <article class="ais-summary-card">
        <h3><span>01</span> 统计时空分布规律</h3>
        <p class="ais-summary-goal">目标覆盖昼夜、节假日、禁渔期，以及江段和重点水域。</p>
        <p class="ais-summary-result"><strong>现有结果</strong>　夜间观测略多。活动最集中的网格中心为 <strong>(${s.peak_grid_lat.toFixed(2)}, ${s.peak_grid_lon.toFixed(2)})</strong>，全期累计 <strong>${fmt(s.peak_grid_count)}</strong> 个网格时段观测计数。</p>
        <figure class="ais-summary-figure">
          <figcaption>昼夜观测计数 · 各 12 小时</figcaption>
          <div class="ais-summary-bars">${bars(s.day_count, s.night_count)}</div>
        </figure>
        <a class="ais-summary-link" href="#ais-history">查看时空统计与缺失维度 <span aria-hidden="true">↓</span></a>
      </article>
      <article class="ais-summary-card">
        <h3><span>02</span> 绘制热力图与时序图</h3>
        <p class="ais-summary-goal">目标是用地图显示违规集中区域，用时序图显示高发日期和时段。</p>
        <p class="ais-summary-result"><strong>现有结果</strong>　已用实际 AIS 观测绘制可联动的 <strong>${s.grids} 个网格</strong>热力图与 <strong>${s.days} 天</strong>时序图。</p>
        <a class="ais-summary-link" href="#ais-history">查看联动图表 <span aria-hidden="true">↓</span></a>
      </article>
      <article class="ais-summary-card">
        <h3><span>03</span> 建模预测与回测评估</h3>
        <p class="ais-summary-goal">将观测聚合成网格 × 4 小时计数，比较历史均值和泊松梯度提升树的次日预测。</p>
        <p class="ais-summary-result"><strong>现有结果</strong>　7 日同期均值的热点覆盖率最高（<strong>${pct(m7.top10_count_capture)}</strong>），全区峰值时段命中 <strong>${m7.peak_window_hits}/6 天</strong>，每网格时段平均绝对误差 <strong>${m7.mae.toFixed(2)} 艘</strong>。</p>
        <figure class="ais-summary-figure">
          <figcaption>测试集 · 每时段前 10 个网格覆盖的实际活动量</figcaption>
          <div class="ais-summary-bars">${methodBars}</div>
        </figure>
        <a class="ais-summary-link" href="#ais-backtest">查看算法与完整回测指标 <span aria-hidden="true">↓</span></a>
      </article>`;
  }

  function renderHistoryDesc() {
    const s = bundle.summary;
    $("#ais-history-desc").innerHTML =
      `从 2019-10-28 至 2019-11-23 的原始 AIS 记录汇总。全期共计 <strong>${fmt(s.grid_timeslot_count)}</strong> 个网格时段内的船舶观测计数。中心坐标 <strong>(${s.peak_grid_lat.toFixed(2)}, ${s.peak_grid_lon.toFixed(2)})</strong> 的网格累计最高，为 <strong>${fmt(s.peak_grid_count)}</strong>。全区 <strong>16:00–20:00</strong> 与 <strong>12:00–16:00</strong> 的计数接近持平。`;
  }

  // ---------- 历史联动图 ----------
  function populateSelects() {
    const dates = bundle.history.dates;
    const startEl = $("#ais-start"), endEl = $("#ais-end"), gridEl = $("#ais-grid");
    dates.forEach((date, i) => {
      startEl.add(new Option(date, i));
      endEl.add(new Option(date, i));
    });
    endEl.value = dates.length - 1;

    const totals = new Map();
    for (const row of bundle.history.rows) {
      const key = row[1] + "," + row[2];
      totals.set(key, (totals.get(key) || 0) + sum6(row.slice(3)));
    }
    const sortedCells = [...cells.values()]
      .sort((a, b) => (totals.get(b.y + "," + b.x) || 0) - (totals.get(a.y + "," + a.x) || 0));
    for (const c of sortedCells) gridEl.add(new Option(c.label, c.y + "," + c.x));
  }

  function renderHistory() {
    const { start, end, slot, grid } = historyState;
    const gridTotals = new Map();
    const hourly = [0, 0, 0, 0, 0, 0];
    const daily = Array(end - start + 1).fill(0);
    const phase = [0, 0];

    for (const row of bundle.history.rows) {
      const day = row[0], y = row[1], x = row[2], counts = row.slice(3);
      if (day < start || day > end) continue;
      const key = y + "," + x;
      const count = slot < 0 ? sum6(counts) : counts[slot];
      gridTotals.set(key, (gridTotals.get(key) || 0) + count);
      if (grid && grid !== key) continue;
      daily[day - start] += count;
      for (let i = 0; i < 6; i++) hourly[i] += counts[i];
      phase[0] += counts[6];
      phase[1] += counts[7];
    }

    renderHistoryMap(gridTotals, grid, slot);
    renderHistorySeries(hourly, daily, phase, start, end, slot, grid);
  }

  function renderHistoryMap(gridTotals, selected, slot) {
    let max = 0, topKey = "";
    for (const [key, count] of gridTotals) if (count > max) { max = count; topKey = key; }
    const thresholds = max
      ? [...new Set([1, ...[.025, .10, .25, .50, .75].map((r) => Math.ceil(max * r))])]
      : [];

    let tiles = "";
    for (const [key, c] of cells) {
      const count = gridTotals.get(key) || 0;
      const color = TEAL[thresholds.filter((t) => t <= count).length];
      const pos = tileXY(c.lat, c.lon);
      tiles +=
        `<rect class="ais-tile${key === selected ? " is-selected" : ""}" data-grid="${key}" ` +
        `x="${pos.x.toFixed(2)}" y="${pos.y.toFixed(2)}" width="7" height="7" fill="${color}">` +
        `<title>${c.label}｜${fmt(count)} 个观测计数</title></rect>`;
    }
    if (selected || topKey) {
      const c = cells.get(selected || topKey);
      const pos = tileXY(c.lat, c.lon);
      tiles +=
        `<circle cx="${(pos.x + 3.5).toFixed(2)}" cy="${(pos.y + 3.5).toFixed(2)}" r="7" ` +
        `fill="none" stroke="${SVG.ring}" stroke-width="2.2" pointer-events="none"/>`;
    }

    $("#ais-map").innerHTML =
      `<rect x="${SVG.left}" y="${SVG.top}" width="${SVG.w}" height="${SVG.h}" rx="3" fill="${SVG.plot}"/>` +
      `<g class="gridlines">${gridLinesSvg()}</g>${tiles}${axisCaptions()}`;

    const bins = [0, ...thresholds];
    $("#ais-legend").innerHTML = bins.map((low, i) => {
      const high = i === bins.length - 1 ? max : bins[i + 1] - 1;
      const label = low === high ? fmt(low) : fmt(low) + "–" + fmt(high);
      return `<span><i style="--swatch:${TEAL[i]}"></i>${label}</span>`;
    }).join("");

    const window = slot < 0 ? "全天" : HOURS[slot];
    if (selected || topKey) {
      const key = selected || topKey;
      const c = cells.get(key);
      $("#ais-map-status").textContent =
        `${window} · ${selected ? "所选网格" : "排名第一网格"} (${c.label})，计数 ${fmt(gridTotals.get(key) || 0)}。`;
    } else {
      $("#ais-map-status").textContent = `${window} · 当前日期和时段无轨迹计数。`;
    }
  }

  function renderHistorySeries(hourly, daily, phase, start, end, slot, grid) {
    const scope = grid ? "网格 " + $("#ais-grid").selectedOptions[0].text : "全区";
    $("#ais-series-status").textContent =
      `${scope} · ${bundle.history.dates[start]} 至 ${bundle.history.dates[end]} · 日内各时段合计`;

    const hourMax = Math.max(1, ...hourly);
    $("#ais-hour-bars").innerHTML = HOURS.map((label, i) =>
      `<button class="ais-hour-bar" type="button" data-slot="${i}" aria-pressed="${slot === i}">` +
      `<span>${label}</span><span class="ais-track"><i style="--width:${(hourly[i] / hourMax * 100).toFixed(1)}%"></i></span>` +
      `<strong>${fmt(hourly[i])}</strong></button>`
    ).join("");

    const dayMax = Math.max(1, ...daily);
    $("#ais-daily-status").textContent =
      `${slot < 0 ? "全天" : HOURS[slot]} · 纵向满刻度 ${fmt(dayMax)} 个观测计数`;
    const dayEl = $("#ais-day-bars");
    dayEl.style.setProperty("--day-count", daily.length);
    dayEl.innerHTML = daily.map((count, i) => {
      const date = bundle.history.dates[start + i];
      const label = `${date} · ${fmt(count)} 个观测计数`;
      return (
        `<button class="ais-day${isWeekend(date) ? " is-weekend" : ""}" type="button" data-day="${start + i}" ` +
        `title="${label}" aria-label="${label}"><i style="--height:${(count / dayMax * 100).toFixed(1)}%"></i></button>`
      );
    }).join("");
    $("#ais-day-labels").innerHTML =
      `<span>${bundle.history.dates[start]}</span>` +
      (start === end ? "" : `<span>${bundle.history.dates[end]}</span>`);

    const weekTotals = [0, 0], weekDays = [0, 0];
    daily.forEach((count, i) => {
      const idx = Number(isWeekend(bundle.history.dates[start + i]));
      weekTotals[idx] += count;
      weekDays[idx]++;
    });
    const avg = (i) => weekDays[i] ? fmt(weekTotals[i] / weekDays[i]) + `（${weekDays[i]} 天）` : "无样本日";
    $("#ais-total").textContent = `${scope} · ${fmt(daily.reduce((a, b) => a + b, 0))}`;
    $("#ais-weekpart").textContent = `工作日 ${avg(0)} ／ 周末 ${avg(1)}`;
    $("#ais-daynight").textContent = `白天 ${fmt(phase[0])} ／ 夜间 ${fmt(phase[1])}`;
  }

  // ---------- 回测对比 ----------
  function daySpan(range) {
    return Math.round((Date.parse(range[1]) - Date.parse(range[0])) / 86400000) + 1;
  }

  function metricsTable(dataset, totalDays) {
    const rows = METHODS.map((m) => ({ ...m, ...dataset[m.key] }));
    const best = {
      mae: Math.min(...rows.map((r) => r.mae)),
      rmse: Math.min(...rows.map((r) => r.rmse)),
      top10: Math.max(...rows.map((r) => r.top10_count_capture)),
      peak: Math.max(...rows.map((r) => r.peak_window_hits)),
    };
    const body = rows.map((r) =>
      `<tr><th scope="row">${r.label}${r.tag ? `<span class="ais-mini-tag">${r.tag}</span>` : ""}</th>` +
      `<td class="${r.mae === best.mae ? "best" : ""}">${r.mae.toFixed(3)}</td>` +
      `<td class="${r.rmse === best.rmse ? "best" : ""}">${r.rmse.toFixed(3)}</td>` +
      `<td class="${r.top10_count_capture === best.top10 ? "best" : ""}">${pct(r.top10_count_capture)}</td>` +
      `<td class="${r.peak_window_hits === best.peak ? "best" : ""}">${r.peak_window_hits} / ${totalDays}</td></tr>`
    ).join("");
    return (
      `<thead><tr><th scope="col">预测方法</th><th scope="col">MAE ↓</th><th scope="col">RMSE ↓</th>` +
      `<th scope="col">Top 10 覆盖率 ↑</th><th scope="col">峰值时段命中 ↑</th></tr></thead><tbody>${body}</tbody>`
    );
  }

  function renderBacktest() {
    const m = bundle.metrics;
    const testDays = daySpan(m.test_dates);
    const valDays = daySpan(m.validation_dates);
    $("#ais-test-table").innerHTML =
      `<caption>测试集表现 · 2019-11-18 至 2019-11-23</caption>` + metricsTable(m.test, testDays);
    $("#ais-val-table").innerHTML =
      `<caption>验证集表现 · 2019-11-13 至 2019-11-17</caption>` + metricsTable(m.validation, valDays);
    $("#ais-backtest-conclusion").innerHTML =
      `<strong>核心结论。</strong> 验证集按 Top 10 覆盖率选择 7 日均值（${pct(m.validation.mean_7d.top10_count_capture)}），测试集上它也排名第一（${pct(m.test.mean_7d.top10_count_capture)}）。沿用昨天的平均误差最低，说明「估准每格船数」和「找出最值得关注的网格」是两个评价目标。全区峰值只命中 ${m.test.mean_7d.peak_window_hits}/6 天，时段排序可用于筛查。`;
  }

  // ---------- 预测结果 ----------
  function groupForecast() {
    const byHour = {};
    for (const r of bundle.forecast) (byHour[r.hour] = byHour[r.hour] || []).push(r);
    return byHour;
  }

  function renderForecast() {
    const rows = bundle.forecast;
    const byHour = groupForecast();

    const totals = HOUR_STARTS.map((h) => ({
      h,
      label: HOURS[HOUR_STARTS.indexOf(h)],
      total: (byHour[h] || []).reduce((s, r) => s + r.predicted, 0),
    }));
    const peakHour = totals.reduce((a, b) => (a.total >= b.total ? a : b)).h;
    const top5 = [...rows].sort((a, b) => b.predicted - a.predicted).slice(0, 5);
    const top0 = top5[0];

    $("#ais-forecast-desc").innerHTML =
      `以最后一天记录为预测起点，用 7 日均值输出 <strong>${bundle.metrics.forecast_cells} 个网格 × 6 个时段</strong>，共 <strong>${fmt(rows.length)}</strong> 条预测。` +
      `全区预计最活跃的时段是 <strong>${HOURS[HOUR_STARTS.indexOf(peakHour)]}</strong>。单网格最高值出现在 <strong>${HOURS[HOUR_STARTS.indexOf(top0.hour)]}</strong> 的 <strong>(${top0.lat.toFixed(2)}, ${top0.lon.toFixed(2)})</strong>，约 <strong>${fmt(top0.predicted)} 艘</strong>。`;

    $("#ais-forecast-tabs").innerHTML = HOUR_STARTS.map((h) => {
      const label = HOURS[HOUR_STARTS.indexOf(h)];
      return `<button class="ais-tab${h === fcHour ? " active" : ""}" type="button" data-hour="${h}">${label}</button>`;
    }).join("");

    renderForecastMap(byHour);
    renderForecastBars(totals, peakHour);
    renderForecastTop5(top5);
  }

  function forecastColor(v) {
    return FORECAST_PALETTE[FORECAST_BINS.filter((t) => v >= t).length];
  }

  function renderForecastMap(byHour) {
    const list = byHour[fcHour] || [];
    let tiles = "";
    let top = null;
    for (const r of list) {
      if (!top || r.predicted > top.predicted) top = r;
      const pos = tileXY(r.lat, r.lon);
      tiles +=
        `<rect x="${pos.x.toFixed(2)}" y="${pos.y.toFixed(2)}" width="7" height="7" fill="${forecastColor(r.predicted)}">` +
        `<title>中心 ${r.lat.toFixed(2)}, ${r.lon.toFixed(2)}｜预计 ${fmt(r.predicted)} 艘</title></rect>`;
    }
    if (top) {
      const pos = tileXY(top.lat, top.lon);
      tiles +=
        `<circle cx="${(pos.x + 3.5).toFixed(2)}" cy="${(pos.y + 3.5).toFixed(2)}" r="7" ` +
        `fill="none" stroke="#ff964e" stroke-width="2.4" pointer-events="none"/>`;
    }

    $("#ais-forecast-map").innerHTML =
      `<rect x="${SVG.left}" y="${SVG.top}" width="${SVG.w}" height="${SVG.h}" rx="3" fill="${SVG.plot}"/>` +
      `<g class="gridlines">${gridLinesSvg()}</g>${tiles}${axisCaptions()}`;

    $("#ais-forecast-legend").innerHTML = [
      "0–<1", "1–<5", "5–<20", "20–<50", "50–<100", "≥100",
    ].map((label, i) => `<span><i style="--swatch:${FORECAST_PALETTE[i]}"></i>${label}</span>`).join("");

    if (top) {
      $("#ais-forecast-status").textContent =
        `橙色圆圈标出该时段排名第一的网格。中心 (${top.lat.toFixed(2)}, ${top.lon.toFixed(2)})，预计 ${fmt(top.predicted)} 艘。`;
    } else {
      $("#ais-forecast-status").textContent = "该时段无预测记录。";
    }
  }

  function renderForecastBars(totals, peakHour) {
    const max = Math.max(...totals.map((t) => t.total)) || 1;
    $("#ais-forecast-bars").innerHTML = totals.map((t) =>
      `<div class="ais-bar-row${t.h === peakHour ? " is-peak" : ""}">` +
      `<span>${t.label}</span><span class="ais-track"><i style="--width:${(t.total / max * 100).toFixed(1)}%"></i></span>` +
      `<strong>${fmt(t.total)}</strong></div>`
    ).join("");
  }

  function renderForecastTop5(top5) {
    const body = top5.map((r, i) =>
      `<tr><td>${i + 1}</td><th scope="row">${HOURS[HOUR_STARTS.indexOf(r.hour)]}</th>` +
      `<td>${r.lat.toFixed(2)}, ${r.lon.toFixed(2)}</td><td>${fmt(r.predicted)}</td></tr>`
    ).join("");
    $("#ais-forecast-top5").innerHTML =
      `<caption>船数预测最高的 5 个网格时段</caption>` +
      `<thead><tr><th scope="col">排名</th><th scope="col">时段</th><th scope="col">中心（纬度, 经度）</th><th scope="col">预计艘数</th></tr></thead><tbody>${body}</tbody>`;
  }

  // ---------- 事件绑定 ----------
  function bindEvents() {
    const startEl = $("#ais-start"), endEl = $("#ais-end"),
      windowEl = $("#ais-window"), gridEl = $("#ais-grid");

    startEl.addEventListener("change", () => {
      if (Number(startEl.value) > Number(endEl.value)) endEl.value = startEl.value;
      historyState.start = Number(startEl.value);
      historyState.end = Number(endEl.value);
      renderHistory();
    });
    endEl.addEventListener("change", () => {
      if (Number(endEl.value) < Number(startEl.value)) startEl.value = endEl.value;
      historyState.start = Number(startEl.value);
      historyState.end = Number(endEl.value);
      renderHistory();
    });
    windowEl.addEventListener("change", () => {
      historyState.slot = Number(windowEl.value);
      renderHistory();
    });
    gridEl.addEventListener("change", () => {
      historyState.grid = gridEl.value;
      renderHistory();
    });
    $("#ais-reset").addEventListener("click", () => {
      historyState = { start: 0, end: bundle.history.dates.length - 1, slot: -1, grid: "" };
      startEl.value = 0;
      endEl.value = bundle.history.dates.length - 1;
      windowEl.value = -1;
      gridEl.value = "";
      renderHistory();
    });

    $("#ais-map").addEventListener("click", (e) => {
      const tile = e.target.closest("[data-grid]");
      if (!tile) return;
      gridEl.value = gridEl.value === tile.dataset.grid ? "" : tile.dataset.grid;
      historyState.grid = gridEl.value;
      renderHistory();
    });
    $("#ais-hour-bars").addEventListener("click", (e) => {
      const bar = e.target.closest("[data-slot]");
      if (!bar) return;
      windowEl.value = windowEl.value === bar.dataset.slot ? -1 : bar.dataset.slot;
      historyState.slot = Number(windowEl.value);
      renderHistory();
    });
    $("#ais-day-bars").addEventListener("click", (e) => {
      const bar = e.target.closest("[data-day]");
      if (!bar) return;
      historyState.start = Number(bar.dataset.day);
      historyState.end = Number(bar.dataset.day);
      startEl.value = bar.dataset.day;
      endEl.value = bar.dataset.day;
      renderHistory();
    });

    $("#ais-forecast-tabs").addEventListener("click", (e) => {
      const tab = e.target.closest("[data-hour]");
      if (!tab) return;
      fcHour = Number(tab.dataset.hour);
      $$(".ais-tab").forEach((t) => t.classList.toggle("active", t === tab));
      renderForecastMap(groupForecast());
    });
  }

  return { init };
})();
