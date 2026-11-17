/* 前端逻辑：视图切换 + 数据拉取 + 渲染（无框架依赖）。 */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------- 视图切换 ----------
  function showView(name) {
    $$(".view").forEach((v) => v.classList.add("hidden"));
    const target = document.getElementById("view-" + name);
    if (target) target.classList.remove("hidden");
    if (name === "fishing") loadFishingView();
    if (name === "ais" && window.AIS) window.AIS.init();
  }

  function bindNavigation() {
    document.body.addEventListener("click", (e) => {
      const nav = e.target.closest("[data-view]");
      if (nav) showView(nav.dataset.view);
    });
  }

  // ---------- 数据拉取 ----------
  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + " -> " + res.status);
    return res.json();
  }

  async function loadFishingView() {
    try {
      const [summary, videoCases, imageCases] = await Promise.all([
        fetchJSON("/api/summary"),
        fetchJSON("/api/video-cases"),
        fetchJSON("/api/image-cases"),
      ]);
      renderMetrics(summary);
      renderVideoCases(videoCases);
      renderImageCases(imageCases);
    } catch (err) {
      console.error(err);
      $("#video-cases").innerHTML =
        '<p style="color:#ff6b6b">数据加载失败，请确认服务已启动且 output 目录存在。</p>';
    }
  }

  // ---------- 指标卡 ----------
  function renderMetrics(summary) {
    const m = summary.image_metrics || {};
    const pct = (v) => (v == null ? "-" : (v * 100).toFixed(2) + "%");
    const items = [
      { val: pct(m.overall_detection_rate), lbl: "总体检出率" },
      { val: pct(m.bulao_detection_rate), lbl: "捕捞检出率" },
      { val: pct(m.chuidiao_detection_rate), lbl: "垂钓检出率" },
      { val: summary.video_count ?? "-", lbl: "视频案例" },
      { val: summary.evidence_count ?? "-", lbl: "证据蒙太奇" },
      { val: summary.fishing_video_count ?? "-", lbl: "检出钓鱼视频" },
    ];
    $("#metrics").innerHTML = items
      .map(
        (it) =>
          `<div class="metric"><div class="val">${it.val}</div><div class="lbl">${it.lbl}</div></div>`
      )
      .join("");
  }

  // ---------- 视频案例（只展示证据帧及其判定结果） ----------
  function renderVideoCases(cases) {
    if (!cases.length) {
      $("#video-cases").innerHTML = '<p style="color:#93a3ba">暂无视频案例。</p>';
      return;
    }
    $("#video-cases").innerHTML = cases.map(videoCaseHTML).join("");
  }

  function videoCaseHTML(c) {
    const badge = c.fishing_detected
      ? `<span class="badge yes">检出钓鱼行为 · ${c.evidence_count} 条证据</span>`
      : '<span class="badge no">未检出</span>';
    const d = c.detection || {};

    const stats = `
      <div class="video-stats">
        <span class="chip">总帧数 ${c.total_frames ?? 0}</span>
        <span class="chip">检测到人 ${d.person_frames ?? 0} 帧</span>
        <span class="chip">检测到船 ${d.boat_frames ?? 0} 帧</span>
        <span class="chip">人员框 ${d.person_boxes ?? 0} 个</span>
        <span class="chip">船只框 ${d.boat_boxes ?? 0} 个</span>
      </div>`;

    const list = c.evidence_cases.length
      ? c.evidence_cases.map(evidenceCaseHTML).join("")
      : '<p class="empty-hint">该视频暂无证据帧。</p>';

    return `
      <div class="video-case">
        <div class="video-case-head">
          <h4>${escapeHTML(c.video_name)}</h4>
          ${badge}
        </div>
        <div class="video-case-body">
          <div class="video-panel">
            <video src="${c.video_url}" controls preload="metadata"></video>
            ${stats}
          </div>
          <div class="evidence-panel">
            <p class="panel-title">证据帧与判定</p>
            <div class="evidence-list">${list}</div>
          </div>
        </div>
      </div>`;
  }

  function evidenceCaseHTML(e) {
    const badgeCls = e.answer === "Yes" ? "badge yes" : "badge no";
    return `
      <div class="evidence-case">
        <img class="evidence-img" src="${e.url}" alt="证据 ${e.seq}" data-lightbox="${e.url}">
        <div class="evidence-info">
          <div class="evidence-meta">
            <span class="badge ${badgeCls}">${e.answer}</span>
            <span class="evidence-ts">🕒 ${e.timestamp || "-"}</span>
            <span class="evidence-seq">证据 #${e.seq}</span>
          </div>
          <p class="evidence-reason">${escapeHTML(e.reasoning || "")}</p>
        </div>
      </div>`;
  }

  // ---------- 图像案例 ----------
  function renderImageCases({ cases = [], error_cases = [] }) {
    const html = [];
    if (cases.length) {
      html.push(
        `<h4 class="image-sub-title">正确识别样例 <span class="count">${cases.length}</span></h4>
         <div class="image-cases">${cases.map(imageCaseHTML).join("")}</div>`
      );
    }
    if (error_cases.length) {
      html.push(
        `<h4 class="image-sub-title">典型误判样例 <span class="count">${error_cases.length}</span></h4>
         <div class="image-cases">${error_cases.map(imageCaseHTML).join("")}</div>`
      );
    }
    $("#image-cases").innerHTML = html.join("");
  }

  function imageCaseHTML(c) {
    const badgeCls = c.prediction === "Yes" ? "badge yes" : "badge no";
    const stateCls = c.correct ? "correct" : "error";
    return `
      <div class="image-case ${stateCls}">
        <img src="${c.image_url}" alt="${c.category_label}案例" data-lightbox="${c.image_url}">
        <div class="ic-body">
          <div class="ic-top">
            <span class="ic-cat">${c.category_label}</span>
            <span class="badge ${badgeCls}">${c.prediction}</span>
          </div>
          <p class="ic-reason">${escapeHTML(c.reasoning || "")}</p>
        </div>
      </div>`;
  }

  // ---------- 灯箱 ----------
  function bindLightbox() {
    const box = $("#lightbox");
    const img = $("#lightbox-img");
    document.body.addEventListener("click", (e) => {
      const trigger = e.target.closest("[data-lightbox]");
      if (trigger) {
        img.src = trigger.dataset.lightbox;
        box.classList.remove("hidden");
      }
    });
    $("#lightbox-close").addEventListener("click", () => box.classList.add("hidden"));
    box.addEventListener("click", (e) => {
      if (e.target === box) box.classList.add("hidden");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") box.classList.add("hidden");
    });
  }

  // ---------- 工具 ----------
  function escapeHTML(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ---------- 启动 ----------
  document.addEventListener("DOMContentLoaded", () => {
    bindNavigation();
    bindLightbox();
    showView("home");
  });
})();
