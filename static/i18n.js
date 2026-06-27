/* SafetyCommander i18n — English default, switch to 中文 / Español.
   Static text uses data-i18n="key"; JS-built text uses window.T(key).
   Choice persists in localStorage; pages re-render via window.onLangChange. */
(function () {
  const DICT = {
    en: {
      choose_view: "Autonomous safety officer · choose your view",
      worker_title: "Worker · Floor", manager_title: "Manager · Operations",
      worker_desc: "Floor status, alerts to act on, your corrective tasks, today's safety focus.",
      manager_desc: "Monthly KPIs, violation trends, open corrective backlog, AI weekly plan, live monitor.",
      footer: "VLM reasons risk from the policy · perception measures · RAG cites OSHA · actions auto-fire",
      switch_view: "switch view", live_monitor: "▶ Live monitor",
      status_label: "CURRENT FLOOR STATUS", clear: "CLEAR", all_clear: "✓ All clear",
      action_prefix: "Action:",
      alerts_h: "⚠ Alerts to act on", tasks_h: "📋 Your corrective tasks", focus_h: "📅 Today's focus",
      no_alerts: "No active high-risk alerts. ✅", no_tasks: "No open corrective actions assigned. ✅",
      no_plan: "No plan yet (run planner.py).",
      kpi_violations: "Violations", kpi_nearmiss: "Near-misses", kpi_critical: "Critical",
      kpi_backlog: "Open CA backlog", kpi_frames: "Frames observed",
      panel_hazards: "Top hazards", panel_zone: "By zone",
      panel_backlog: "Open corrective-action backlog", panel_plan: "AI weekly plan",
      none_open: "None open ✅", loading: "Loading…",
      monitor_sub: "Autonomous safety officer · Qwen3-VL reasons risk from the written policy",
      monitor_curframe: "Current frame", monitor_verdict: "VLM safety verdict", monitor_feed: "Live event feed",
    },
    zh: {
      choose_view: "自主安全员 · 选择视图",
      worker_title: "员工端 · 现场", manager_title: "管理端 · 运营",
      worker_desc: "现场状态、需注意的告警、你的整改任务、今日安全重点。",
      manager_desc: "月度 KPI、违规趋势、未关闭整改、AI 周计划、实时监控。",
      footer: "VLM 依规程推理风险 · 感知测量 · RAG 引用 OSHA · 动作自动触发",
      switch_view: "切换视图", live_monitor: "▶ 实时监控大屏",
      status_label: "当前现场状态", clear: "正常", all_clear: "✓ 一切正常",
      action_prefix: "处理：",
      alerts_h: "⚠ 需要注意的告警", tasks_h: "📋 我的整改任务", focus_h: "📅 今日安全重点",
      no_alerts: "暂无高风险告警 ✅", no_tasks: "暂无分配给你的整改任务 ✅",
      no_plan: "暂无计划（运行 planner.py）。",
      kpi_violations: "违规", kpi_nearmiss: "近距/未遂", kpi_critical: "严重",
      kpi_backlog: "未关闭整改", kpi_frames: "已观测帧",
      panel_hazards: "主要危险", panel_zone: "按区域",
      panel_backlog: "未关闭整改 backlog", panel_plan: "AI 周计划",
      none_open: "无未关闭 ✅", loading: "加载中…",
      monitor_sub: "自主安全员 · Qwen3-VL 依书面规程推理风险",
      monitor_curframe: "当前帧", monitor_verdict: "VLM 安全判定", monitor_feed: "实时事件流",
    },
    es: {
      choose_view: "Oficial de seguridad autónomo · elija su vista",
      worker_title: "Operario · Planta", manager_title: "Gerente · Operaciones",
      worker_desc: "Estado de planta, alertas a atender, sus acciones correctivas, enfoque de hoy.",
      manager_desc: "KPIs mensuales, tendencias, correctivas abiertas, plan semanal con IA, monitor en vivo.",
      footer: "La IA razona el riesgo según la política · la percepción mide · RAG cita OSHA · acciones automáticas",
      switch_view: "cambiar vista", live_monitor: "▶ Monitor en vivo",
      status_label: "ESTADO ACTUAL DE PLANTA", clear: "DESPEJADO", all_clear: "✓ Todo en orden",
      action_prefix: "Acción:",
      alerts_h: "⚠ Alertas a atender", tasks_h: "📋 Sus tareas correctivas", focus_h: "📅 Enfoque de hoy",
      no_alerts: "Sin alertas de alto riesgo. ✅", no_tasks: "Sin acciones correctivas asignadas. ✅",
      no_plan: "Sin plan aún (ejecute planner.py).",
      kpi_violations: "Infracciones", kpi_nearmiss: "Cuasi-accidentes", kpi_critical: "Críticos",
      kpi_backlog: "Correctivas abiertas", kpi_frames: "Cuadros observados",
      panel_hazards: "Principales peligros", panel_zone: "Por zona",
      panel_backlog: "Correctivas pendientes", panel_plan: "Plan semanal con IA",
      none_open: "Ninguna abierta ✅", loading: "Cargando…",
      monitor_sub: "Oficial de seguridad autónomo · Qwen3-VL razona el riesgo según la política escrita",
      monitor_curframe: "Cuadro actual", monitor_verdict: "Veredicto de seguridad (VLM)", monitor_feed: "Eventos en vivo",
    },
  };
  let cur = localStorage.getItem("sc_lang") || "en";
  window.T = (k) => (DICT[cur] && DICT[cur][k]) || DICT.en[k] || k;
  window.SCLang = () => cur;

  function apply() {
    document.documentElement.lang = cur;
    document.querySelectorAll("[data-i18n]").forEach((e) => (e.textContent = window.T(e.dataset.i18n)));
    document.querySelectorAll("#langsw [data-lang]").forEach((b) => b.classList.toggle("on", b.dataset.lang === cur));
    if (window.onLangChange) window.onLangChange();
  }
  window.applyLang = (l) => { cur = l; localStorage.setItem("sc_lang", l); apply(); };

  function init() {
    const st = document.createElement("style");
    st.textContent =
      "#langsw{display:inline-flex;gap:4px}#langsw button{background:#1b2230;color:#8a94a6;" +
      "border:1px solid #252d3b;border-radius:6px;padding:3px 9px;font-size:12px;cursor:pointer}" +
      "#langsw button.on{background:#7c5cff;color:#fff;border-color:#7c5cff}";
    document.head.appendChild(st);
    const slot = document.getElementById("langsw");
    if (slot)
      slot.innerHTML =
        '<button data-lang="en" onclick="applyLang(\'en\')">EN</button>' +
        '<button data-lang="zh" onclick="applyLang(\'zh\')">中文</button>' +
        '<button data-lang="es" onclick="applyLang(\'es\')">ES</button>';
    apply();
  }
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
