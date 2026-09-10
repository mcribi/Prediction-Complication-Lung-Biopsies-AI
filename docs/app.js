(() => {
  "use strict";
  const data = window.TFM_DATA;
  const format = value => typeof value === "number" ? value.toFixed(4).replace(".", ",") : value;
  const short = value => typeof value === "number" ? value.toFixed(3).replace(".", ",") : value;

  const overviewChart = document.querySelector("#overviewChart");
  const chartInsight = document.querySelector("#chartInsight");
  function renderOverview(metric = "micro") {
    const values = [...data.overview].sort((a, b) => b[metric] - a[metric]);
    const colors = { green: "var(--green)", blue: "var(--blue)", violet: "var(--violet)", coral: "var(--coral)" };
    overviewChart.innerHTML = values.map(item => `
      <div class="bar-row">
        <span>${item.name}</span>
        <div class="bar-track"><div class="bar-fill" style="--bar-color:${colors[item.color]};width:${item[metric] * 100}%"></div></div>
        <span class="bar-value">${format(item[metric])}</span>
      </div>`).join("");
    const top = values[0];
    const bottom = values[values.length - 1];
    const gap = top[metric] - bottom[metric];
    chartInsight.textContent = `${top.name} encabeza este comparador, con una diferencia de ${format(gap)} respecto al último enfoque mostrado. Las diferencias son reducidas y deben interpretarse junto con las métricas por complicación.`;
  }
  document.querySelectorAll("[data-chart-metric]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-chart-metric]").forEach(b => b.classList.toggle("active", b === button));
    renderOverview(button.dataset.chartMetric);
  }));
  renderOverview();

  let radiomicsRows = [...data.radiomics];
  let radiomicsSort = { key: "micro", direction: -1 };
  const radiomicsBody = document.querySelector("#radiomicsTable tbody");
  function renderRadiomics() {
    const query = document.querySelector("#radiomicsSearch").value.trim().toLowerCase();
    const rows = radiomicsRows.filter(row => Object.values(row).join(" ").toLowerCase().includes(query));
    rows.sort((a, b) => {
      const x = a[radiomicsSort.key], y = b[radiomicsSort.key];
      return (typeof x === "number" ? x - y : String(x).localeCompare(String(y), "es")) * radiomicsSort.direction;
    });
    radiomicsBody.innerHTML = rows.map(row => `<tr class="${row.id === "R07" ? "best" : ""}">
      <td>${row.id}</td><td>${row.model}</td><td>${row.features}</td><td>${row.data}</td><td>${row.masks}</td>
      <td class="numeric">${format(row.micro)}</td><td class="numeric">${format(row.macro)}</td><td class="numeric">${format(row.subset)}</td>
    </tr>`).join("");
  }
  document.querySelector("#radiomicsSearch").addEventListener("input", renderRadiomics);
  document.querySelectorAll("#radiomicsTable th").forEach(th => th.addEventListener("click", () => {
    const key = th.dataset.sort;
    radiomicsSort = { key, direction: radiomicsSort.key === key ? -radiomicsSort.direction : (typeof data.radiomics[0][key] === "number" ? -1 : 1) };
    renderRadiomics();
  }));
  renderRadiomics();

  let radiomicsClassTarget = "hemorrhage";
  let radiomicsClassSort = { key: "f1", direction: -1 };
  const classMetricIndex = { precision: 0, sensitivity: 1, f1: 2, specificity: 3 };
  const radiomicsClassBody = document.querySelector("#radiomicsClassTable tbody");
  const pairFormat = pair => `${format(pair[0])} ± ${format(pair[1])}`;
  function classSortValue(row, key) {
    if (key in classMetricIndex) return row.metrics[classMetricIndex[key]][0];
    return row[key];
  }
  function renderRadiomicsClass() {
    const rows = [...data.radiomicsByClass[radiomicsClassTarget]];
    rows.sort((a, b) => {
      const x = classSortValue(a, radiomicsClassSort.key);
      const y = classSortValue(b, radiomicsClassSort.key);
      return (typeof x === "number" ? x - y : String(x).localeCompare(String(y), "es")) * radiomicsClassSort.direction;
    });
    const bestF1 = Math.max(...rows.map(row => row.metrics[2][0]));
    radiomicsClassBody.innerHTML = rows.map(row => `<tr class="${row.metrics[2][0] === bestF1 ? "best" : ""}">
      <td>${row.id}</td><td>${row.model}</td><td>${row.data}</td>
      ${row.metrics.map(metric => `<td class="numeric">${pairFormat(metric)}</td>`).join("")}
    </tr>`).join("");
  }
  document.querySelectorAll("[data-radiomics-target]").forEach(button => button.addEventListener("click", () => {
    radiomicsClassTarget = button.dataset.radiomicsTarget;
    document.querySelectorAll("[data-radiomics-target]").forEach(b => b.classList.toggle("active", b === button));
    renderRadiomicsClass();
  }));
  document.querySelectorAll("#radiomicsClassTable th").forEach(th => th.addEventListener("click", () => {
    const key = th.dataset.classSort;
    const sample = classSortValue(data.radiomicsByClass[radiomicsClassTarget][0], key);
    radiomicsClassSort = { key, direction: radiomicsClassSort.key === key ? -radiomicsClassSort.direction : (typeof sample === "number" ? -1 : 1) };
    renderRadiomicsClass();
  }));
  renderRadiomicsClass();

  const dlTable = document.querySelector("#dlTable");
  function renderDL(key) {
    const block = data.deepLearning[key];
    document.querySelector("#dlTableTitle").textContent = block.title;
    document.querySelector("#dlNote").textContent = block.note;
    dlTable.querySelector("thead").innerHTML = `<tr>${block.columns.map(col => `<th>${col}</th>`).join("")}</tr>`;
    dlTable.querySelector("tbody").innerHTML = block.rows.map((row, index) => `<tr class="${index === 0 ? "best" : ""}">${row.map(value => `<td class="${typeof value === "number" ? "numeric" : ""}">${format(value)}</td>`).join("")}</tr>`).join("");
  }
  document.querySelectorAll("[data-dl-tab]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-dl-tab]").forEach(b => { b.classList.toggle("active", b === button); b.setAttribute("aria-selected", b === button ? "true" : "false"); });
    renderDL(button.dataset.dlTab);
  }));
  renderDL("image");

  let puTarget = "hemorrhage";
  let puSort = { index: 2, direction: -1 };
  const puBody = document.querySelector("#puTable tbody");
  function renderPU() {
    const rows = [...data.pu[puTarget]].sort((a, b) => {
      const x = a[puSort.index], y = b[puSort.index];
      return (typeof x === "number" ? x - y : String(x).localeCompare(String(y), "es")) * puSort.direction;
    });
    const bestAP = Math.max(...rows.map(r => r[2]));
    puBody.innerHTML = rows.map(row => `<tr class="${row[2] === bestAP ? "best" : ""}">${row.map(value => `<td class="${typeof value === "number" ? "numeric" : ""}">${short(value)}</td>`).join("")}</tr>`).join("");
  }
  document.querySelectorAll("[data-pu-target]").forEach(button => button.addEventListener("click", () => {
    puTarget = button.dataset.puTarget;
    document.querySelectorAll("[data-pu-target]").forEach(b => b.classList.toggle("active", b === button));
    renderPU();
  }));
  document.querySelectorAll("#puTable th").forEach((th, index) => th.addEventListener("click", () => {
    puSort = { index, direction: puSort.index === index ? -puSort.direction : (index > 1 ? -1 : 1) };
    renderPU();
  }));
  renderPU();

  const pu3dCards = document.querySelector("#pu3dCards");
  const best3d = data.pu3d.filter(row => row.method === "Supervisado");
  pu3dCards.innerHTML = best3d.map(row => `<div class="mini-card"><strong>${row.target}</strong><span>Supervisado · AP ${short(row.ap)} · F1 ${short(row.f1)}</span></div>`).join("");

  const ruleGrid = document.querySelector("#ruleGrid");
  function renderRules(target = "all") {
    const rules = data.rules.filter(rule => target === "all" || rule.target === target);
    ruleGrid.innerHTML = rules.map(rule => `<article class="rule-card">
      <div class="rule-top"><span>${rule.set}</span><span class="consensus">${rule.consensus}</span></div>
      <h3>${rule.rule}</h3>
      <div class="rule-stats"><span>Objetivo<br><strong>${rule.target}</strong></span><span>Δ prevalencia<br><strong>+${short(rule.delta)}</strong></span><span>Config.<br><strong>${rule.configs}</strong></span></div>
    </article>`).join("");
  }
  document.querySelectorAll("[data-rule-target]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-rule-target]").forEach(b => b.classList.toggle("active", b === button));
    renderRules(button.dataset.ruleTarget);
  }));
  renderRules();

  document.querySelector("#subgroupChart").innerHTML = data.subgroupModels.map(row => {
    const delta = row.specialized - row.general;
    return `<div class="pair-row"><div class="pair-label"><strong>${row.subgroup}</strong><span>${row.model}</span></div>
      <div class="pair-bars"><div class="pair-bar general"><i style="width:${row.general * 100}%"></i></div><div class="pair-bar special"><i style="width:${row.specialized * 100}%"></i></div></div>
      <span class="pair-delta">+${short(delta)}</span></div>`;
  }).join("");

  const lightbox = document.querySelector("#lightbox");
  document.querySelectorAll("[data-lightbox]").forEach(button => button.addEventListener("click", () => {
    document.querySelector("#lightboxImage").src = button.dataset.lightbox;
    document.querySelector("#lightboxImage").alt = button.dataset.caption;
    document.querySelector("#lightboxCaption").textContent = button.dataset.caption;
    lightbox.showModal();
  }));
  document.querySelector("#closeLightbox").addEventListener("click", () => lightbox.close());
  lightbox.addEventListener("click", event => { if (event.target === lightbox) lightbox.close(); });

  const menuButton = document.querySelector("#menuButton");
  const sidebar = document.querySelector("#sidebar");
  menuButton.addEventListener("click", () => {
    const open = sidebar.classList.toggle("open");
    menuButton.setAttribute("aria-expanded", String(open));
  });
  document.querySelectorAll("#mainNav a").forEach(link => link.addEventListener("click", () => { sidebar.classList.remove("open"); menuButton.setAttribute("aria-expanded", "false"); }));

  const themeToggle = document.querySelector("#themeToggle");
  const savedTheme = localStorage.getItem("tfm-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  themeToggle.addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("tfm-theme", theme);
  });

  const sections = [...document.querySelectorAll("main section[id]")];
  const navLinks = [...document.querySelectorAll("#mainNav a")];
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach(link => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}` || (visible.target.id === "inicio" && link.getAttribute("href") === "#resumen")));
  }, { rootMargin: "-20% 0px -65%", threshold: [0.05, 0.25] });
  sections.forEach(section => observer.observe(section));
})();
