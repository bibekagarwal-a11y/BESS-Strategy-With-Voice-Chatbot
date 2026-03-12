let data = [];

const RULE_LABELS = {
  DA_IDA1: "Day Ahead ↔ IDA1",
  DA_IDA2: "Day Ahead ↔ IDA2",
  DA_IDA3: "Day Ahead ↔ IDA3",
  DA_VWAP: "Day Ahead ↔ Intraday VWAP",
  IDA1_IDA2: "IDA1 ↔ IDA2",
  IDA1_IDA3: "IDA1 ↔ IDA3",
  IDA1_VWAP: "IDA1 ↔ Intraday VWAP",
  IDA2_IDA3: "IDA2 ↔ IDA3",
  IDA2_VWAP: "IDA2 ↔ Intraday VWAP",
  IDA3_VWAP: "IDA3 ↔ Intraday VWAP"
};

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const el = byId(id);
  if (el) el.innerText = value;
}

function setHTML(id, value) {
  const el = byId(id);
  if (el) el.innerHTML = value;
}

function unique(arr) {
  return [...new Set(arr)];
}

function parseDateContractToIndex(row) {
  const datePart = String(row.date ?? "");
  const contractPart = Number(row.contract_sort ?? 0);
  return { datePart, contractPart };
}

function compareRowsChronologically(a, b) {
  const aKey = parseDateContractToIndex(a);
  const bKey = parseDateContractToIndex(b);
  const dateCompare = aKey.datePart.localeCompare(bKey.datePart);
  if (dateCompare !== 0) return dateCompare;
  return aKey.contractPart - bKey.contractPart;
}

function getActiveDateField() {
  const mode = byId("dateMode")?.value || "market";
  return mode === "run" ? "run_date" : "date";
}

async function loadData() {
  try {
    const res = await fetch("./data/contract_profits.json", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Failed to load data: ${res.status} ${res.statusText}`);
    }

    data = await res.json();

    if (!Array.isArray(data) || data.length === 0) {
      throw new Error("contract_profits.json loaded, but it is empty.");
    }

    populateSelectors();
  } catch (err) {
    console.error(err);
    showError(err.message);
  }
}

function showError(message) {
  setHTML(
    "table",
    `<div style="padding:16px;border:1px solid #fda29b;background:#fff1f3;border-radius:12px;color:#b42318;">
      <strong>Data loading error</strong><br />
      ${message}
    </div>`
  );
}

function showNoDataMessage(message = "No data for selected filters.") {
  setText("profit", "-");
  setText("contractCount", "0");
  setText("avgProfit", "-");
  setText("winRate", "-");
  setText("bestContract", "-");
  setText("worstContract", "-");
  setText("bessStrategy", message);
  setText("bessMultiCycle", message);
  setHTML("topContracts", "<div>No data</div>");
  setHTML("bottomContracts", "<div>No data</div>");
  setHTML(
    "table",
    `<div style="padding:16px;border:1px solid #fda29b;background:#fff1f3;border-radius:12px;color:#b42318;">
      <strong>No data for selected filters</strong><br />
      ${message}
    </div>`
  );

  renderEmptyPlot("cumulativeCurve");
  renderEmptyPlot("contractBar");
  renderEmptyPlot("heatmap");
  renderEmptyPlot("histogram");
}

function renderEmptyPlot(id) {
  const el = byId(id);
  if (!el || typeof Plotly === "undefined") return;

  Plotly.newPlot(
    id,
    [],
    {
      annotations: [{ text: "No data", showarrow: false }],
      margin: { l: 40, r: 20, t: 20, b: 40 },
      paper_bgcolor: "white",
      plot_bgcolor: "white"
    },
    { responsive: true, displayModeBar: false }
  );
}

function setOptions(id, values, labelMap = null) {
  const el = byId(id);
  if (!el) return;
  el.innerHTML = "";

  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.text = labelMap && labelMap[v] ? labelMap[v] : v;
    el.appendChild(opt);
  });
}

function selectAllOptions(id) {
  const el = byId(id);
  if (!el) return;
  [...el.options].forEach(o => {
    o.selected = true;
  });
}

function clearAllOptions(id) {
  const el = byId(id);
  if (!el) return;
  [...el.options].forEach(o => {
    o.selected = false;
  });
}

function getSelectedValues(id) {
  const el = byId(id);
  if (!el) return [];
  return [...el.selectedOptions].map(x => x.value);
}

function setActivePreset(buttonId) {
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.classList.remove("active-preset");
  });

  if (buttonId) {
    const btn = byId(buttonId);
    if (btn) btn.classList.add("active-preset");
  }
}

function parseContractStartMinutes(contractLabel) {
  if (!contractLabel || !contractLabel.includes("-")) return null;
  const start = contractLabel.split("-")[0];
  const [hh, mm] = start.split(":").map(Number);
  if (!Number.isFinite(hh) || !Number.isFinite(mm)) return null;
  return hh * 60 + mm;
}

function applyContractPreset(presetName) {
  const contractsEl = byId("contracts");
  if (!contractsEl) return;

  const options = [...contractsEl.options];

  options.forEach(opt => {
    const mins = parseContractStartMinutes(opt.value);
    if (mins === null) {
      opt.selected = false;
      return;
    }

    let selected = false;
    if (presetName === "base") selected = true;
    else if (presetName === "peak") selected = mins >= 8 * 60 && mins < 20 * 60;
    else if (presetName === "offpeak") selected = mins < 8 * 60 || mins >= 20 * 60;
    else if (presetName === "morning") selected = mins >= 6 * 60 && mins < 12 * 60;
    else if (presetName === "evening") selected = mins >= 17 * 60 && mins < 23 * 60;

    opt.selected = selected;
  });

  const presetMap = {
    base: "presetBaseBtn",
    peak: "presetPeakBtn",
    offpeak: "presetOffPeakBtn",
    morning: "presetMorningBtn",
    evening: "presetEveningBtn"
  };

  setActivePreset(presetMap[presetName] || null);
  render();
}

function updateDateInputs() {
  const dateField = getActiveDateField();
  const dates = unique(data.map(x => x[dateField])).filter(Boolean).sort();

  if (!dates.length) {
    if (dateField === "run_date") {
      throw new Error("No run_date values found in data. Add run_date to contract_profits.json generation.");
    }
    throw new Error(`No ${dateField} values found in data.`);
  }

  if (byId("startDate")) byId("startDate").value = dates[0];
  if (byId("endDate")) byId("endDate").value = dates[dates.length - 1];
}

function populateSelectors() {
  const areas = unique(data.map(x => x.area)).filter(Boolean).sort();
  const rules = unique(data.map(x => x.rule)).filter(Boolean).sort();

  if (!areas.length) throw new Error("No areas found in data.");
  if (!rules.length) throw new Error("No strategies found in data.");

  setOptions("area", areas);
  setOptions("rule", rules, RULE_LABELS);

  updateDateInputs();
  updateContracts();
}

function updateContracts() {
  const area = byId("area")?.value || "";
  const rule = byId("rule")?.value || "";
  const startDate = byId("startDate")?.value || "";
  const endDate = byId("endDate")?.value || "";
  const dateField = getActiveDateField();

  const filtered = data
    .filter(d => {
      if (area && d.area !== area) return false;
      if (rule && d.rule !== rule) return false;
      if (startDate && String(d[dateField] ?? "") < startDate) return false;
      if (endDate && String(d[dateField] ?? "") > endDate) return false;
      return true;
    })
    .sort(compareRowsChronologically);

  const contracts = unique(filtered.map(x => x.contract)).filter(Boolean);
  setOptions("contracts", contracts);

  if (contracts.length) {
    selectAllOptions("contracts");
    setActivePreset("presetBaseBtn");
  } else {
    setActivePreset(null);
  }

  render();
}

function getFilteredRows() {
  const area = byId("area")?.value || "";
  const rule = byId("rule")?.value || "";
  const direction = byId("direction")?.value || "forward";
  const startDate = byId("startDate")?.value || "";
  const endDate = byId("endDate")?.value || "";
  const selectedContracts = getSelectedValues("contracts");
  const dateField = getActiveDateField();

  let filtered = data
    .filter(d => {
      if (area && d.area !== area) return false;
      if (rule && d.rule !== rule) return false;
      if (startDate && String(d[dateField] ?? "") < startDate) return false;
      if (endDate && String(d[dateField] ?? "") > endDate) return false;
      if (selectedContracts.length && !selectedContracts.includes(d.contract)) return false;
      return true;
    })
    .sort(compareRowsChronologically);

  if (direction === "reverse") {
    filtered = filtered.map(d => ({
      ...d,
      buy_price: Number(d.sell_price),
      sell_price: Number(d.buy_price),
      profit: -Number(d.profit)
    }));
  } else {
    filtered = filtered.map(d => ({
      ...d,
      buy_price: Number(d.buy_price),
      sell_price: Number(d.sell_price),
      profit: Number(d.profit)
    }));
  }

  return filtered;
}

function renderMetricCards(filtered) {
  const profits = filtered.map(x => Number(x.profit));
  const total = profits.reduce((a, b) => a + b, 0);
  const avg = profits.length ? total / profits.length : 0;
  const winRate = profits.length ? (profits.filter(x => x > 0).length / profits.length) * 100 : 0;

  const best = filtered.length ? [...filtered].sort((a, b) => b.profit - a.profit)[0] : null;
  const worst = filtered.length ? [...filtered].sort((a, b) => a.profit - b.profit)[0] : null;

  setText("profit", `${total.toFixed(2)} €/MWh`);
  setText("contractCount", `${filtered.length}`);
  setText("avgProfit", `${avg.toFixed(2)} €/MWh`);
  setText("winRate", `${winRate.toFixed(1)}%`);

  setHTML("bestContract", best ? `${best.date}<br>${best.contract}<br>${Number(best.profit).toFixed(2)} €/MWh` : "-");
  setHTML("worstContract", worst ? `${worst.date}<br>${worst.contract}<br>${Number(worst.profit).toFixed(2)} €/MWh` : "-");
}

function renderBessStrategy(filtered) {
  const bessEl = byId("bessStrategy");
  if (!bessEl) return;

  if (!filtered.length) {
    bessEl.innerHTML = "-";
    return;
  }

  const ordered = [...filtered].sort(compareRowsChronologically);
  let bestSpread = -Infinity;
  let bestChargeRow = null;
  let bestDischargeRow = null;
  let minBuySoFar = null;

  for (const row of ordered) {
    const buyPrice = Number(row.buy_price);
    const sellPrice = Number(row.sell_price);
    if (!Number.isFinite(buyPrice) || !Number.isFinite(sellPrice)) continue;

    if (!minBuySoFar || buyPrice < Number(minBuySoFar.buy_price)) {
      minBuySoFar = row;
    }

    const spread = sellPrice - Number(minBuySoFar.buy_price);
    if (spread > bestSpread && compareRowsChronologically(minBuySoFar, row) <= 0) {
      bestSpread = spread;
      bestChargeRow = minBuySoFar;
      bestDischargeRow = row;
    }
  }

  if (!bestChargeRow || !bestDischargeRow || !Number.isFinite(bestSpread)) {
    bessEl.innerHTML = "No valid BESS cycle found for the current selection.";
    return;
  }

  bessEl.innerHTML = `
    <strong>Charge:</strong> ${bestChargeRow.date} | ${bestChargeRow.contract} at ${Number(bestChargeRow.buy_price).toFixed(2)} €/MWh
    <br>
    <strong>Discharge:</strong> ${bestDischargeRow.date} | ${bestDischargeRow.contract} at ${Number(bestDischargeRow.sell_price).toFixed(2)} €/MWh
    <br>
    <strong>Single-cycle spread:</strong>
    <span class="${bestSpread >= 0 ? "positive-text" : "negative-text"}">${bestSpread.toFixed(2)} €/MWh</span>
  `;
}

function computeQuarterHours(contractLabel) {
  if (!contractLabel || !contractLabel.includes("-")) return 0.25;

  const [startStr, endStr] = contractLabel.split("-");
  const [sh, sm] = startStr.split(":").map(Number);
  const [eh, em] = endStr.split(":").map(Number);

  if (![sh, sm, eh, em].every(Number.isFinite)) return 0.25;

  let startMins = sh * 60 + sm;
  let endMins = eh * 60 + em;
  if (endMins < startMins) endMins += 24 * 60;
  return (endMins - startMins) / 60;
}

function renderMultiCycleBess(filtered) {
  const el = byId("bessMultiCycle");
  if (!el) return;

  if (!filtered.length) {
    el.innerHTML = "-";
    return;
  }

  const capacityMWh = Number(byId("bessCapacity")?.value || 1);
  const powerMW = Number(byId("bessPower")?.value || 1);
  const efficiency = Number(byId("bessEfficiency")?.value || 0.9);

  if (!Number.isFinite(capacityMWh) || !Number.isFinite(powerMW) || !Number.isFinite(efficiency) || capacityMWh <= 0 || powerMW <= 0 || efficiency <= 0 || efficiency > 1) {
    el.innerHTML = "Invalid BESS settings.";
    return;
  }

  const ordered = [...filtered].sort(compareRowsChronologically);
  let soc = 0;
  let totalPnL = 0;
  let chargeActions = 0;
  let dischargeActions = 0;
  let throughputMWh = 0;

  const avgFutureSell = ordered.map((_, i) => {
    const future = ordered.slice(i + 1).map(r => Number(r.sell_price)).filter(Number.isFinite);
    if (!future.length) return null;
    return future.reduce((a, b) => a + b, 0) / future.length;
  });

  ordered.forEach((row, i) => {
    const durationH = computeQuarterHours(row.contract);
    const maxEnergyThisStep = Math.min(powerMW * durationH, capacityMWh);

    const buyPrice = Number(row.buy_price);
    const sellPrice = Number(row.sell_price);
    const futureAvgSell = avgFutureSell[i];

    const chargeThreshold = futureAvgSell !== null ? futureAvgSell * efficiency : null;
    const shouldCharge = futureAvgSell !== null && soc < capacityMWh && buyPrice < chargeThreshold;
    const shouldDischarge = soc > 0 && (futureAvgSell === null || sellPrice >= futureAvgSell || i >= ordered.length - 4);

    if (shouldCharge) {
      const availableRoom = capacityMWh - soc;
      const chargeMWh = Math.min(maxEnergyThisStep, availableRoom);
      if (chargeMWh > 0) {
        soc += chargeMWh;
        totalPnL -= chargeMWh * buyPrice;
        throughputMWh += chargeMWh;
        chargeActions += 1;
      }
    } else if (shouldDischarge) {
      const dischargeRawMWh = Math.min(maxEnergyThisStep, soc);
      if (dischargeRawMWh > 0) {
        const deliveredMWh = dischargeRawMWh * efficiency;
        soc -= dischargeRawMWh;
        totalPnL += deliveredMWh * sellPrice;
        throughputMWh += dischargeRawMWh;
        dischargeActions += 1;
      }
    }
  });

  el.innerHTML = `
    <strong>Estimated multi-cycle P&amp;L:</strong>
    <span class="${totalPnL >= 0 ? "positive-text" : "negative-text"}">${totalPnL.toFixed(2)} €</span>
    <br>
    <strong>Charge actions:</strong> ${chargeActions}
    <br>
    <strong>Discharge actions:</strong> ${dischargeActions}
    <br>
    <strong>Total throughput:</strong> ${throughputMWh.toFixed(2)} MWh
    <br>
    <strong>Ending state of charge:</strong> ${soc.toFixed(2)} MWh
  `;
}

function renderHistogram(filtered) {
  if (typeof Plotly === "undefined") return;
  const profits = filtered.map(x => Number(x.profit));

  Plotly.newPlot("histogram", [{
    x: profits,
    type: "histogram",
    marker: { color: "#2563eb" },
    hovertemplate: "Profit: %{x:.2f} €/MWh<br>Count: %{y}<extra></extra>"
  }], {
    margin: { l: 60, r: 20, t: 20, b: 60 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { title: "Profit per row (€/MWh)", gridcolor: "#eaecf0" },
    yaxis: { title: "Count", gridcolor: "#eaecf0" }
  }, {
    responsive: true,
    displayModeBar: false
  });
}

function renderContractBar(filtered) {
  if (typeof Plotly === "undefined") return;
  const labels = filtered.map(x => `${x.date} | ${x.contract}`);
  const profits = filtered.map(x => Number(x.profit));
  const colors = profits.map(v => (v >= 0 ? "#16a34a" : "#dc2626"));

  Plotly.newPlot("contractBar", [{
    x: labels,
    y: profits,
    type: "bar",
    marker: { color: colors },
    hovertemplate: "%{x}<br>Profit: %{y:.2f} €/MWh<extra></extra>"
  }], {
    margin: { l: 60, r: 20, t: 20, b: 120 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { title: "Date | Contract", tickangle: -60, gridcolor: "#eaecf0" },
    yaxis: { title: "Profit (€/MWh)", gridcolor: "#eaecf0" }
  }, {
    responsive: true,
    displayModeBar: false
  });
}
function renderCumulativeCurve(filtered) {
  if (typeof Plotly === "undefined") return;

  const labels = filtered.map(x => `${x.date} | ${x.contract}`);
  const xVals = filtered.map((_, i) => i);
  const profits = filtered.map(x => Number(x.profit));

  const cumulative = [];
  profits.reduce((acc, val, i) => {
    const next = acc + val;
    cumulative[i] = next;
    return next;
  }, 0);

  const step = Math.max(1, Math.ceil(labels.length / 18));
  const tickVals = [];
  const tickText = [];

  for (let i = 0; i < labels.length; i += step) {
    const parts = labels[i].split(" | ");
    if (parts.length === 2) {
      const shortDate = parts[0].slice(5);
      tickVals.push(i);
      tickText.push(`${shortDate}<br>${parts[1]}`);
    } else {
      tickVals.push(i);
      tickText.push(labels[i]);
    }
  }

  Plotly.newPlot(
    "cumulativeCurve",
    [
      {
        x: xVals,
        y: cumulative,
        mode: "lines+markers",
        line: { color: "#16a34a", width: 3 },
        marker: { size: 5 },
        customdata: labels,
        hovertemplate: "%{customdata}<br>Cumulative: %{y:.2f} €/MWh<extra></extra>"
      }
    ],
    {
      margin: { l: 70, r: 20, t: 20, b: 120 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      xaxis: {
        title: "Date | Contract",
        tickmode: "array",
        tickvals: tickVals,
        ticktext: tickText,
        tickangle: 0,
        showgrid: false,
        zeroline: false,
        automargin: true
      },
      yaxis: {
        title: "Cumulative P&L (€/MWh)",
        gridcolor: "#eaecf0",
        automargin: true
      }
    },
    {
      responsive: true,
      displayModeBar: false
    }
  );
}



  Plotly.newPlot(
    "cumulativeCurve",
    [
      {
        x: labels,
        y: cumulative,
        mode: "lines+markers",
        line: { color: "#16a34a", width: 3 },
        marker: { size: 5 },
        hovertemplate: "%{x}<br>Cumulative: %{y:.2f} €/MWh<extra></extra>"
      }
    ],
    {
      margin: { l: 70, r: 20, t: 20, b: 140 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      xaxis: {
        title: "Date | Contract",
        type: "category",
        categoryorder: "array",
        categoryarray: labels,
        tickmode: "array",
        tickvals: tickVals,
        ticktext: tickText,
        tickangle: 0,
        showgrid: false,
        automargin: true
      },
      yaxis: {
        title: "Cumulative P&L (€/MWh)",
        gridcolor: "#eaecf0",
        automargin: true
      }
    },
    {
      responsive: true,
      displayModeBar: false
    }
  );
}

function renderHeatmap(filtered) {
  if (typeof Plotly === "undefined") return;
  if (!filtered.length) {
    renderEmptyPlot("heatmap");
    return;
  }

  const dates = unique(filtered.map(x => x.date)).sort();
  const contracts = unique(filtered.map(x => x.contract)).sort((a, b) => {
    const aRow = filtered.find(x => x.contract === a);
    const bRow = filtered.find(x => x.contract === b);
    return Number(aRow?.contract_sort ?? 0) - Number(bRow?.contract_sort ?? 0);
  });

  const matrix = contracts.map(contract =>
    dates.map(date => {
      const rows = filtered.filter(r => r.contract === contract && r.date === date);
      if (!rows.length) return null;
      return rows.reduce((s, r) => s + Number(r.profit), 0) / rows.length;
    })
  );

  Plotly.newPlot("heatmap", [{
    z: matrix,
    x: dates,
    y: contracts,
    type: "heatmap",
    colorscale: "RdYlGn",
    reversescale: false,
    hovertemplate: "Date: %{x}<br>Contract: %{y}<br>Avg Profit: %{z:.2f} €/MWh<extra></extra>"
  }], {
    margin: { l: 90, r: 20, t: 20, b: 80 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { title: "Date" },
    yaxis: { title: "Quarter-hour contract" }
  }, {
    responsive: true,
    displayModeBar: false
  });
}

function buildMiniTable(rows) {
  if (!rows.length) return "<div>No data</div>";

  return `
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Contract</th>
          <th>Buy</th>
          <th>Sell</th>
          <th>Profit</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(d => `
          <tr>
            <td>${d.date}</td>
            <td>${d.contract}</td>
            <td>${Number(d.buy_price).toFixed(2)}</td>
            <td>${Number(d.sell_price).toFixed(2)}</td>
            <td>${Number(d.profit).toFixed(2)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderTopBottomTables(filtered) {
  const top10 = [...filtered].sort((a, b) => b.profit - a.profit).slice(0, 10);
  const bottom10 = [...filtered].sort((a, b) => a.profit - b.profit).slice(0, 10);
  setHTML("topContracts", buildMiniTable(top10));
  setHTML("bottomContracts", buildMiniTable(bottom10));
}

function renderBreakdownTable(filtered) {
  setHTML("table", `
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Contract</th>
          <th>Buy</th>
          <th>Sell</th>
          <th>Profit</th>
        </tr>
      </thead>
      <tbody>
        ${filtered.map(d => `
          <tr>
            <td>${d.date}</td>
            <td>${d.contract}</td>
            <td>${Number(d.buy_price).toFixed(2)}</td>
            <td>${Number(d.sell_price).toFixed(2)}</td>
            <td>${Number(d.profit).toFixed(2)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `);
}

function render() {
  const filtered = getFilteredRows();

  if (!filtered.length) {
    showNoDataMessage("Try another area, strategy, date range, date type, or contract selection.");
    return;
  }

  renderMetricCards(filtered);
  renderBessStrategy(filtered);
  renderMultiCycleBess(filtered);
  renderCumulativeCurve(filtered);
  renderContractBar(filtered);
  renderHeatmap(filtered);
  renderHistogram(filtered);
  renderTopBottomTables(filtered);
  renderBreakdownTable(filtered);
}

byId("area")?.addEventListener("change", updateContracts);
byId("rule")?.addEventListener("change", updateContracts);
byId("direction")?.addEventListener("change", render);
byId("dateMode")?.addEventListener("change", () => {
  updateDateInputs();
  updateContracts();
});
byId("startDate")?.addEventListener("change", updateContracts);
byId("endDate")?.addEventListener("change", updateContracts);
byId("bessCapacity")?.addEventListener("change", render);
byId("bessPower")?.addEventListener("change", render);
byId("bessEfficiency")?.addEventListener("change", render);

byId("contracts")?.addEventListener("change", () => {
  setActivePreset(null);
  render();
});

byId("selectAllBtn")?.addEventListener("click", () => {
  selectAllOptions("contracts");
  setActivePreset("presetBaseBtn");
  render();
});

byId("clearAllBtn")?.addEventListener("click", () => {
  clearAllOptions("contracts");
  setActivePreset(null);
  render();
});

byId("presetBaseBtn")?.addEventListener("click", () => applyContractPreset("base"));
byId("presetPeakBtn")?.addEventListener("click", () => applyContractPreset("peak"));
byId("presetOffPeakBtn")?.addEventListener("click", () => applyContractPreset("offpeak"));
byId("presetMorningBtn")?.addEventListener("click", () => applyContractPreset("morning"));
byId("presetEveningBtn")?.addEventListener("click", () => applyContractPreset("evening"));

loadData();
