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
  const table = document.getElementById("table");
  if (table) {
    table.innerHTML = `
      <div style="padding:16px;border:1px solid #fda29b;background:#fff1f3;border-radius:12px;color:#b42318;">
        <strong>Data loading error</strong><br />
        ${message}
      </div>
    `;
  }
}

function unique(arr) {
  return [...new Set(arr)];
}

function setOptions(id, values, labelMap = null) {
  const el = document.getElementById(id);
  el.innerHTML = "";

  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.text = labelMap && labelMap[v] ? labelMap[v] : v;
    el.appendChild(opt);
  });
}

function selectAllOptions(id) {
  [...document.getElementById(id).options].forEach(o => {
    o.selected = true;
  });
}

function clearAllOptions(id) {
  [...document.getElementById(id).options].forEach(o => {
    o.selected = false;
  });
}

function getSelectedValues(id) {
  return [...document.getElementById(id).selectedOptions].map(x => x.value);
}

function setActivePreset(buttonId) {
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.classList.remove("active-preset");
  });

  if (buttonId) {
    const btn = document.getElementById(buttonId);
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
  const options = [...document.getElementById("contracts").options];

  options.forEach(opt => {
    const mins = parseContractStartMinutes(opt.value);
    if (mins === null) {
      opt.selected = false;
      return;
    }

    let selected = false;

    if (presetName === "base") {
      selected = true;
    } else if (presetName === "peak") {
      selected = mins >= 8 * 60 && mins < 20 * 60;
    } else if (presetName === "offpeak") {
      selected = mins < 8 * 60 || mins >= 20 * 60;
    } else if (presetName === "morning") {
      selected = mins >= 6 * 60 && mins < 12 * 60;
    } else if (presetName === "evening") {
      selected = mins >= 17 * 60 && mins < 23 * 60;
    }

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

function parseDateContractToIndex(row) {
  const datePart = String(row.date);
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

function populateSelectors() {
  const areas = unique(data.map(x => x.area)).filter(Boolean).sort();
  const rules = unique(data.map(x => x.rule)).filter(Boolean).sort();
  const dates = unique(data.map(x => x.date)).filter(Boolean).sort();

  if (!areas.length) throw new Error("No areas found in data.");
  if (!rules.length) throw new Error("No strategies found in data.");
  if (!dates.length) throw new Error("No dates found in data.");

  setOptions("area", areas);
  setOptions("rule", rules, RULE_LABELS);

  document.getElementById("startDate").value = dates[0];
  document.getElementById("endDate").value = dates[dates.length - 1];

  updateContracts();
}

function updateContracts() {
  const area = document.getElementById("area").value;
  const rule = document.getElementById("rule").value;
  const startDate = document.getElementById("startDate").value;
  const endDate = document.getElementById("endDate").value;

  const filtered = data
    .filter(d => {
      if (d.area !== area) return false;
      if (d.rule !== rule) return false;
      if (startDate && d.date < startDate) return false;
      if (endDate && d.date > endDate) return false;
      return true;
    })
    .sort(compareRowsChronologically);

  const contracts = unique(filtered.map(x => x.contract));
  setOptions("contracts", contracts);
  selectAllOptions("contracts");
  setActivePreset("presetBaseBtn");

  render();
}

function getFilteredRows() {
  const area = document.getElementById("area").value;
  const rule = document.getElementById("rule").value;
  const direction = document.getElementById("direction").value;
  const startDate = document.getElementById("startDate").value;
  const endDate = document.getElementById("endDate").value;
  const selectedContracts = getSelectedValues("contracts");

  let filtered = data
    .filter(d => {
      if (d.area !== area) return false;
      if (d.rule !== rule) return false;
      if (startDate && d.date < startDate) return false;
      if (endDate && d.date > endDate) return false;
      if (!selectedContracts.includes(d.contract)) return false;
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
  const winRate = profits.length
    ? (profits.filter(x => x > 0).length / profits.length) * 100
    : 0;

  const best = filtered.length
    ? [...filtered].sort((a, b) => b.profit - a.profit)[0]
    : null;

  const worst = filtered.length
    ? [...filtered].sort((a, b) => a.profit - b.profit)[0]
    : null;

  const profitEl = document.getElementById("profit");
  const contractCountEl = document.getElementById("contractCount");
  const avgProfitEl = document.getElementById("avgProfit");
  const winRateEl = document.getElementById("winRate");
  const bestEl = document.getElementById("bestContract");
  const worstEl = document.getElementById("worstContract");

  if (profitEl) profitEl.innerText = `${total.toFixed(2)} €/MWh`;
  if (contractCountEl) contractCountEl.innerText = `${filtered.length}`;
  if (avgProfitEl) avgProfitEl.innerText = `${avg.toFixed(2)} €/MWh`;
  if (winRateEl) winRateEl.innerText = `${winRate.toFixed(1)}%`;

  if (bestEl) {
    bestEl.innerHTML = best
      ? `${best.date}<br>${best.contract}<br>${Number(best.profit).toFixed(2)} €/MWh`
      : "-";
  }

  if (worstEl) {
    worstEl.innerHTML = worst
      ? `${worst.date}<br>${worst.contract}<br>${Number(worst.profit).toFixed(2)} €/MWh`
      : "-";
  }
}

function renderBessStrategy(filtered) {
  const bessEl = document.getElementById("bessStrategy");
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

    if (minBuySoFar) {
      const spread = sellPrice - Number(minBuySoFar.buy_price);
      if (spread > bestSpread && compareRowsChronologically(minBuySoFar, row) <= 0) {
        bestSpread = spread;
        bestChargeRow = minBuySoFar;
        bestDischargeRow = row;
      }
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
    <strong>Single-cycle spread:</strong> <span class="${bestSpread >= 0 ? 'positive-text' : 'negative-text'}">${bestSpread.toFixed(2)} €/MWh</span>
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
  const el = document.getElementById("bessMultiCycle");
  if (!el) return;

  if (!filtered.length) {
    el.innerHTML = "-";
    return;
  }

  const capacityMWh = Number(document.getElementById("bessCapacity").value || 1);
  const powerMW = Number(document.getElementById("bessPower").value || 1);
  const efficiency = Number(document.getElementById("bessEfficiency").value || 0.9);

  if (!Number.isFinite(capacityMWh) || !Number.isFinite(powerMW) || !Number.isFinite(efficiency) ||
      capacityMWh <= 0 || powerMW <= 0 || efficiency <= 0 || efficiency > 1) {
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
    const shouldCharge =
      futureAvgSell !== null &&
      soc < capacityMWh &&
      buyPrice < chargeThreshold;

    const shouldDischarge =
      soc > 0 &&
      (
        futureAvgSell === null ||
        sellPrice >= futureAvgSell ||
        i >= ordered.length - 4
      );

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
    <span class="${totalPnL >= 0 ? 'positive-text' : 'negative-text'}">${totalPnL.toFixed(2)} €</span>
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
  const labels = filtered.map(x => `${x.date} | ${x.contract}`);
  const profits = filtered.map(x => Number(x.profit));
  const colors = profits.map(v => v >= 0 ? "#16a34a" : "#dc2626");

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
  const labels = filtered.map(x => `${x.date} | ${x.contract}`);
  const profits = filtered.map(x => Number(x.profit));

  const cumulative = [];
  profits.reduce((acc, val, i) => {
    const next = acc + val;
    cumulative[i] = next;
    return next;
  }, 0);

  Plotly.newPlot("cumulativeCurve", [{
    x: labels,
    y: cumulative,
    mode: "lines+markers",
    line: { color: "#16a34a", width: 3 },
    marker: { size: 6 },
    hovertemplate: "%{x}<br>Cumulative: %{y:.2f} €/MWh<extra></extra>"
  }], {
    margin: { l: 60, r: 20, t: 20, b: 120 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { title: "Date | Contract", tickangle: -60, gridcolor: "#eaecf0" },
    yaxis: { title: "Cumulative P&L (€/MWh)", gridcolor: "#eaecf0" }
  }, {
    responsive: true,
    displayModeBar: false
  });
}

function renderHeatmap(filtered) {
  if (!filtered.length) {
    Plotly.newPlot("heatmap", [], {
      annotations: [{ text: "No data", showarrow: false }]
    });
    return;
  }

  const dates = unique(filtered.map(x => x.date)).sort();
  const contracts = unique(filtered.map(x => x.contract)).sort((a, b) => {
    const aRow = filtered.find(x => x.contract === a);
    const bRow = filtered.find(x => x.contract === b);
    return Number(aRow?.contract_sort ?? 0) - Number(bRow?.contract_sort ?? 0);
  });

  const matrix = contracts.map(contract => {
    return dates.map(date => {
      const rows = filtered.filter(r => r.contract === contract && r.date === date);
      if (!rows.length) return null;
      const avg = rows.reduce((s, r) => s + Number(r.profit), 0) / rows.length;
      return avg;
    });
  });

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

function renderTopBottomTables(filtered) {
  const top10 = [...filtered].sort((a, b) => b.profit - a.profit).slice(0, 10);
  const bottom10 = [...filtered].sort((a, b) => a.profit - b.profit).slice(0, 10);

  document.getElementById("topContracts").innerHTML = buildMiniTable(top10);
  document.getElementById("bottomContracts").innerHTML = buildMiniTable(bottom10);
}

function buildMiniTable(rows) {
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

function renderBreakdownTable(filtered) {
  document.getElementById("table").innerHTML = `
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
  `;
}

function render() {
  const filtered = getFilteredRows();

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

document.getElementById("area").addEventListener("change", updateContracts);
document.getElementById("rule").addEventListener("change", updateContracts);
document.getElementById("direction").addEventListener("change", render);
document.getElementById("startDate").addEventListener("change", updateContracts);
document.getElementById("endDate").addEventListener("change", updateContracts);

document.getElementById("bessCapacity").addEventListener("change", render);
document.getElementById("bessPower").addEventListener("change", render);
document.getElementById("bessEfficiency").addEventListener("change", render);

document.getElementById("contracts").addEventListener("change", () => {
  setActivePreset(null);
  render();
});

document.getElementById("selectAllBtn").addEventListener("click", () => {
  selectAllOptions("contracts");
  setActivePreset("presetBaseBtn");
  render();
});

document.getElementById("clearAllBtn").addEventListener("click", () => {
  clearAllOptions("contracts");
  setActivePreset(null);
  render();
});

document.getElementById("presetBaseBtn").addEventListener("click", () => {
  applyContractPreset("base");
});

document.getElementById("presetPeakBtn").addEventListener("click", () => {
  applyContractPreset("peak");
});

document.getElementById("presetOffPeakBtn").addEventListener("click", () => {
  applyContractPreset("offpeak");
});

document.getElementById("presetMorningBtn").addEventListener("click", () => {
  applyContractPreset("morning");
});

document.getElementById("presetEveningBtn").addEventListener("click", () => {
  applyContractPreset("evening");
});

loadData();
