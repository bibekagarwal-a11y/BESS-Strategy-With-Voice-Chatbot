let optimizerData = [];

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

function unique(arr) {
  return [...new Set(arr)];
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

async function loadOptimizerData() {
  const statusEl = byId("optimizerDataStatus");

  try {
    const res = await fetch("./data/contract_profits.json", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Failed to load data: ${res.status} ${res.statusText}`);
    }

    optimizerData = await res.json();

    if (!Array.isArray(optimizerData) || optimizerData.length === 0) {
      throw new Error("contract_profits.json is empty.");
    }

    populateDatasetScope();

    if (statusEl) {
      statusEl.innerHTML = buildDatasetSummary();
    }
  } catch (err) {
    console.error(err);
    if (statusEl) {
      statusEl.innerHTML = `
        <div style="color:#b42318;">
          <strong>Dataset error</strong><br>
          ${err.message}
        </div>
      `;
    }
  }
}

function buildDatasetSummary() {
  const areas = unique(optimizerData.map(x => x.area)).filter(Boolean).sort();
  const rules = unique(optimizerData.map(x => x.rule)).filter(Boolean).sort();
  const dates = unique(optimizerData.map(x => x.date)).filter(Boolean).sort();

  const minDate = dates[0] || "-";
  const maxDate = dates[dates.length - 1] || "-";

  return `
    <div class="optimizer-result-card">
      <div class="optimizer-result-title">Loaded historical dataset</div>
      <div class="optimizer-result-grid">
        <div><strong>Rows:</strong> ${optimizerData.length}</div>
        <div><strong>Areas:</strong> ${areas.join(", ") || "-"}</div>
        <div><strong>Reference strategies:</strong> ${rules.length}</div>
        <div><strong>Date range:</strong> ${minDate} → ${maxDate}</div>
      </div>
    </div>
  `;
}

function populateDatasetScope() {
  const areas = unique(optimizerData.map(x => x.area)).filter(Boolean).sort();
  const rules = unique(optimizerData.map(x => x.rule)).filter(Boolean).sort();
  const dates = unique(optimizerData.map(x => x.date)).filter(Boolean).sort();

  setOptions("optimizerArea", areas);
  setOptions("optimizerRule", rules, RULE_LABELS);

  if (byId("startDate") && dates.length) byId("startDate").value = dates[0];
  if (byId("endDate") && dates.length) byId("endDate").value = dates[dates.length - 1];
}

function getSelectedMarkets() {
  const markets = [];
  if (byId("marketDA")?.checked) markets.push("DA");
  if (byId("marketIDA1")?.checked) markets.push("IDA1");
  if (byId("marketIDA2")?.checked) markets.push("IDA2");
  if (byId("marketIDA3")?.checked) markets.push("IDA3");
  if (byId("marketVWAP")?.checked) markets.push("VWAP");
  return markets;
}

function getOptimizerInputs() {
  return {
    powerMw: Number(byId("bessPowerMw")?.value || 0),
    capacityMWh: Number(byId("bessCapacityMWh")?.value || 0),
    efficiency: Number(byId("bessEfficiency")?.value || 0),
    minSoc: Number(byId("bessMinSoc")?.value || 0),
    maxSoc: Number(byId("bessMaxSoc")?.value || 0),
    dailyCycleLimit: Number(byId("bessDailyCycleLimit")?.value || 0),
    area: byId("optimizerArea")?.value || "",
    referenceRule: byId("optimizerRule")?.value || "",
    strategyMode: byId("strategyMode")?.value || "charge_discharge",
    startDate: byId("startDate")?.value || "",
    endDate: byId("endDate")?.value || "",
    markets: getSelectedMarkets()
  };
}

function validateInputs(inputs) {
  if (!optimizerData.length) return "No historical dataset is loaded.";
  if (inputs.powerMw <= 0) return "BESS MW must be greater than 0.";
  if (inputs.capacityMWh <= 0) return "BESS MWh must be greater than 0.";
  if (inputs.efficiency <= 0 || inputs.efficiency > 1) return "η must be between 0 and 1.";
  if (inputs.minSoc < 0 || inputs.minSoc > 100) return "Min SoC must be between 0 and 100.";
  if (inputs.maxSoc < 0 || inputs.maxSoc > 100) return "Max SoC must be between 0 and 100.";
  if (inputs.minSoc >= inputs.maxSoc) return "Min SoC must be lower than Max SoC.";
  if (inputs.dailyCycleLimit < 0) return "Daily cycle limit cannot be negative.";
  if (!inputs.startDate || !inputs.endDate) return "Please select both start and end dates.";
  if (inputs.startDate > inputs.endDate) return "Start date must be before end date.";
  if (!inputs.area) return "Please select an area.";
  if (!inputs.referenceRule) return "Please select a reference strategy.";
  if (!inputs.markets.length) return "Select at least one market.";

  const ruleMarkets = inputs.referenceRule.split("_");
  const missing = ruleMarkets.filter(m => !inputs.markets.includes(m));
  if (missing.length) {
    return `Selected markets must include the reference rule markets: ${ruleMarkets.join(", ")}.`;
  }

  return null;
}

function getScopedRows(inputs) {
  return optimizerData
    .filter(row => {
      if (inputs.area && row.area !== inputs.area) return false;
      if (inputs.referenceRule && row.rule !== inputs.referenceRule) return false;
      if (inputs.startDate && String(row.date ?? "") < inputs.startDate) return false;
      if (inputs.endDate && String(row.date ?? "") > inputs.endDate) return false;
      return true;
    })
    .sort((a, b) => {
      const dateCompare = String(a.date).localeCompare(String(b.date));
      if (dateCompare !== 0) return dateCompare;
      return Number(a.contract_sort ?? 0) - Number(b.contract_sort ?? 0);
    });
}

function percentile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx];
}

function parseContractHours(contractLabel) {
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

function buildCandidateStrategies(mode) {
  const buyThresholds = [0.15, 0.20, 0.25, 0.30, 0.35];
  const sellThresholds = [0.65, 0.70, 0.75, 0.80, 0.85];
  const candidates = [];

  if (mode === "charge_only") {
    buyThresholds.forEach(bq => {
      candidates.push({ mode, buyQ: bq, sellQ: null });
    });
    return candidates;
  }

  if (mode === "discharge_only") {
    sellThresholds.forEach(sq => {
      candidates.push({ mode, buyQ: null, sellQ: sq });
    });
    return candidates;
  }

  buyThresholds.forEach(bq => {
    sellThresholds.forEach(sq => {
      if (sq > bq) {
        candidates.push({ mode, buyQ: bq, sellQ: sq });
      }
    });
  });

  return candidates;
}

function runSingleBacktest(rows, inputs, candidate) {
  if (!rows.length) {
    return null;
  }

  const buyPrices = rows.map(r => Number(r.buy_price)).filter(Number.isFinite);
  const sellPrices = rows.map(r => Number(r.sell_price)).filter(Number.isFinite);

  const buyThreshold = candidate.buyQ !== null ? percentile(buyPrices, candidate.buyQ) : null;
  const sellThreshold = candidate.sellQ !== null ? percentile(sellPrices, candidate.sellQ) : null;

  const eta = Math.sqrt(inputs.efficiency);
  const minSocMWh = inputs.capacityMWh * (inputs.minSoc / 100);
  const maxSocMWh = inputs.capacityMWh * (inputs.maxSoc / 100);

  let soc =
    candidate.mode === "discharge_only"
      ? maxSocMWh
      : candidate.mode === "charge_only"
      ? minSocMWh
      : (minSocMWh + maxSocMWh) / 2;

  let totalPnL = 0;
  let chargeEnergyRaw = 0;
  let dischargeEnergyRaw = 0;
  let chargeActions = 0;
  let dischargeActions = 0;

  let currentDate = null;
  let chargedTodayRaw = 0;

  const actions = [];

  rows.forEach(row => {
    const date = String(row.date);
    if (currentDate !== date) {
      currentDate = date;
      chargedTodayRaw = 0;
    }

    const durationH = parseContractHours(row.contract);
    const stepRawLimit = inputs.powerMw * durationH;
    const dailyChargeBudgetRaw = inputs.dailyCycleLimit * inputs.capacityMWh;
    const dailyChargeRemaining = Math.max(0, dailyChargeBudgetRaw - chargedTodayRaw);

    const buyPrice = Number(row.buy_price);
    const sellPrice = Number(row.sell_price);

    if (!Number.isFinite(buyPrice) || !Number.isFinite(sellPrice)) {
      return;
    }

    let action = "idle";
    let energyRaw = 0;
    let pnlDelta = 0;

    const canCharge =
      candidate.mode !== "discharge_only" &&
      buyThreshold !== null &&
      buyPrice <= buyThreshold &&
      soc < maxSocMWh &&
      dailyChargeRemaining > 0;

    const canDischarge =
      candidate.mode !== "charge_only" &&
      sellThreshold !== null &&
      sellPrice >= sellThreshold &&
      soc > minSocMWh;

    if (canCharge) {
      const socRoomRaw = (maxSocMWh - soc) / eta;
      const chargeRaw = Math.max(0, Math.min(stepRawLimit, dailyChargeRemaining, socRoomRaw));

      if (chargeRaw > 0) {
        soc += chargeRaw * eta;
        chargedTodayRaw += chargeRaw;
        chargeEnergyRaw += chargeRaw;
        chargeActions += 1;
        pnlDelta = -(chargeRaw * buyPrice);
        totalPnL += pnlDelta;
        action = "charge";
        energyRaw = chargeRaw;
      }
    } else if (canDischarge) {
      const availableRaw = Math.max(0, soc - minSocMWh);
      const dischargeRaw = Math.min(stepRawLimit, availableRaw);

      if (dischargeRaw > 0) {
        const delivered = dischargeRaw * eta;
        soc -= dischargeRaw;
        dischargeEnergyRaw += dischargeRaw;
        dischargeActions += 1;
        pnlDelta = delivered * sellPrice;
        totalPnL += pnlDelta;
        action = "discharge";
        energyRaw = dischargeRaw;
      }
    }

    actions.push({
      date: row.date,
      contract: row.contract,
      action,
      buy_price: buyPrice,
      sell_price: sellPrice,
      energy_raw_mwh: energyRaw,
      pnl_delta: pnlDelta,
      soc_after: soc
    });
  });

  const activeActions = actions.filter(a => a.action !== "idle");
  const equivalentCycles = inputs.capacityMWh > 0 ? chargeEnergyRaw / inputs.capacityMWh : 0;

  return {
    candidate,
    buyThreshold,
    sellThreshold,
    totalPnL,
    chargeEnergyRaw,
    dischargeEnergyRaw,
    equivalentCycles,
    chargeActions,
    dischargeActions,
    endingSoc: soc,
    actions,
    activeActions
  };
}

function runOptimizerBacktest(rows, inputs) {
  const candidates = buildCandidateStrategies(inputs.strategyMode);
  const results = candidates
    .map(candidate => runSingleBacktest(rows, inputs, candidate))
    .filter(Boolean);

  if (!results.length) return null;

  results.sort((a, b) => b.totalPnL - a.totalPnL);
  return results[0];
}

function formatMode(mode) {
  if (mode === "charge_only") return "Charge only";
  if (mode === "discharge_only") return "Discharge only";
  if (mode === "arbitrage") return "Quant / Buy low, sell high";
  return "Charge + Discharge";
}

function renderActionTable(actions) {
  if (!actions.length) {
    return "<div>No charge/discharge actions were triggered by the selected strategy.</div>";
  }

  const topActions = actions.slice(0, 20);

  return `
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Contract</th>
          <th>Action</th>
          <th>Buy</th>
          <th>Sell</th>
          <th>Energy</th>
          <th>P&amp;L Δ</th>
          <th>SoC After</th>
        </tr>
      </thead>
      <tbody>
        ${topActions.map(a => `
          <tr>
            <td>${a.date}</td>
            <td>${a.contract}</td>
            <td>${a.action}</td>
            <td>${a.buy_price.toFixed(2)}</td>
            <td>${a.sell_price.toFixed(2)}</td>
            <td>${a.energy_raw_mwh.toFixed(2)}</td>
            <td>${a.pnl_delta.toFixed(2)}</td>
            <td>${a.soc_after.toFixed(2)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderBacktestResult(inputs, rows, bestResult) {
  const resultEl = byId("optimizerResult");
  if (!resultEl) return;

  if (!bestResult) {
    resultEl.innerHTML = `
      <div class="optimizer-placeholder" style="border-color:#fda29b;background:#fff1f3;color:#b42318;">
        <strong>Backtest error</strong><br>
        No valid strategy result could be computed.
      </div>
    `;
    return;
  }

  const strategyNote =
    inputs.strategyMode === "charge_only"
      ? `Recommended charge threshold: buy when buy price is in the lowest ${(bestResult.candidate.buyQ * 100).toFixed(0)}% of scoped history.`
      : inputs.strategyMode === "discharge_only"
      ? `Recommended discharge threshold: discharge when sell price is in the highest ${(100 - bestResult.candidate.sellQ * 100).toFixed(0)}% tail of scoped history.`
      : `Recommended thresholds: charge when buy price is below ${bestResult.buyThreshold?.toFixed(2) ?? "-"} €/MWh and discharge when sell price is above ${bestResult.sellThreshold?.toFixed(2) ?? "-"} €/MWh.`;

  resultEl.innerHTML = `
    <div class="optimizer-result-card">
      <div class="optimizer-result-title">Recommended strategy</div>
      <div><strong>Mode:</strong> ${formatMode(inputs.strategyMode)}</div>
      <div><strong>Reference strategy:</strong> ${RULE_LABELS[inputs.referenceRule] || inputs.referenceRule}</div>
      <div><strong>Markets selected:</strong> ${inputs.markets.join(", ")}</div>
      <div><strong>Scope:</strong> ${inputs.area} | ${inputs.startDate} → ${inputs.endDate}</div>
      <div style="margin-top:8px;">${strategyNote}</div>

      <div class="optimizer-result-grid" style="margin-top:14px;">
        <div><strong>Scoped rows:</strong> ${rows.length}</div>
        <div><strong>Total P&amp;L:</strong> ${bestResult.totalPnL.toFixed(2)} €</div>
        <div><strong>Charge actions:</strong> ${bestResult.chargeActions}</div>
        <div><strong>Discharge actions:</strong> ${bestResult.dischargeActions}</div>
        <div><strong>Charged energy:</strong> ${bestResult.chargeEnergyRaw.toFixed(2)} MWh</div>
        <div><strong>Discharged energy:</strong> ${bestResult.dischargeEnergyRaw.toFixed(2)} MWh</div>
        <div><strong>Equivalent charge cycles:</strong> ${bestResult.equivalentCycles.toFixed(2)}</div>
        <div><strong>Ending SoC:</strong> ${bestResult.endingSoc.toFixed(2)} MWh</div>
      </div>

      <div style="margin-top:16px;">
        <strong>Notes</strong><br>
        This Step 4 optimizer uses the selected historical reference strategy rows and battery constraints only. It does not yet do full multi-market market-routing; that comes in the next step.
      </div>
    </div>

    <div style="margin-top:16px;">
      <div class="optimizer-result-title">First 20 active actions</div>
      ${renderActionTable(bestResult.activeActions)}
    </div>
  `;
}

function handleRunOptimizer() {
  const inputs = getOptimizerInputs();
  const error = validateInputs(inputs);

  if (error) {
    const resultEl = byId("optimizerResult");
    if (resultEl) {
      resultEl.innerHTML = `
        <div class="optimizer-placeholder" style="border-color:#fda29b;background:#fff1f3;color:#b42318;">
          <strong>Input error</strong><br>
          ${error}
        </div>
      `;
    }
    return;
  }

  const rows = getScopedRows(inputs);
  if (!rows.length) {
    const resultEl = byId("optimizerResult");
    if (resultEl) {
      resultEl.innerHTML = `
        <div class="optimizer-placeholder" style="border-color:#fda29b;background:#fff1f3;color:#b42318;">
          <strong>No scoped data</strong><br>
          No historical rows match the selected area, strategy, and date range.
        </div>
      `;
    }
    return;
  }

  const bestResult = runOptimizerBacktest(rows, inputs);
  renderBacktestResult(inputs, rows, bestResult);
}

byId("runOptimizerBtn")?.addEventListener("click", handleRunOptimizer);

loadOptimizerData();

console.log("Advanced BESS Optimizer Step 4 loaded.");
