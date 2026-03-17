let optimizerData = [];

const RULE_LABELS = {
  DA_IDA1: "Day Ahead â IDA1",
  DA_IDA2: "Day Ahead â IDA2",
  DA_IDA3: "Day Ahead â IDA3",
  DA_VWAP: "Day Ahead â Intraday VWAP",
  IDA1_IDA2: "IDA1 â IDA2",
  IDA1_IDA3: "IDA1 â IDA3",
  IDA1_VWAP: "IDA1 â Intraday VWAP",
  IDA2_IDA3: "IDA2 â IDA3",
  IDA2_VWAP: "IDA2 â Intraday VWAP",
  IDA3_VWAP: "IDA3 â Intraday VWAP",
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
  values.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.text = labelMap && labelMap[v] ? labelMap[v] : v;
    el.appendChild(opt);
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "-";
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
        <div class="optimizer-card">
          <strong>Dataset error</strong><br />
          ${escapeHtml(err.message)}
        </div>
      `;
    }
  }
}

function buildDatasetSummary() {
  const areas = unique(optimizerData.map((x) => x.area)).filter(Boolean).sort();
  const rules = unique(optimizerData.map((x) => x.rule)).filter(Boolean).sort();
  const dates = unique(optimizerData.map((x) => x.date)).filter(Boolean).sort();

  return `
    <div class="optimizer-card">
      <strong>Loaded historical dataset</strong><br />
      Rows: ${optimizerData.length}<br />
      Areas: ${escapeHtml(areas.join(", ") || "-")}<br />
      Historical market pairs: ${rules.length}<br />
      Date range: ${escapeHtml(dates[0] || "-")} â ${escapeHtml(dates[dates.length - 1] || "-")}
    </div>
  `;
}

function populateDatasetScope() {
  const areas = unique(optimizerData.map((x) => x.area)).filter(Boolean).sort();
  const dates = unique(optimizerData.map((x) => x.date)).filter(Boolean).sort();

  setOptions("optimizerArea", areas);

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

function getMarketCosts() {
  return {
    transactionCost: Number(byId("transactionCostEurMWh")?.value || 0.5),
    bidAskSpread: Number(byId("bidAskSpreadEurMWh")?.value || 1.0),
    slippage: Number(byId("slippagePercent")?.value || 0.5) / 100,
    wearCost: Number(byId("wearCostEurMWh")?.value || 2.0),
    gateClosureHours: Number(byId("gateClosureHoursInput")?.value || 0),
    lotSizeMw: Number(byId("lotSizeMwInput")?.value || 0.1),
  };
}

function getOptimizerInputs() {
  const costs = getMarketCosts();
  return {
    powerMw: Number(byId("bessPowerMw")?.value || 0),
    capacityMWh: Number(byId("bessCapacityMWh")?.value || 0),
    efficiency: Number(byId("bessEfficiency")?.value || 0),
    minSoc: Number(byId("bessMinSoc")?.value || 0),
    maxSoc: Number(byId("bessMaxSoc")?.value || 0),
    dailyCycleLimit: Number(byId("bessDailyCycleLimit")?.value || 0),
    area: byId("optimizerArea")?.value || "",
    strategyMode: byId("strategyMode")?.value || "charge_discharge",
    startDate: byId("startDate")?.value || "",
    endDate: byId("endDate")?.value || "",
    markets: getSelectedMarkets(),
    costs,
  };
}

function validateInputs(inputs) {
  if (!optimizerData.length) return "No historical dataset is loaded.";
  if (inputs.powerMw <= 0) return "BESS MW must be greater than 0.";
  if (inputs.capacityMWh <= 0) return "BESS MWh must be greater than 0.";
  if (inputs.efficiency <= 0 || inputs.efficiency > 1) return "Î· must be between 0 and 1.";
  if (inputs.minSoc < 0 || inputs.minSoc > 100) return "Min SoC must be between 0 and 100.";
  if (inputs.maxSoc < 0 || inputs.maxSoc > 100) return "Max SoC must be between 0 and 100.";
  if (inputs.minSoc >= inputs.maxSoc) return "Min SoC must be lower than Max SoC.";
  if (inputs.dailyCycleLimit < 0) return "Daily cycle limit cannot be negative.";
  if (!inputs.startDate || !inputs.endDate) return "Please select both start and end dates.";
  if (inputs.startDate > inputs.endDate) return "Start date must be before end date.";
  if (!inputs.area) return "Please select an area.";
  if (!inputs.markets.length) return "Select at least one market.";
  return null;
}

function getScopedRows(inputs) {
  return optimizerData
    .filter((row) => {
      if (inputs.area && row.area !== inputs.area) return false;
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

function getEligibleRules(rows, selectedMarkets) {
  const allRules = unique(rows.map((r) => r.rule)).filter(Boolean);
  return allRules.filter((rule) => rule.split("_").every((p) => selectedMarkets.includes(p)));
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

/**
 * Real Nord Pool / EPEX gate closure rules (factual market constraints):
 *
 *  DA Auction    â D-1 at 12:00 CET â covers all 24 delivery hours for day D
 *  IDA1 Auction  â D-1 at 15:00 CET â covers all 24 delivery hours for day D
 *  IDA2 Auction  â D-1 at 22:00 CET â covers all 24 delivery hours for day D
 *  IDA3 Auction  â D    at 10:00 CET â covers ONLY delivery hours 12:00â24:00 (hours â¥ 12)
 *  Continuous    â 5 minutes before contract start (e.g. 11:55 for 12:00â12:15)
 *
 * In backtesting all DA/IDA1/IDA2 prices are always accessible.
 * The one hard filter that removes real opportunities:
 *   IDA3 contracts with delivery starting before 12:00 are simply not tradeable via IDA3.
 *
 * Returns false if the row cannot be traded given real market gate closure rules.
 */
function isContractEligibleByMarketGate(row) {
  const rule = String(row.rule || "").toUpperCase();
  const contract = String(row.contract || "");

  // IDA3 auction only covers delivery hours 12:00â24:00
  if (rule.includes("IDA3")) {
    const startStr = contract.split("-")[0]; // e.g. "09:00"
    const startHour = parseInt((startStr || "0").split(":")[0], 10);
    if (Number.isFinite(startHour) && startHour < 12) {
      return false; // IDA3 gate (D 10:00) does not cover hours before 12:00
    }
  }

  return true;
}

function buildCandidateStrategies(mode) {
  const buyThresholds = [0.15, 0.2, 0.25, 0.3, 0.35];
  const sellThresholds = [0.65, 0.7, 0.75, 0.8, 0.85];
  const candidates = [];

  if (mode === "charge_only") {
    buyThresholds.forEach((bq) => candidates.push({ mode, buyQ: bq, sellQ: null }));
    return candidates;
  }

  if (mode === "discharge_only") {
    sellThresholds.forEach((sq) => candidates.push({ mode, buyQ: null, sellQ: sq }));
    return candidates;
  }

  buyThresholds.forEach((bq) => {
    sellThresholds.forEach((sq) => {
      if (sq > bq) candidates.push({ mode, buyQ: bq, sellQ: sq });
    });
  });

  return candidates;
}

function standardDeviation(values) {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
}

function runSingleBacktest(rows, inputs, candidate, rule, costs) {
  if (!rows.length) return null;

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
  const historicalBuyPrices = [];
  const historicalSellPrices = [];

  rows.forEach((row) => {
    const date = String(row.date);

    if (currentDate !== date) {
      currentDate = date;
      chargedTodayRaw = 0;
    }

    const durationH = parseContractHours(row.contract);

    // Gate closure check 1: user-configured duration filter
    if (durationH < costs.gateClosureHours) {
      return;
    }

    // Gate closure check 2: real Nord Pool market rules (IDA3 â¥ 12:00 only, etc.)
    if (!isContractEligibleByMarketGate(row)) {
      return;
    }

    const stepRawLimit = inputs.powerMw * durationH;
    const dailyChargeBudgetRaw = inputs.dailyCycleLimit * inputs.capacityMWh;
    const dailyChargeRemaining = Math.max(0, dailyChargeBudgetRaw - chargedTodayRaw);

    const buyPrice = Number(row.buy_price);
    const sellPrice = Number(row.sell_price);

    if (!Number.isFinite(buyPrice) || !Number.isFinite(sellPrice)) return;

    const buyThreshold =
      candidate.buyQ !== null && historicalBuyPrices.length > 0
        ? percentile(historicalBuyPrices, candidate.buyQ)
        : null;

    const sellThreshold =
      candidate.sellQ !== null && historicalSellPrices.length > 0
        ? percentile(historicalSellPrices, candidate.sellQ)
        : null;

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
      let chargeRaw = Math.max(0, Math.min(stepRawLimit, dailyChargeRemaining, socRoomRaw));

      // Lot size rounding
      chargeRaw = Math.round(chargeRaw / costs.lotSizeMw) * costs.lotSizeMw;

      if (chargeRaw > 0) {
        const adjustedBuyPrice = buyPrice * (1 + costs.slippage) + costs.bidAskSpread / 2;
    2   const actionCosts = chargeRaw * costs.transactionCost + chargeRaw * costs.wearCost;

        soc += chargeRaw * eta;
        chargedTodayRaw += chargeRaw;
        chargeEnergyRaw += chargeRaw;
        chargeActions += 1;
        pnlDelta = -(chargeRaw * adjustedBuyPrice + actionCosts);
        totalPnL += pnlDelta;
        action = "charge";
        energyRaw = chargeRaw;
      }
    } else if (canDischarge) {
      const availableRaw = Math.max(0, soc - minSocMWh);
      let dischargeRaw = Math.min(stepRawLimit, availableRaw);

      // Lot size rounding
      dischargeRaw = Math.round(dischargeRaw / costs.lotSizeMw) * costs.lotSizeMw;

      if (dischargeRaw > 0) {
        const adjustedSellPrice = sellPrice * (1 - costs.slippage) - costs.bidAskSpread / 2;
        const actionCosts = dischargeRaw * costs.transactionCost + dischargeRaw * costs.wearCost;
        const delivered = dischargeRaw * eta;

        soc -= dischargeRaw;
        dischargeEnergyRaw += dischargeRaw;
        dischargeActions += 1;
        pnlDelta = delivered * adjustedSellPrice - actionCosts;
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
      soc_after: soc,
      buy_threshold_used: buyThreshold,
      sell_threshold_used: sellThreshold,
    });

    historicalBuyPrices.push(buyPrice);
    historicalSellPrices.push(sellPrice);
  });

  const activeActions = actions.filter((a) => a.action !== "idle");

  const finalBuyThreshold =
    candidate.buyQ !== null && historicalBuyPrices.length > 0
      ? percentile(historicalBuyPrices, candidate.buyQ)
      : null;

  const finalSellThreshold =
    candidate.sellQ !== null && historicalSellPrices.length > 0
      ? percentile(historicalSellPrices, candidate.sellQ)
      : null;

  const equivalentCycles = inputs.capacityMWh > 0 ? chargeEnergyRaw / inputs.capacityMWh : 0;

  return {
    rule,
    candidate,
    buyThreshold: finalBuyThreshold,
    sellThreshold: finalSellThreshold,
    totalPnL,
    chargeEnergyRaw,
    dischargeEnergyRaw,
    equivalentCycles,
    chargeActions,
    dischargeActions,
    endingSoc: soc,
    actions,
    activeActions,
  };
}

function splitTrainValTest(rows) {
  const total = rows.length;
  const trainSize = Math.floor(total * 0.6);
  const valSize = Math.floor(total * 0.2);

  const trainRows = rows.slice(0, trainSize);
  const valRows = rows.slice(trainSize, trainSize + valSize);
  const testRows = rows.slice(trainSize + valSize);

  return { trainRows, valRows, testRows };
}

function runPerfectForesightBacktest(rows, inputs, costs) {
  if (!rows.length) return null;

  const eta = Math.sqrt(inputs.efficiency);
  const minSocMWh = inputs.capacityMWh * (inputs.minSoc / 100);
  const maxSocMWh = inputs.capacityMWh * (inputs.maxSoc / 100);

  let soc = (minSocMWh + maxSocMWh) / 2;
  let totalPnL = 0;
  let chargeEnergyRaw = 0;
  let dischargeEnergyRaw = 0;

  // Group by date for greedy algorithm
  const byDate = new Map();
  rows.forEach((row) => {
    const date = String(row.date);
    if (!byDate.has(date)) byDate.set(date, []);
    byDate.get(date).push(row);
  });

  let currentDate = null;
  let chargedTodayRaw = 0;

  const actions = [];

  rows.forEach((row) => {
    const date = String(row.date);

    if (currentDate !== date) {
      currentDate = date;
      chargedTodayRaw = 0;
    }

    const durationH = parseContractHours(row.contract);

    // Gate closure check 1: user-configured duration filter
    if (durationH < costs.gateClosureHours) {
      return;
    }

    // Gate closure check 2: real Nord Pool market rules
    if (!isContractEligibleByMarketGate(row)) {
      return;
    }

    const stepRawLimit = inputs.powerMw * durationH;
    const dailyChargeBudgetRaw = inputs.dailyCycleLimit * inputs.capacityMWh;
    const dailyChargeRemaining = Math.max(0, dailyChargeBudgetRaw - chargedTodayRaw);

    const buyPrice = Number(row.buy_price);
    const sellPrice = Number(row.sell_price);

    if (!Number.isFinite(buyPrice) || !Number.isFinite(sellPrice)) return;

    // Get all sell prices for this date to compute median
    const dateRows = byDate.get(date) || [];
    const sellPrices = dateRows
      .map((r) => Number(r.sell_price))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const medianSell = sellPrices.length > 0 ? sellPrices[Math.floor(sellPrices.length / 2)] : 0;

    let action = "idle";
    let energyRaw = 0;
    let pnlDelta = 0;

    // Greedy: charge at cheapest prices where buyPrice < medianSell * eta^2
    const shouldCharge =
      buyPrice < medianSell * Math.pow(eta, 2) &&
      soc < maxSocMWh &&
      dailyChargeRemaining > 0;

    // Discharge at most expensive prices
    const shouldDischarge =
      sellPrice >= medianSell && soc > minSocMWh;

    if (shouldCharge) {
      const socRoomRaw = (maxSocMWh - soc) / eta;
      let chargeRaw = Math.max(0, Math.min(stepRawLimit, dailyChargeRemaining, socRoomRaw));
      chargeRaw = Math.round(chargeRaw / costs.lotSizeMw) * costs.lotSizeMw;

      if (chargeRaw > 0) {
        const adjustedBuyPrice = buyPrice * (1 + costs.slippage) + costs.bidAskSpread / 2;
        const actionCosts = chargeRaw * costs.transactionCost + chargeRaw * costs.wearCost;

        soc += chargeRaw * eta;
        chargedTodayRaw += chargeRaw;
        chargeEnergyRaw += chargeRaw;
        pnlDelta = -(chargeRaw * adjustedBuyPrice + actionCosts);
        totalPnL += pnlDelta;
        action = "charge";
        energyRaw = chargeRaw;
      }
    } else if (shouldDischarge) {
      const availableRaw = Math.max(0, soc - minSocMWh);
      let dischargeRaw = Math.min(stepRawLimit, availableRaw);
      dischargeRaw = Math.round(dischargeRaw / costs.lotSizeMw) * costs.lotSizeMw;

      if (dischargeRaw > 0) {
        const adjustedSellPrice = sellPrice * (1 - costs.slippage) - costs.bidAskSpread / 2;
        const actionCosts = dischargeRaw * costs.transactionCost + dischargeRaw * costs.wearCost;
        const delivered = dischargeRaw * eta;

        soc -= dischargeRaw;
        dischargeEnergyRaw += dischargeRaw;
        pnlDelta = delivered * adjustedSellPrice - actionCosts;
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
      soc_after: soc,
    });
  });

  return {
    totalPnL,
    chargeEnergyRaw,
    dischargeEnergyRaw,
    endingSoc: soc,
    actions,
  };
}

function runForecastDrivenBacktest(rows, inputs, costs) {
  if (!rows.length) return null;

  const eta = Math.sqrt(inputs.efficiency);
  const minSocMWh = inputs.capacityMWh * (inputs.minSoc / 100);
  const maxSocMWh = inputs.capacityMWh * (inputs.maxSoc / 100);

  let soc = (minSocMWh + maxSocMWh) / 2;
  let totalPnL = 0;
  let chargeEnergyRaw = 0;
  let dischargeEnergyRaw = 0;
  let currentDate = null;
  let chargedTodayRaw = 0;

  const historicalBuyPrices = [];
  const historicalSellPrices = [];
  const actions = [];

  rows.forEach((row) => {
    const date = String(row.date);

    if (currentDate !== date) {
      currentDate = date;
      chargedTodayRaw = 0;
    }

    const durationH = parseContractHours(row.contract);

    // Gate closure check 1: user-configured duration filter
    if (durationH < costs.gateClosureHours) {
      return;
    }

    // Gate closure check 2: real Nord Pool market rules
    if (!isContractEligibleByMarketGate(row)) {
      return;
    }

    const stepRawLimit = inputs.powerMw * durationH;
    const dailyChargeBudgetRaw = inputs.dailyCycleLimit * inputs.capacityMWh;
    const dailyChargeRemaining = Math.max(0, dailyChargeBudgetRaw - chargedTodayRaw);

    const buyPrice = Number(row.buy_price);
    const sellPrice = Number(row.sell_price);

    if (!Number.isFinite(buyPrice) || !Number.isFinite(sellPrice)) return;

    // 24-period MA forecast
    const buyForecast =
      historicalBuyPrices.length >= 24
        ? historicalBuyPrices.slice(-24).reduce((a, b) => a + b, 0) / 24
        : historicalBuyPrices.length > 0
        ? historicalBuyPrices.reduce((a, b) => a + b, 0) / historicalBuyPrices.length
        : 0;

    const sellForecast =
      historicalSellPrices.length >= 24
        ? historicalSellPrices.slice(-24).reduce((a, b) => a + b, 0) / 24
        : historicalSellPrices.length > 0
        ? historicalSellPrices.reduce((a, b) => a + b, 0) / historicalSellPrices.length
        : 0;

    const buyStdDev = standardDeviation(
      historicalBuyPrices.length >= 24 ? historicalBuyPrices.slice(-24) : historicalBuyPrices
    );
    const sellStdDev = standardDeviation(
      historicalSellPrices.length >= 24 ? historicalSellPrices.slice(-24) : historicalSellPrices
    );

    let action = "idle";
    let energyRaw = 0;
    let pnlDelta = 0;

    // Charge when price < forecast - 1 stddev
    const shouldCharge =
      inputs.strategyMode !== "discharge_only" &&
      buyPrice < buyForecast - buyStdDev &&
      soc < maxSocMWh &&
      dailyChargeRemaining > 0;

    // Discharge when price > forecast + 1 stddev
    const shouldDischarge =
      inputs.strategyMode !== "charge_only" &&
      sellPrice > sellForecast + sellStdDev &&
      soc > minSocMWh;

    if (shouldCharge) {
      const socRoomRaw = (maxSocMWh - soc) / eta;
      let chargeRaw = Math.max(0, Math.min(stepRawLimit, dailyChargeRemaining, socRoomRaw));
      chargeRaw = Math.round(chargeRaw / costs.lotSizeMw) * costs.lotSizeMw;

      if (chargeRaw > 0) {
        const adjustedBuyPrice = buyPrice * (1 + costs.slippage) + costs.bidAskSpread / 2;
        const actionCosts = chargeRaw * costs.transactionCost + chargeRaw * costs.wearCost;

        soc += chargeRaw * eta;
        chargedTodayRaw += chargeRaw;
        chargeEnergyRaw += chargeRaw;
        pnlDelta = -(chargeRaw * adjustedBuyPrice + actionCosts);
        totalPnL += pnlDelta;
        action = "charge";
        energyRaw = chargeRaw;
      }
    } else if (shouldDischarge) {
      const availableRaw = Math.max(0, soc - minSocMWh);
      let dischargeRaw = Math.min(stepRawLimit, availableRaw);
      dischargeRaw = Math.round(dischargeRaw / costs.lotSizeMw) * costs.lotSizeMw;

      if (dischargeRaw > 0) {
        const adjustedSellPrice = sellPrice * (1 - costs.slippage) - costs.bidAskSpread / 2;
        const actionCosts = dischargeRaw * costs.transactionCost + dischargeRaw * costs.wearCost;
        const delivered = dischargeRaw * eta;

        soc -= dischargeRaw;
        dischargeEnergyRaw += dischargeRaw;
        pnlDelta = delivered * adjustedSellPrice - actionCosts;
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
      soc_after: soc,
    });

    historicalBuyPrices.push(buyPrice);
    historicalSellPrices.push(sellPrice);
  });

  return {
    totalPnL,
    chargeEnergyRaw,
    dischargeEnergyRaw,
    endingSoc: soc,
    actions,
  };
}

function summarizeDailyFromActions(actions) {
  const map = new Map();

  actions.forEach((a) => {
    const key = a.date;

    if (!map.has(key)) {
      map.set(key, {
        date: key,
        pnl: 0,
        chargeEnergy: 0,
        dischargeEnergy: 0,
        charges: 0,
        discharges: 0,
        lastSoc: a.soc_after,
      });
    }

    const row = map.get(key);
    row.pnl += a.pnl_delta;
    row.lastSoc = a.soc_after;

    if (a.action === "charge") {
      row.chargeEnergy += a.energy_raw_mwh;
      row.charges += 1;
    } else if (a.action === "discharge") {
      row.dischargeEnergy += a.energy_raw_mwh;
      row.discharges += 1;
    }
  });

  return [...map.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function runOptimizerBacktest(trainRows, valRows, testRows, inputs) {
  const costs = inputs.costs;
  const eligibleRules = getEligibleRules(trainRows, inputs.markets);
  const candidates = buildCandidateStrategies(inputs.strategyMode);
  const allResults = [];
  const trainResults = [];

  // Phase 1: Train on training set
  eligibleRules.forEach((rule) => {
    const ruleTrainRows = trainRows.filter((r) => r.rule === rule);
    if (!ruleTrainRows.length) return;

    candidates.forEach((candidate) => {
      const result = runSingleBacktest(ruleTrainRows, inputs, candidate, rule, costs);
      if (result) {
        trainResults.push(result);
        allResults.push(result);
      }
    });
  });

  if (!trainResults.length) {
    return { best: null, allResults: [], eligibleRules: [], trainResults: [], valResult: null, testResult: null, perfectForesight: null, forecastDrivenTrain: null, forecastDrivenVal: null, forecastDrivenTest: null };
  }

  // Sort by training P&L and select best
  trainResults.sort((a, b) => b.totalPnL - a.totalPnL);
  const bestFromTraining = trainResults[0];

  // Phase 2: Validate on validation set with same rule+candidate
  const valRuleRows = valRows.filter((r) => r.rule === bestFromTraining.rule);
  const valResult = valRuleRows.length > 0
    ? runSingleBacktest(valRuleRows, inputs, bestFromTraining.candidate, bestFromTraining.rule, costs)
    : null;

  // Phase 3: Test on test set with same rule+candidate
  const testRuleRows = testRows.filter((r) => r.rule === bestFromTraining.rule);
  const testResult = testRuleRows.length > 0
    ? runSingleBacktest(testRuleRows, inputs, bestFromTraining.candidate, bestFromTraining.rule, costs)
    : null;

  // Phase 3 Optimization: Perfect Foresight
  const perfectForesight = runPerfectForesightBacktest(testRows, inputs, costs);

  // Phase 3 Optimization: Forecast-Driven
  const forecastDrivenTrain = runForecastDrivenBacktest(trainRows, inputs, costs);
  const forecastDrivenVal = runForecastDrivenBacktest(valRows, inputs, costs);
  const forecastDrivenTest = runForecastDrivenBacktest(testRows, inputs, costs);

  return {
    best: testResult || bestFromTraining,
    allResults,
    eligibleRules,
    trainResults,
    trainPnL: bestFromTraining.totalPnL,
    valResult,
    testResult,
    perfectForesight,
    forecastDrivenTrain,
    forecastDrivenVal,
    forecastDrivenTest,
  };
}

function formatMode(mode) {
  if (mode === "charge_only") return "Charge only";
  if (mode === "discharge_only") return "Discharge only";
  if (mode === "arbitrage") return "Quant / Buy low, sell high";
  return "Charge + Discharge";
}

function renderActionTable(actions) {
  if (!actions.length) {
    return `<p>No charge/discharge actions were triggered by the selected strategy.</p>`;
  }

  const rows = actions
    .slice(0, 20)
    .map(
      (a) => `
        <tr>
          <td>${escapeHtml(a.date)}</td>
          <td>${escapeHtml(a.contract)}</td>
          <td>${escapeHtml(a.action)}</td>
          <td>${formatNumber(a.buy_price)}</td>
          <td>${formatNumber(a.sell_price)}</td>
          <td>${formatNumber(a.energy_raw_mwh)}</td>
          <td>${formatNumber(a.pnl_delta)}</td>
          <td>${formatNumber(a.soc_after)}</td>
        </tr>
      `
    )
    .join("");

  return `
    <div style="overflow-x:auto;">
      <table class="optimizer-table" style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th>Date</th>
            <th>Contract</th>
            <th>Action</th>
            <th>Buy</th>
            <th>Sell</th>
            <th>Energy</th>
            <th>P&amp;L Î</th>
            <th>SoC After</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderTopAlternatives(allResults) {
  if (!allResults.length) {
    return `<p>No alternatives found.</p>`;
  }

  const rows = allResults
    .slice(0, 5)
    .map(
      (r, idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td>${escapeHtml(RULE_LABELS[r.rule] || r.rule)}</td>
          <td>${escapeHtml(formatMode(r.candidate.mode))}</td>
          <td>${r.candidate.buyQ !== null ? `${(r.candidate.buyQ * 100).toFixed(0)}%` : "-"}</td>
          <td>${r.candidate.sellQ !== null ? `${(r.candidate.sellQ * 100).toFixed(0)}%` : "-"}</td>
          <td>${formatNumber(r.totalPnL)} â¬</td>
        </tr>
      `
    )
    .join("");

  return `
    <div style="overflow-x:auto;">
      <table class="optimizer-table" style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Market Pair</th>
            <th>Mode</th>
            <th>Buy Q</th>
            <th>Sell Q</th>
            <th>Total P&amp;L</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function summarizeDaily(bestResult) {
  const map = new Map();

  bestResult.actions.forEach((a) => {
    const key = a.date;

    if (!map.has(key)) {
      map.set(key, {
        date: key,
        pnl: 0,
        chargeEnergy: 0,
        dischargeEnergy: 0,
        charges: 0,
        discharges: 0,
        lastSoc: a.soc_after,
      });
    }

    const row = map.get(key);
    row.pnl += a.pnl_delta;
    row.lastSoc = a.soc_after;

    if (a.action === "charge") {
      row.chargeEnergy += a.energy_raw_mwh;
      row.charges += 1;
    } else if (a.action === "discharge") {
      row.dischargeEnergy += a.energy_raw_mwh;
      row.discharges += 1;
    }
  });

  return [...map.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function renderDailySummaryTable(dailyRows) {
  const el = byId("optimizerDailySummary");
  if (!el) return;

  if (!dailyRows.length) {
    el.innerHTML = `<p>No daily summary available.</p>`;
    return;
  }

  const rows = dailyRows
    .map(
      (r) => `
        <tr>
          <td>${escapeHtml(r.date)}</td>
          <td>${formatNumber(r.pnl)}</td>
          <td>${formatNumber(r.chargeEnergy)}</td>
          <td>${formatNumber(r.dischargeEnergy)}</td>
          <td>${r.charges}</td>
          <td>${r.discharges}</td>
          <td>${formatNumber(r.lastSoc)}</td>
        </tr>
      `
    )
    .join("");

  el.innerHTML = `
    <div style="overflow-x:auto;">
      <table class="optimizer-table" style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th>Date</th>
            <th>Daily P&amp;L</th>
            <th>Charge Energy</th>
            <th>Discharge Energy</th>
            <th>Charge Actions</th>
            <th>Discharge Actions</th>
            <th>Ending SoC</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderStrategyComparison(run) {
  if (!run.testResult) return "";

  const quantileTestPnL = run.testResult.totalPnL;
  const perfectForesightTestPnL = run.perfectForesight?.totalPnL || 0;
  const forecastDrivenTestPnL = run.forecastDrivenTest?.totalPnL || 0;

  const captureRatio =
    perfectForesightTestPnL !== 0
      ? (quantileTestPnL / perfectForesightTestPnL) * 100
      : 0;

  const comparisonRows = [
    {
      strategy: "Quantile Baseline",
      trainPnL: run.trainPnL || 0,
      valPnL: run.valResult?.totalPnL || 0,
      testPnL: quantileTestPnL,
      captureRatio,
    },
    {
      strategy: "Perfect Foresight",
      trainPnL: 0,
      valPnL: 0,
      testPnL: perfectForesightTestPnL,
      captureRatio: 100,
    },
    {
      strategy: "Forecast-Driven",
      trainPnL: run.forecastDrivenTrain?.totalPnL || 0,
      valPnL: run.forecastDrivenVal?.totalPnL || 0,
      testPnL: forecastDrivenTestPnL,
      captureRatio: perfectForesightTestPnL !== 0 ? (forecastDrivenTestPnL / perfectForesightTestPnL) * 100 : 0,
    },
  ];

  const rows = comparisonRows
    .map(
      (r) => `
        <tr>
          <td>${escapeHtml(r.strategy)}</td>
          <td>${formatNumber(r.trainPnL)} â¬</td>
          <td>${formatNumber(r.valPnL)} â¬</td>
          <td>${formatNumber(r.testPnL)} â¬</td>
          <td>${formatNumber(r.captureRatio)} %</td>
        </tr>
      `
    )
    .join("");

  return `
    <h3>Strategy Comparison (Train / Validation / Out-of-Sample Test)</h3>
    <div style="overflow-x:auto;">
      <table class="optimizer-table" style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Train P&amp;L</th>
            <th>Validation P&amp;L</th>
            <th>Test P&amp;L</th>
            <th>Capture Ratio</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderOptimizerCharts(run) {
  if (typeof Plotly === "undefined" || !run.testResult) return;

  const bestResult = run.testResult;
  const dailyRows = summarizeDaily(bestResult);
  const dailyDates = dailyRows.map((r) => r.date);
  const dailyPnL = dailyRows.map((r) => r.pnl);

  const cumulative = [];
  dailyPnL.reduce((acc, val, i) => {
    cumulative[i] = acc + val;
    return cumulative[i];
  }, 0);

  const stepLabels = bestResult.actions.map((a, i) => `${a.date} | ${a.contract} | ${i + 1}`);
  const socTrace = bestResult.actions.map((a) => a.soc_after);

  // Get min/max SoC
  const minSocMWh = bestResult.actions.length > 0
    ? Math.min(...bestResult.actions.map((a) => a.soc_after))
    : 0;
  const maxSocMWh = bestResult.actions.length > 0
    ? Math.max(...bestResult.actions.map((a) => a.soc_after))
    : 0;

  // Daily P&L bar chart
  Plotly.newPlot(
    "optimizerDailyPnL",
    [
      {
        x: dailyDates,
        y: dailyPnL,
        type: "bar",
        hovertemplate: "Date: %{x}<br>Daily P&L: %{y:.2f} â¬<extra></extra>",
      },
    ],
    {
      title: "Daily P&L (Test Set)",
      margin: { l: 60, r: 20, t: 50, b: 90 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      xaxis: { title: "Date", tickangle: -45, automargin: true },
      yaxis: { title: "P&L (â¬)", gridcolor: "#eaecf0", automargin: true },
    },
    { responsive: true, displayModeBar: false }
  );

  // Cumulative P&L comparison
  const quantileCumulative = [];
  let quantileAcc = 0;
  dailyRows.forEach((r) => {
    quantileAcc += r.pnl;
    quantileCumulative.push(quantileAcc);
  });

  const pfDailyRows = run.perfectForesight && run.perfectForesight.actions
    ? summarizeDailyFromActions(run.perfectForesight.actions)
    : [];
  const pfCumulative = [];
  let pfAcc = 0;
  pfDailyRows.forEach((r) => {
    pfAcc += r.pnl;
    pfCumulative.push(pfAcc);
  });

  const fdDailyRows = run.forecastDrivenTest && run.forecastDrivenTest.actions
    ? summarizeDailyFromActions(run.forecastDrivenTest.actions)
    : [];
  const fdCumulative = [];
  let fdAcc = 0;
  fdDailyRows.forEach((r) => {
    fdAcc += r.pnl;
    fdCumulative.push(fdAcc);
  });

  const cumulativeTraces = [
    {
      x: dailyDates,
      y: quantileCumulative,
      mode: "lines+markers",
      name: "Quantile Baseline",
      hovertemplate: "Date: %{x}<br>Cumulative P&L: %{y:.2f} â¬<extra></extra>",
    },
  ];

  if (pfDailyRows.length > 0) {
    cumulativeTraces.push({
      x: pfDailyRows.map((r) => r.date),
      y: pfCumulative,
      mode: "lines+markers",
      name: "Perfect Foresight",
      hovertemplate: "Date: %{x}<br>Cumulative P&L: %{y:.2f} â¬<extra></extra>",
    });
  }

  if (fdDailyRows.length > 0) {
    cumulativeTraces.push({
      x: fdDailyRows.map((r) => r.date),
      y: fdCumulative,
      mode: "lines+markers",
      name: "Forecast-Driven",
      hovertemplate: "Date: %{x}<br>Cumulative P&L: %{y:.2f} â¬<extra></extra>",
    });
  }

  Plotly.newPlot(
    "optimizerCumulativePnL",
    cumulativeTraces,
    {
      title: "Cumulative P&L Comparison",
      margin: { l: 60, r: 20, t: 50, b: 90 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      xaxis: { title: "Date", tickangle: -45, automargin: true },
      yaxis: { title: "Cumulative (â¬)", gridcolor: "#eaecf0", automargin: true },
    },
    { responsive: true, displayModeBar: false }
  );

  // SoC trace with min/max reference lines
  const socTraces = [
    {
      x: stepLabels,
      y: socTrace,
      mode: "lines",
      customdata: bestResult.actions.map((a) => `${a.date} | ${a.contract} | ${a.action}`),
      hovertemplate: "%{customdata}<br>SoC: %{y:.2f} MWh<extra></extra>",
      name: "SoC",
    },
    {
      x: [stepLabels[0], stepLabels[stepLabels.length - 1]],
      y: [minSocMWh, minSocMWh],
      mode: "lines",
      name: "Min SoC",
      line: { color: "red", dash: "dash" },
    },
    {
      x: [stepLabels[0], stepLabels[stepLabels.length - 1]],
      y: [maxSocMWh, maxSocMWh],
      mode: "lines",
      name: "Max SoC",
      line: { color: "green", dash: "dash" },
    },
  ];

  Plotly.newPlot(
    "optimizerSocTrace",
    socTraces,
    {
      title: "SoC Trace (Test Set)",
      margin: { l: 60, r: 20, t: 50, b: 110 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      xaxis: { title: "Step", showticklabels: false, automargin: true },
      yaxis: { title: "SoC (MWh)", gridcolor: "#eaecf0", automargin: true },
    },
    { responsive: true, displayModeBar: false }
  );

  // Price Signal Chart
  if (run.testResult && run.testResult.actions && run.testResult.actions.length > 0) {
    const priceActions = run.testResult.actions;
    const stepIndices = priceActions.map((_, i) => i);

    const chargeMarkers = [];
    const dischargeMarkers = [];

    priceActions.forEach((action, i) => {
      if (action.action === "charge") {
        chargeMarkers.push({ step: i, price: action.buy_price });
      } else if (action.action === "discharge") {
        dischargeMarkers.push({ step: i, price: action.sell_price });
      }
    });

    const priceTraces = [
      {
        x: stepIndices,
        y: priceActions.map((a) => a.buy_price),
        mode: "lines",
        name: "Buy Prices",
        line: { color: "lightblue" },
        hovertemplate: "Step %{x}<br>Buy: â¬%{y:.2f}<extra></extra>",
      },
      {
        x: stepIndices,
        y: priceActions.map((a) => a.sell_price),
        mode: "lines",
        name: "Sell Prices",
        line: { color: "orange" },
        hovertemplate: "Step %{x}<br>Sell: â¬%{y:.2f}<extra></extra>",
      },
    ];

    if (chargeMarkers.length > 0) {
      priceTraces.push({
        x: chargeMarkers.map((m) => m.step),
        y: chargeMarkers.map((m) => m.price),
        mode: "markers",
        name: "Charge",
        marker: { symbol: "triangle-up", color: "green", size: 10 },
        hovertemplate: "Charge at â¬%{y:.2f}<extra></extra>",
      });
    }

    if (dischargeMarkers.length > 0) {
      priceTraces.push({
        x: dischargeMarkers.map((m) => m.step),
        y: dischargeMarkers.map((m) => m.price),
        mode: "markers",
        name: "Discharge",
        marker: { symbol: "triangle-down", color: "red", size: 10 },
        hovertemplate: "Discharge at â¬%{y:.2f}<extra></extra>",
      });
    }

    const priceEl = byId("optimizerPriceSignal");
    if (priceEl) {
      Plotly.newPlot(
        "optimizerPriceSignal",
        priceTraces,
        {
          title: "Price History with Trade Signals (Test Set)",
          margin: { l: 60, r: 20, t: 50, b: 90 },
          paper_bgcolor: "white",
          plot_bgcolor: "white",
          xaxis: { title: "Step", showticklabels: false, automargin: true },
          yaxis: { title: "Price (â¬/MWh)", gridcolor: "#eaecf0", automargin: true },
        },
        { responsive: true, displayModeBar: false }
      );
    }
  }

  // Strategy P&L by Phase Bar Chart
  const strategyBarEl = byId("optimizerStrategyBars");
  if (strategyBarEl && run.testResult) {
    const strategies = ["Quantile Baseline", "Perfect Foresight", "Forecast-Driven"];
    const trainPnLs = [
      run.trainPnL || 0,
      0,
      run.forecastDrivenTrain?.totalPnL || 0,
    ];
    const valPnLs = [
      run.valResult?.totalPnL || 0,
      0,
      run.forecastDrivenVal?.totalPnL || 0,
    ];
    const testPnLs = [
      run.testResult.totalPnL,
      run.perfectForesight?.totalPnL || 0,
      run.forecastDrivenTest?.totalPnL || 0,
    ];

    const strategyBarTraces = [
      {
        x: strategies,
        y: trainPnLs,
        name: "Train P&L",
        type: "bar",
      },
      {
        x: strategies,
        y: valPnLs,
        name: "Val P&L",
        type: "bar",
      },
      {
        x: strategies,
        y: testPnLs,
        name: "Test P&L",
        type: "bar",
      },
    ];

    Plotly.newPlot(
      "optimizerStrategyBars",
      strategyBarTraces,
      {
        title: "Strategy P&L by Phase (â¬)",
        barmode: "group",
        margin: { l: 60, r: 20, t: 50, b: 90 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        xaxis: { title: "Strategy" },
        yaxis: { title: "P&L (â¬)", gridcolor: "#eaecf0" },
      },
      { responsive: true, displayModeBar: false }
    );
  }

  // Capture Ratio Bar Chart
  const captureRatioEl = byId("optimizerCaptureRatio");
  if (captureRatioEl && run.testResult && run.perfectForesight) {
    const perfectForesightTestPnL = run.perfectForesight.totalPnL;

    const quantileCR = perfectForesightTestPnL !== 0
      ? (run.testResult.totalPnL / perfectForesightTestPnL) * 100
      : 0;
    const forecastCR = perfectForesightTestPnL !== 0
      ? (run.forecastDrivenTest?.totalPnL || 0) / perfectForesightTestPnL * 100
      : 0;

    const captureRatioTraces = [
      {
        y: ["Quantile Baseline", "Forecast-Driven", "Perfect Foresight"],
        x: [quantileCR, forecastCR, 100],
        type: "bar",
        orientation: "h",
        marker: {
          color: [
            quantileCR >= 60 ? "green" : quantileCR >= 30 ? "yellow" : "red",
            forecastCR >= 60 ? "green" : forecastCR >= 30 ? "yellow" : "red",
            "blue",
          ],
        },
      },
    ];

    Plotly.newPlot(
      "optimizerCaptureRatio",
      captureRatioTraces,
      {
        title: "Capture Ratio vs Perfect Foresight (%)",
        margin: { l: 150, r: 20, t: 50, b: 50 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        xaxis: { title: "Capture Ratio (%)", gridcolor: "#eaecf0" },
      },
      { responsive: true, displayModeBar: false }
    );
  }

  renderDailySummaryTable(dailyRows);
}

function clearOptimizerCharts() {
  ["optimizerDailyPnL", "optimizerCumulativePnL", "optimizerSocTrace", "optimizerPriceSignal", "optimizerStrategyBars", "optimizerCaptureRatio"].forEach((id) => {
    const el = byId(id);
    if (!el || typeof Plotly === "undefined") return;

    Plotly.newPlot(
      id,
      [],
      {
        annotations: [{ text: "No data", showarrow: false }],
        margin: { l: 40, r: 20, t: 40, b: 40 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      },
      { responsive: true, displayModeBar: false }
    );
  });

  const summaryEl = byId("optimizerDailySummary");
  if (summaryEl) {
    summaryEl.innerHTML = `<p>No daily summary yet.</p>`;
  }
}

function renderBacktestResult(inputs, scopedRows, run) {
  const resultEl = byId("optimizerResult");
  if (!resultEl) return;

  if (!run.best) {
    resultEl.innerHTML = `
      <div class="optimizer-card">
        <strong>No valid strategy found</strong><br />
        No eligible historical market pair exists inside the selected markets and date range.
      </div>
    `;
    clearOptimizerCharts();
    return;
  }

  const best = run.best;

  const strategyNote =
    inputs.strategyMode === "charge_only"
      ? `Best pair: ${RULE_LABELS[best.rule] || best.rule}. Charge when buy price is in the lowest ${(best.candidate.buyQ * 100).toFixed(0)}% of prior history for this pair.`
      : inputs.strategyMode === "discharge_only"
      ? `Best pair: ${RULE_LABELS[best.rule] || best.rule}. Discharge when sell price is in the highest ${(100 - best.candidate.sellQ * 100).toFixed(0)}% tail of prior history for this pair.`
      : `Best pair: ${RULE_LABELS[best.rule] || best.rule}. Charge below ${best.buyThreshold?.toFixed(2) ?? "-"} â¬/MWh and discharge above ${best.sellThreshold?.toFixed(2) ?? "-"} â¬/MWh using walk-forward thresholds.`;

  const trainPnL = run.trainResults.length > 0 ? run.trainResults[0].totalPnL : 0;
  const valPnL = run.valResult?.totalPnL || 0;
  const testPnL = run.testResult?.totalPnL || 0;

  resultEl.innerHTML = `
    <div class="optimizer-card">
      <h3>Recommended strategy</h3>
      <p><strong>Mode:</strong> ${escapeHtml(formatMode(inputs.strategyMode))}</p>
      <p><strong>Recommended market pair:</strong> ${escapeHtml(RULE_LABELS[best.rule] || best.rule)}</p>
      <p><strong>Markets selected:</strong> ${escapeHtml(inputs.markets.join(", "))}</p>
      <p><strong>Scope:</strong> ${escapeHtml(inputs.area)} | ${escapeHtml(inputs.startDate)} â ${escapeHtml(inputs.endDate)}</p>
      <p><strong>Data split:</strong> 60% training, 20% validation, 20% out-of-sample test</p>
      <p>${escapeHtml(strategyNote)}</p>
      <p><strong>Scoped rows:</strong> ${scopedRows.length}</p>
      <p><strong>Eligible market pairs:</strong> ${run.eligibleRules.length}</p>
      <p><strong>Training P&L:</strong> ${formatNumber(trainPnL)} â¬</p>
      <p><strong>Validation P&L:</strong> ${formatNumber(valPnL)} â¬</p>
      <p><strong>Test P&L (out-of-sample):</strong> ${formatNumber(testPnL)} â¬</p>
      <p><strong>Charge actions:</strong> ${best.chargeActions}</p>
      <p><strong>Discharge actions:</strong> ${best.dischargeActions}</p>
      <p><strong>Charged energy:</strong> ${formatNumber(best.chargeEnergyRaw)} MWh</p>
      <p><strong>Discharged energy:</strong> ${formatNumber(best.dischargeEnergyRaw)} MWh</p>
      <p><strong>Equivalent charge cycles:</strong> ${formatNumber(best.equivalentCycles)}</p>
      <p><strong>Ending SoC:</strong> ${formatNumber(best.endingSoc)} MWh</p>

      ${renderStrategyComparison(run)}

      <h3>Top 5 strategy alternatives (from training)</h3>
      ${renderTopAlternatives(run.trainResults)}

      <h3>First 20 active actions of recommended strategy (test set)</h3>
      ${renderActionTable(best.activeActions)}
    </div>
  `;

  renderOptimizerCharts(run);
}

function handleRunOptimizer() {
  const inputs = getOptimizerInputs();
  const error = validateInputs(inputs);

  if (error) {
    const resultEl = byId("optimizerResult");
    if (resultEl) {
      resultEl.innerHTML = `
        <div class="optimizer-card">
          <strong>Input error</strong><br />
          ${escapeHtml(error)}
        </div>
      `;
    }
    clearOptimizerCharts();
    return;
  }

  const scopedRows = getScopedRows(inputs);

  if (!scopedRows.length) {
    const resultEl = byId("optimizerResult");
    if (resultEl) {
      resultEl.innerHTML = `
        <div class="optimizer-card">
          <strong>No scoped data</strong><br />
          No historical rows match the selected area and date range.
        </div>
      `;
    }
    clearOptimizerCharts();
    return;
  }

  const split = splitTrainValTest(scopedRows);
  const run = runOptimizerBacktest(split.trainRows, split.valRows, split.testRows, inputs);
  renderBacktestResult(inputs, scopedRows, run);
}

byId("runOptimizerBtn")?.addEventListener("click", handleRunOptimizer);

loadOptimizerData();
clearOptimizerCharts();
console.log("Advanced BESS Optimizer v2 loaded â 3-phase backtest, realistic market costs, real Nord Pool gate closure rules (IDA3 â¥12:00), and enhanced charting.");
