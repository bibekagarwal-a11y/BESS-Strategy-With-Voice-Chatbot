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
  return null;
}

function getScopedRows(inputs) {
  return optimizerData.filter(row => {
    if (inputs.area && row.area !== inputs.area) return false;
    if (inputs.referenceRule && row.rule !== inputs.referenceRule) return false;
    if (inputs.startDate && String(row.date ?? "") < inputs.startDate) return false;
    if (inputs.endDate && String(row.date ?? "") > inputs.endDate) return false;
    return true;
  });
}

function renderMockResult(inputs) {
  const resultEl = byId("optimizerResult");
  if (!resultEl) return;

  const scopedRows = getScopedRows(inputs);
  const dates = unique(scopedRows.map(x => x.date)).filter(Boolean).sort();
  const contracts = unique(scopedRows.map(x => x.contract)).filter(Boolean).sort();

  resultEl.innerHTML = `
    <div class="optimizer-result-card">
      <div class="optimizer-result-title">Step 3 dataset-aware mock result</div>
      <div>This page is now reading your historical optimizer data. No real optimization has been run yet.</div>

      <div class="optimizer-result-grid">
        <div><strong>Area:</strong> ${inputs.area}</div>
        <div><strong>Reference strategy:</strong> ${RULE_LABELS[inputs.referenceRule] || inputs.referenceRule}</div>
        <div><strong>Mode:</strong> ${inputs.strategyMode}</div>
        <div><strong>Markets:</strong> ${inputs.markets.join(", ")}</div>
        <div><strong>Start date:</strong> ${inputs.startDate}</div>
        <div><strong>End date:</strong> ${inputs.endDate}</div>
        <div><strong>Scoped rows:</strong> ${scopedRows.length}</div>
        <div><strong>Days in range:</strong> ${dates.length}</div>
        <div><strong>Contracts found:</strong> ${contracts.length}</div>
        <div><strong>Power / Capacity:</strong> ${inputs.powerMw} MW / ${inputs.capacityMWh} MWh</div>
      </div>
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

  renderMockResult(inputs);
}

byId("runOptimizerBtn")?.addEventListener("click", handleRunOptimizer);

loadOptimizerData();

console.log("Advanced BESS Optimizer Step 3 loaded.");
