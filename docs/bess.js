// BESS Optimizer Dashboard
// Analysis and visualization of Battery Energy Storage System optimization

// Global state
let bessData = {
    summary: [],
    trades: [],
    schedule: []
};

let filteredData = {
    summary: [],
    trades: [],
    schedule: []
};

let selectors = {};

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Starting BESS application initialization');
    loadBessData();
    setupEventListeners();
    updateLastUpdatedTime();
});

/**
 * Load BESS data from JSON files
 */
function loadBessData() {
    console.log('Loading BESS data from JSON files...');
    
    let loadedCount = 0;
    const totalFiles = 3;
    
    // Load summary data
    fetch('data/bess_summary.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('BESS summary data loaded:', data.length, 'records');
            bessData.summary = data;
            filteredData.summary = [...data];
            loadedCount++;
            checkAllDataLoaded(loadedCount, totalFiles);
        })
        .catch(error => {
            console.error('Error loading BESS summary data:', error);
            showError('Failed to load BESS summary data. Please refresh the page.');
        });
    
    // Load trades data
    fetch('data/bess_trades.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('BESS trades data loaded:', data.length, 'records');
            bessData.trades = data;
            filteredData.trades = [...data];
            loadedCount++;
            checkAllDataLoaded(loadedCount, totalFiles);
        })
        .catch(error => {
            console.error('Error loading BESS trades data:', error);
            showError('Failed to load BESS trades data. Please refresh the page.');
        });
    
    // Load schedule data
    fetch('data/bess_schedule.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('BESS schedule data loaded:', data.length, 'records');
            bessData.schedule = data;
            filteredData.schedule = [...data];
            loadedCount++;
            checkAllDataLoaded(loadedCount, totalFiles);
        })
        .catch(error => {
            console.error('Error loading BESS schedule data:', error);
            showError('Failed to load BESS schedule data. Please refresh the page.');
        });
}

/**
 * Check if all data files have been loaded
 */
function checkAllDataLoaded(loadedCount, totalFiles) {
    if (loadedCount === totalFiles) {
        console.log('All BESS data loaded successfully');
        populateSelectors();
        renderCharts();
        renderTables();
        renderMetrics();
    }
}

/**
 * Populate filter dropdowns with unique values
 */
function populateSelectors() {
    console.log('Populating BESS selectors with options');
    
    const areaSelect = document.getElementById('area');
    const stageSelect = document.getElementById('stage');
    
    if (!areaSelect || !stageSelect) {
        console.warn('Some select elements not found in DOM');
        return;
    }
    
    // Get unique areas from summary data
    const areas = [...new Set(bessData.summary.map(r => r.area))].sort();
    areas.forEach(area => {
        const option = document.createElement('option');
        option.value = area;
        option.textContent = area;
        areaSelect.appendChild(option);
    });
    
    // Get unique stages from summary data
    const stages = [...new Set(bessData.summary.map(r => r.stage))].sort();
    stages.forEach(stage => {
        const option = document.createElement('option');
        option.value = stage;
        option.textContent = stage;
        stageSelect.appendChild(option);
    });
    
    console.log('BESS selectors populated successfully');
}

/**
 * Setup event listeners for filters
 */
function setupEventListeners() {
    console.log('Setting up BESS event listeners');
    
    const areaSelect = document.getElementById('area');
    const stageSelect = document.getElementById('stage');
    
    if (areaSelect) areaSelect.addEventListener('change', applyFilters);
    if (stageSelect) stageSelect.addEventListener('change', applyFilters);
    
    console.log('BESS event listeners setup complete');
}

/**
 * Apply filters to BESS data
 */
function applyFilters() {
    console.log('Applying BESS filters...');
    
    const area = document.getElementById('area')?.value || '';
    const stage = document.getElementById('stage')?.value || '';
    
    filteredData.summary = bessData.summary.filter(record => {
        const areaMatch = !area || record.area === area;
        const stageMatch = !stage || record.stage === stage;
        return areaMatch && stageMatch;
    });
    
    filteredData.trades = bessData.trades.filter(record => {
        const areaMatch = !area || record.area === area;
        return areaMatch;
    });
    
    filteredData.schedule = bessData.schedule.filter(record => {
        const areaMatch = !area || record.area === area;
        return areaMatch;
    });
    
    console.log('BESS filters applied');
    
    renderCharts();
    renderTables();
    renderMetrics();
}

/**
 * Render all charts
 */
function renderCharts() {
    console.log('Rendering BESS charts...');
    
    if (filteredData.summary.length === 0) {
        showEmptyState();
        return;
    }
    
    renderStageProfitChart();
    renderTradeDistributionChart();
    renderScheduleChart();
}

/**
 * Render stage-wise profit chart
 */
function renderStageProfitChart() {
    const container = document.getElementById('stageProfitChart');
    if (!container) return;
    
    // Group by stage and calculate total profit
    const stageMetrics = {};
    
    filteredData.summary.forEach(record => {
        if (!stageMetrics[record.stage]) {
            stageMetrics[record.stage] = {
                totalProfit: 0,
                count: 0
            };
        }
        
        stageMetrics[record.stage].totalProfit += parseFloat(record.pni_eur) || 0;
        stageMetrics[record.stage].count += 1;
    });
    
    const stages = Object.keys(stageMetrics).sort();
    const profits = stages.map(s => stageMetrics[s].totalProfit);
    
    const trace = {
        x: stages,
        y: profits,
        type: 'bar',
        marker: {
            color: profits.map(p => p >= 0 ? '#10b981' : '#dc2626')
        }
    };
    
    const layout = {
        title: '',
        xaxis: { title: 'Stage' },
        yaxis: { title: 'Total Profit (EUR)' },
        plot_bgcolor: '#1a1a2e',
        paper_bgcolor: '#1a1a2e',
        font: { color: '#f1f5f9' },
        margin: { t: 20, b: 40, l: 60, r: 20 }
    };
    
    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

/**
 * Render trade distribution chart
 */
function renderTradeDistributionChart() {
    const container = document.getElementById('tradeDistributionChart');
    if (!container) return;
    
    // Count trades by type
    const tradeTypes = {};
    
    filteredData.trades.forEach(record => {
        const type = record.trade_type || 'Unknown';
        tradeTypes[type] = (tradeTypes[type] || 0) + 1;
    });
    
    const types = Object.keys(tradeTypes);
    const counts = types.map(t => tradeTypes[t]);
    
    const trace = {
        labels: types,
        values: counts,
        type: 'pie',
        marker: {
            colors: ['#2575d4', '#10b981', '#f59e0b', '#dc2626', '#8b5cf6']
        }
    };
    
    const layout = {
        title: '',
        plot_bgcolor: '#1a1a2e',
        paper_bgcolor: '#1a1a2e',
        font: { color: '#f1f5f9' },
        margin: { t: 20, b: 20, l: 20, r: 20 }
    };
    
    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

/**
 * Render schedule timeline chart
 */
function renderScheduleChart() {
    const container = document.getElementById('scheduleChart');
    if (!container) return;
    
    // Sort schedule by date and time
    const sortedSchedule = [...filteredData.schedule].sort((a, b) => {
        const dateA = new Date(`${a.date} ${a.time}`);
        const dateB = new Date(`${b.date} ${b.time}`);
        return dateA - dateB;
    });
    
    const dates = sortedSchedule.map(s => `${s.date} ${s.time}`);
    const actions = sortedSchedule.map(s => s.action === 'CHARGE' ? 1 : -1);
    
    const trace = {
        x: dates,
        y: actions,
        type: 'bar',
        marker: {
            color: actions.map(a => a > 0 ? '#2575d4' : '#f59e0b')
        }
    };
    
    const layout = {
        title: '',
        xaxis: { title: 'Date & Time' },
        yaxis: { title: 'Action (Charge/Discharge)' },
        plot_bgcolor: '#1a1a2e',
        paper_bgcolor: '#1a1a2e',
        font: { color: '#f1f5f9' },
        margin: { t: 20, b: 80, l: 60, r: 20 }
    };
    
    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

/**
 * Render metrics
 */
function renderMetrics() {
    const container = document.getElementById('metricsGrid');
    if (!container) return;
    
    if (filteredData.summary.length === 0) {
        container.innerHTML = '';
        return;
    }
    
    // Calculate metrics
    const totalProfit = filteredData.summary.reduce((sum, r) => sum + (parseFloat(r.pni_eur) || 0), 0);
    const avgProfit = totalProfit / filteredData.summary.length;
    const maxProfit = Math.max(...filteredData.summary.map(r => parseFloat(r.pni_eur) || 0));
    const minProfit = Math.min(...filteredData.summary.map(r => parseFloat(r.pni_eur) || 0));
    
    const html = `
        <div class="metric-card">
            <div class="metric-value">${totalProfit.toFixed(2)}</div>
            <div class="metric-label">Total Profit (EUR)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${avgProfit.toFixed(2)}</div>
            <div class="metric-label">Average Profit</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${maxProfit.toFixed(2)}</div>
            <div class="metric-label">Max Profit</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${minProfit.toFixed(2)}</div>
            <div class="metric-label">Min Profit</div>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Render data tables
 */
function renderTables() {
    renderSummaryTable();
    renderTradesTable();
    renderScheduleTable();
}

/**
 * Render summary table
 */
function renderSummaryTable() {
    const container = document.getElementById('summaryTable');
    if (!container) return;
    
    if (filteredData.summary.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #cbd5e1; padding: 2rem;">No data available.</p>';
        return;
    }
    
    let html = '<table><thead><tr><th>Area</th><th>Stage</th><th>Date</th><th>Profit (EUR)</th></tr></thead><tbody>';
    
    filteredData.summary.slice(0, 20).forEach(record => {
        const profitClass = parseFloat(record.pni_eur) >= 0 ? 'positive' : 'negative';
        html += `<tr>
            <td>${record.area}</td>
            <td>${record.stage}</td>
            <td>${record.date}</td>
            <td class="${profitClass}">${parseFloat(record.pni_eur).toFixed(2)}</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Render trades table
 */
function renderTradesTable() {
    const container = document.getElementById('tradesTable');
    if (!container) return;
    
    if (filteredData.trades.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #cbd5e1; padding: 2rem;">No trades available.</p>';
        return;
    }
    
    let html = '<table><thead><tr><th>Area</th><th>Date</th><th>Time</th><th>Type</th><th>Price</th><th>Volume</th></tr></thead><tbody>';
    
    filteredData.trades.slice(0, 20).forEach(record => {
        html += `<tr>
            <td>${record.area}</td>
            <td>${record.date}</td>
            <td>${record.time}</td>
            <td>${record.trade_type}</td>
            <td>${parseFloat(record.price).toFixed(2)}</td>
            <td>${parseFloat(record.volume).toFixed(2)}</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Render schedule table
 */
function renderScheduleTable() {
    const container = document.getElementById('scheduleTable');
    if (!container) return;
    
    if (filteredData.schedule.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #cbd5e1; padding: 2rem;">No schedule available.</p>';
        return;
    }
    
    let html = '<table><thead><tr><th>Area</th><th>Date</th><th>Time</th><th>Action</th><th>Expected Price</th></tr></thead><tbody>';
    
    filteredData.schedule.slice(0, 20).forEach(record => {
        const actionClass = record.action === 'CHARGE' ? 'positive' : 'negative';
        html += `<tr>
            <td>${record.area}</td>
            <td>${record.date}</td>
            <td>${record.time}</td>
            <td class="${actionClass}">${record.action}</td>
            <td>${parseFloat(record.expected_price).toFixed(2)}</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Show empty state message
 */
function showEmptyState() {
    const charts = ['stageProfitChart', 'tradeDistributionChart', 'scheduleChart'];
    
    charts.forEach(chartId => {
        const container = document.getElementById(chartId);
        if (container) {
            container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-title">No Data Available</div><p>Try adjusting your filters to see results.</p></div>';
        }
    });
}

/**
 * Show error message
 */
function showError(message) {
    const container = document.querySelector('.container');
    if (container) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = '❌ ' + message;
        container.insertBefore(errorDiv, container.firstChild);
    }
}

/**
 * Update last updated time
 */
function updateLastUpdatedTime() {
    const element = document.getElementById('lastUpdated');
    if (element) {
        const now = new Date();
        element.textContent = now.toLocaleString();
    }
}
