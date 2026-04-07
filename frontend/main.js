import './style.css';
import Chart from 'chart.js/auto';

// --- FRONTEND STATE ---
let state = {
    scanned: 0,
    threats: 0,
    live_history: [],
    radar_chart: null,
    event_source: null
};

// --- INITIALIZE CHARTS ---
const ctxBatch = document.getElementById('live-chart');
if (ctxBatch) {
    state.radar_chart = new Chart(ctxBatch, {
        type: 'doughnut',
        data: {
            labels: ['BENIGN', 'THREAT'],
            datasets: [{
                data: [1, 0],
                backgroundColor: ['#00ff88', '#ff0055'],
                hoverOffset: 4,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            cutout: '75%'
        }
    });
}

// --- API CLIENT ---
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function updateUI(data) {
    state.scanned++;
    if (data.event !== 'BENIGN') state.threats++;

    // Update Stats
    document.getElementById('stat-scanned').innerText = state.scanned;
    document.getElementById('stat-threats').innerText = state.threats;
    document.getElementById('stat-consensus').innerText = data.consensus;
    document.getElementById('stat-latency').innerText = data.latency;

    // Update Stream Log
    const log = document.getElementById('log-stream');
    const row = document.createElement('div');
    row.className = `alert-row ${data.event === 'BENIGN' ? 'benign' : 'malicious'}`;
    row.innerHTML = `
        <span style="opacity: 0.5;">[${data.time}]</span>
        <span style="font-weight: bold; letter-spacing: 0.5px;">${data.event}</span>
        <span style="text-align: right; font-family: monospace;">${data.traffic} pkt/s</span>
        <span style="text-align: right; opacity: 0.7;">${data.status}</span>
    `;
    log.prepend(row);

    // Limit log size
    if (log.children.length > 20) log.lastChild.remove();

    // Update Chart
    state.radar_chart.data.datasets[0].data = [state.scanned - state.threats, state.threats];
    state.radar_chart.update();
}

// --- STREAM CONTROL ---
function startStream(mode = 'stream') {
    if (state.event_source) state.event_source.close();
    
    const endpoint = mode === 'sniff' ? 'monitor/sniff' : 'monitor/stream';
    state.event_source = new EventSource(`${API_URL}/${endpoint}`);
    
    state.event_source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };

    state.event_source.onerror = (err) => {
        console.error("Stream failed: ", err);
        state.event_source.close();
    };
}

// --- NAVIGATION ---
const navItems = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        // Update nav UI
        navItems.forEach(i => i.classList.remove('active'));
        item.classList.add('active');

        // Update Title
        const page = item.id.replace('nav-', '');
        document.getElementById('page-title').innerText = `📡 ${page.toUpperCase().replace('-', ' ')}`;

        // Show/Hide Content
        tabContents.forEach(content => {
            content.style.display = 'none';
        });
        const targetContent = document.getElementById(`content-${page}`);
        if (targetContent) targetContent.style.display = 'block';

        // Specific Tab Logic
        if (page === 'quantum') {
            loadBenchmarks();
        }
    });
});

async function loadBenchmarks() {
    const container = document.getElementById('benchmark-results');
    container.innerHTML = '<div style="color: var(--primary);">LOADING QUANTUM DATA...</div>';
    try {
        const resp = await fetch(`${API_URL}/quantum/benchmarks`);
        const data = await resp.json();
        if (data.models) {
            container.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    ${data.models.map(m => `
                        <div class="glass-card" style="background: rgba(0,0,0,0.2); border: 1px solid rgba(0,242,255,0.1);">
                            <div style="font-size: 0.8rem; color: #888;">${m.model}</div>
                            <div style="font-size: 1.2rem; color: var(--primary); margin-top: 5px;">${m.accuracy}% ACC</div>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            container.innerHTML = '<div style="color: var(--warning);">No benchmark data found. Run a new study via the CLI.</div>';
        }
    } catch (e) {
        container.innerHTML = '<div style="color: var(--danger);">Failed to load benchmarks.</div>';
    }
}

// --- LIVE MONITOR LOGIC ---
let isPlaying = true;
let currentMode = 'stream';

const playPauseBtn = document.getElementById('btn-play-pause');
const modeBtn = document.getElementById('toggle-mode');

playPauseBtn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
        playPauseBtn.innerHTML = '<i class="fas fa-pause"></i> PAUSE';
        playPauseBtn.style.background = 'var(--success)';
        startStream(currentMode);
    } else {
        playPauseBtn.innerHTML = '<i class="fas fa-play"></i> RESUME';
        playPauseBtn.style.background = 'var(--secondary)';
        if (state.event_source) state.event_source.close();
    }
});

modeBtn.addEventListener('click', () => {
    currentMode = currentMode === 'stream' ? 'sniff' : 'stream';
    const isStream = currentMode === 'stream';
    modeBtn.innerText = `MODE: ${isStream ? 'SIM' : 'LIVE'}`;
    
    // Update classes for visual feedback
    modeBtn.classList.remove('mode-sim', 'mode-live');
    modeBtn.classList.add(isStream ? 'mode-sim' : 'mode-live');

    if (isPlaying) startStream(currentMode);
});

startStream();

// --- BATCH AUDIT / DRAG & DROP ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropText = document.getElementById('drop-text');

dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

dropZone.addEventListener('dragover', () => {
    dropZone.style.borderColor = 'var(--primary)';
    dropZone.style.background = 'rgba(0, 242, 255, 0.05)';
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        dropZone.style.background = 'transparent';
    });
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFiles(e.target.files[0]);
});

async function handleFiles(file) {
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        alert('Please upload a CSV file.');
        return;
    }
    
    dropText.innerHTML = `<i class="fas fa-spinner fa-spin"></i> PROCCESSING ${file.name}...`;
    dropZone.style.pointerEvents = 'none';
    dropZone.style.opacity = '0.7';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch(`${API_URL}/predict/batch`, {
            method: 'POST',
            body: formData
        });
        const result = await resp.json();
        
        if (resp.status !== 200) throw new Error(result.detail || 'Batch processing failed');

        dropText.innerHTML = `
            <div style="color: var(--success); font-weight: bold;">✅ AUDIT COMPLETE: ${result.filename}</div>
            <div style="font-size: 0.8rem; margin-top: 10px;">
                Found <span style="color: var(--danger); font-weight: bold;">${result.threats_found}</span> threats 
                out of ${result.total_packets} packets scanned.
            </div>
        `;
        
        // Show results
        renderAuditTable(result.sample_results);
        renderBatchCharts(result.summary);
        document.getElementById('batch-results-container').style.display = 'flex';
        document.getElementById('forensics-section').style.display = 'block';

        // Auto-Forensics for the first row
        fetchForensics(result.sample_results[0], 0);

    } catch (e) {
        console.error("Batch error:", e);
        dropText.innerHTML = `<div style="color: var(--danger);">❌ FAILED: ${e.message}</div>`;
    } finally {
        dropZone.style.pointerEvents = 'auto';
        dropZone.style.opacity = '1';
    }
}

function renderAuditTable(data) {
    const wrapper = document.getElementById('audit-table-wrapper');
    if (!data.length) return;

    const headers = Object.keys(data[0]);
    let html = `<table class="audit-table"><thead><tr>`;
    headers.forEach(h => html += `<th>${h}</th>`);
    html += `</tr></thead><tbody>`;

    data.forEach(row => {
        html += `<tr>`;
        headers.forEach(h => {
            const val = row[h];
            const style = h === 'Detection' ? `color: ${val === 'BENIGN' ? 'var(--success)' : 'var(--danger)'}; font-weight: bold;` : '';
            html += `<td style="${style}">${val}</td>`;
        });
        html += `</tr>`;
    });
    html += `</tbody></table>`;
    wrapper.innerHTML = html;
}

let batchChart = null;
let benchmarkChart = null;

function renderBatchCharts(summary) {
    const ctxStats = document.getElementById('batch-stats-chart');
    const ctxBench = document.getElementById('benchmark-comp-chart');

    if (batchChart) batchChart.destroy();
    if (benchmarkChart) benchmarkChart.destroy();

    // 1. Classification Bar
    batchChart = new Chart(ctxStats, {
        type: 'bar',
        data: {
            labels: Object.keys(summary),
            datasets: [{
                label: 'Detections',
                data: Object.values(summary),
                backgroundColor: ['#00f2ff', '#7000ff', '#00ff88', '#ff0055', '#505050'],
                borderWidth: 0
            }]
        },
        options: {
            scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } } }
        }
    });

    // 2. Benchmarks (Matching original dashboard)
    benchmarkChart = new Chart(ctxBench, {
        type: 'bar',
        data: {
            labels: ["NetSage (Hybrid)", "Random Forest", "CNN-IDS", "Deep-DNN", "SVM-Radial"],
            datasets: [{
                label: 'Accuracy %',
                data: [99.7, 98.2, 97.5, 96.8, 94.2],
                backgroundColor: 'rgba(0, 242, 255, 0.4)',
                borderColor: 'var(--primary)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            scales: { x: { min: 90, max: 100 } }
        }
    });
}

async function fetchForensics(packetData, index) {
    const shapContainer = document.getElementById('shap-container');
    shapContainer.innerHTML = '<div class="glass-card"><i class="fas fa-brain fa-spin"></i> GENERATING SHAP...</div>';
    
    try {
        const resp = await fetch(`${API_URL}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                packet_index: index,
                data_source: 'simulation'
            })
        });
        const result = await resp.json();
        
        // Render SHAP Plot
        shapContainer.innerHTML = `<img src="data:image/png;base64,${result.shap_plot_base64}" style="width: 100%; border-radius: 12px;">`;
        
        // Render LIME Chart
        renderLimeChart(result.lime_chart_data);

    } catch (e) {
        shapContainer.innerHTML = `<div style="color: var(--danger);">Forensics Error: ${e.message}</div>`;
    }
}

let limeChart = null;
function renderLimeChart(data) {
    const ctx = document.getElementById('lime-chart');
    if (limeChart) limeChart.destroy();

    limeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.feature_names,
            datasets: [{
                label: 'LIME Feature weights',
                data: data.importance_values,
                backgroundColor: data.importance_values.map(v => v > 0 ? 'var(--danger)' : 'var(--success)'),
                borderWidth: 0
            }]
        },
        options: { indexAxis: 'y' }
    });
}
