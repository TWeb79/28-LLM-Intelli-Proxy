/**
 * IntelliProxy Dashboard JavaScript
 * Handles all frontend functionality for the LLM routing dashboard
 */

// Configuration
const API_URL = `http://${window.location.hostname}:8128`;

// Active nav link tracking
const navLinks = document.querySelectorAll('nav a');
const sections = document.querySelectorAll('.section');

// Initialize intersection observer for navigation
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      navLinks.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === `#${id}`);
      });
    }
  });
}, { threshold: 0.25 });

sections.forEach(s => observer.observe(s));

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
  loadAllData();
  loadConfiguration();
  setInterval(loadAllData, 15000);
});

/**
 * Load configuration from server
 */
async function loadConfiguration() {
  try {
    const [ollamaConfig, nvidiaConfig] = await Promise.all([
      fetch(`${API_URL}/api/config/ollama`).then(r => r.json()),
      fetch(`${API_URL}/api/config/nvidia`).then(r => r.json())
    ]);

    const ollamaHost = document.getElementById('ollama-host');
    const ollamaPort = document.getElementById('ollama-port');
    const nvidiaApiKey = document.getElementById('nvidia-api-key');

    if (!ollamaHost.value) ollamaHost.value = ollamaConfig.host || 'localhost';
    if (!ollamaPort.value) ollamaPort.value = ollamaConfig.port || 11434;
    if (!nvidiaApiKey.value) nvidiaApiKey.value = nvidiaConfig.api_key || '';

  } catch (e) {
    console.error('Failed to load configuration:', e);
    if (!document.getElementById('ollama-host').value) document.getElementById('ollama-host').value = 'localhost';
    if (!document.getElementById('ollama-port').value) document.getElementById('ollama-port').value = 11434;
  }
}

/**
 * Load all dashboard data
 */
async function loadAllData() {
  try {
    const [healthRes, statsRes, modelsRes] = await Promise.all([
      fetch(`${API_URL}/health`),
      fetch(`${API_URL}/stats`),
      fetch(`${API_URL}/models`)
    ]);

    const health = await healthRes.json();
    const stats = await statsRes.json();
    const models = await modelsRes.json();

    // Update status indicators
    document.getElementById('ollama-status').className = `status-dot ${health.ollama?.status === 'running' ? 'healthy' : ''}`;
    document.getElementById('nvidia-status').className = `status-dot ${health.nvidia?.status === 'running' ? 'healthy' : ''}`;
    document.getElementById('router-status').className = `status-dot ${health.overall_status === 'healthy' ? 'healthy' : 'warning'}`;

    // Update metrics
    document.getElementById('total-requests').textContent = stats.requests?.total_requests || 0;
    document.getElementById('models-count').textContent = models.total || 0;
    document.getElementById('cache-hit-rate').textContent = stats.cache?.hit_rate || '0%';
    document.getElementById('system-status').textContent = health.overall_status === 'healthy' ? 'Healthy' : 'Degraded';
    document.getElementById('system-status').className = 'badge ' + (health.overall_status === 'healthy' ? 'badge-green' : 'badge-red');

    // Render tables
    renderModelsTable(models, stats);
    renderFallbacks(await fetch(`${API_URL}/api/config/fallbacks`).then(r => r.json()));

  } catch (e) {
    console.error('Failed to load data:', e);
  }
}

/**
 * Render models table with statistics
 */
function renderModelsTable(models, stats) {
  const tbody = document.getElementById('models-body');
  tbody.innerHTML = '';
  const modelStats = stats.requests?.models || {};
  const modelAvgTimes = stats.requests?.model_avg_times || {};
  const categories = models.categories || {};

  const entries = Object.entries(models.models || {});
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No models found. Click Refresh to discover models.</td></tr>';
    return;
  }

  entries.forEach(([name, data]) => {
    const stat = modelStats[name] || { count: 0 };
    const avgTime = modelAvgTimes[name] || 0;
    const provider = data.provider || 'ollama';
    const category = Object.entries(categories).find(([_, list]) => list.includes(name))?.[0] || 'general';

    const providerBadge = provider === 'nvidia'
      ? '<span class="badge badge-green">nvidia</span>'
      : '<span class="badge badge-blue">ollama</span>';

    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>${name}</strong></td>
      <td>${providerBadge}</td>
      <td><span class="badge badge-yellow">${category}</span></td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">${data.speed || '—'}<span style="color:var(--text-3)">/10</span></td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">${data.complexity || '—'}<span style="color:var(--text-3)">/10</span></td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">${stat.count}</td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">${avgTime}s</td>
    `;
    tbody.appendChild(row);
  });
}

/**
 * Render fallback configuration
 */
function renderFallbacks(fbConfig) {
  const container = document.getElementById('fallback-list');
  const fallbacks = fbConfig.fallbacks || {};

  if (Object.keys(fallbacks).length === 0) {
    container.innerHTML = '<p style="color:var(--text-3);text-align:center;padding:20px;font-size:12px;">No fallback rules configured.</p>';
    return;
  }

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Primary Model</th>
          <th>Fallback Model</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        ${Object.entries(fallbacks).map(([model, fallback]) => `
          <tr>
            <td><strong>${model}</strong></td>
            <td><span class="badge badge-green">${fallback}</span></td>
            <td><button class="btn btn-danger" onclick="removeFallback('${model}')">Remove</button></td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

/**
 * Save Ollama configuration
 */
async function saveOllamaConfig() {
  const host = document.getElementById('ollama-host').value;
  const port = parseInt(document.getElementById('ollama-port').value);
  const statusEl = document.getElementById('ollama-status-text');

  try {
    const res = await fetch(`${API_URL}/api/config/ollama`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ host, port })
    });

    if (res.ok) {
      statusEl.textContent = '✓ Configuration saved successfully';
      statusEl.style.color = 'var(--green)';
      setTimeout(loadAllData, 1000);
    } else {
      throw new Error('Failed to save');
    }
  } catch (e) {
    statusEl.textContent = '✗ Error saving configuration';
    statusEl.style.color = 'var(--red)';
  }
}

/**
 * Save NVIDIA configuration
 */
async function saveNvidiaConfig() {
  const apiKey = document.getElementById('nvidia-api-key').value;
  const statusEl = document.getElementById('nvidia-status');
  const statusElMsg = document.getElementById('nvidia-status-text');


  if (!apiKey.trim()) {
    statusElMsg.textContent = '⚠ Please enter an API key';
    statusElMsg.style.color = 'var(--amber)';
    return;
  }

  statusElMsg.textContent = 'Testing NVIDIA configuration…';
  statusElMsg.style.color = 'var(--text-3)';

  try {
    const res = await fetch(`${API_URL}/api/config/nvidia`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey })
    });

    const data = await res.json();
    if (res.ok) {
      statusElMsg.textContent = `✓ ${data.message || 'NVIDIA configuration saved'}`;
      statusElMsg.style.color = 'var(--green)';
      setTimeout(loadAllData, 1000);
    } else {
      statusElMsg.textContent = `✗ ${data.detail || 'Failed to save configuration'}`;
      statusEl.style.color = 'var(--red)';
    }
  } catch (e) {
    statusElMsg.textContent = '✗ Network error. Please check your connection.';
    statusEl.style.color = 'var(--red)';
  }
}

/**
 * Add fallback rule
 */
async function addFallback() {
  const primary = document.getElementById('fb-primary').value;
  const fallback = document.getElementById('fb-fallback').value;
  const statusElMsg = document.getElementById('fb-status-text');

  if (!primary || !fallback || primary === fallback) {
    statusElMsg.textContent = '⚠ Please select two different models';
    statusElMsg.style.color = 'var(--amber)';
    return;
  }

  try {
    const res = await fetch(`${API_URL}/api/config/fallbacks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fallbacks: { [primary]: fallback } })
    });

    if (res.ok) {
      statusElMsg.textContent = '✓ Fallback rule added';
      statusElMsg.style.color = 'var(--green)';
      loadAllData();
      populateModelSelects();
    } else {
      throw new Error('Failed');
    }
  } catch (e) {
    statusElMsg.textContent = '✗ Error adding fallback rule';
    statusElMsg.style.color = 'var(--red)';
  }
}

/**
 * Remove fallback rule
 */
async function removeFallback(model) {
  if (!confirm(`Remove fallback for "${model}"?`)) return;

  try {
    const res = await fetch(`${API_URL}/api/config/fallbacks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    loadAllData();
  } catch (e) {
    console.error('Remove failed:', e);
  }
}

/**
 * Refresh models from registry
 */
async function refreshModels() {
  const btn = event.target;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>&nbsp; Refreshing…';

  try {
    const response = await fetch(`${API_URL}/api/registry/refresh`, { method: 'POST' });
    const data = await response.json();

    if (data.status === 'completed') {
      await loadAllData();

      const notification = document.createElement('div');
      notification.className = 'alert alert-success';
      notification.innerHTML = `
        <strong>✓ Refresh completed</strong><br>
        Total: ${data.total_models || 0} models &nbsp;·&nbsp;
        Ollama: ${data.providers?.ollama?.models || 0} &nbsp;·&nbsp;
        NVIDIA: ${data.providers?.nvidia?.models || 0}
      `;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 5000);
    }

    btn.disabled = false;
    btn.innerHTML = '↻&nbsp; Refresh';
  } catch (e) {
    console.error('Refresh failed:', e);

    const notification = document.createElement('div');
    notification.className = 'alert alert-error';
    notification.innerHTML = `<strong>✗ Refresh failed</strong><br>${e.message || 'Check console for details'}`;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);

    btn.disabled = false;
    btn.innerHTML = '↻&nbsp; Refresh';
  }
}

/**
 * Run performance test
 */
async function runPerfTest() {
  const btn = document.getElementById('perf-btn');
  const status = document.getElementById('perf-status');
  const prompt = document.getElementById('perf-prompt').value;
  const tbody = document.getElementById('perf-body');

  btn.disabled = true;
  status.textContent = 'Running benchmark…';
  status.style.color = 'var(--text-3)';
  tbody.innerHTML = '<tr><td colspan="5" class="empty"><span class="spinner"></span>&nbsp; Running tests…</td></tr>';

  try {
    const res = await fetch(`${API_URL}/performance-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, mode: 'all' })
    });

    const data = await res.json();
    tbody.innerHTML = '';
    data.results.forEach(r => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td><strong>${r.label}</strong></td>
        <td>${r.model}</td>
        <td style="font-family:'JetBrains Mono',monospace;color:var(--cyan);">${r.duration}s</td>
        <td style="font-family:'JetBrains Mono',monospace;">${r.tokens}</td>
        <td style="font-size:12px;color:var(--text-3);max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.response.substring(0, 80)}…</td>
      `;
      tbody.appendChild(row);
    });
    status.textContent = '✓ Benchmark completed';
    status.style.color = 'var(--green)';
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty" style="color:var(--red);">✗ Error: ${e.message}</td></tr>`;
    status.textContent = '✗ Test failed';
    status.style.color = 'var(--red)';
  } finally {
    btn.disabled = false;
  }
}

/**
 * Populate model selects for fallback configuration
 */
function populateModelSelects() {
  fetch(`${API_URL}/models`).then(r => r.json()).then(data => {
    const models = Object.keys(data.models || {}).sort();
    const select1 = document.getElementById('fb-primary');
    const select2 = document.getElementById('fb-fallback');
    select1.innerHTML = '<option value="">Select primary model…</option>';
    select2.innerHTML = '<option value="">Select fallback model…</option>';
    models.forEach(m => {
      select1.innerHTML += `<option value="${m}">${m}</option>`;
      select2.innerHTML += `<option value="${m}">${m}</option>`;
    });
  });
}

// Initialize model selects
populateModelSelects();