// App Configuration
const APP_CONFIG = {
    apiUrl: 'http://localhost:9998'  // Changed from 8000 to 9998
};

// Feature matrix loaded from JSON config
let FEATURE_MATRIX = {};
let FEATURE_LABELS = {};
let CATEGORY_COLORS = {};
let DEFAULT_MODEL = {};

// Load model configuration from JSON file
async function loadModelConfig() {
    try {
        // Load from static path (served by backend at /static/)
        const response = await fetch('./static/models.json');
        if (response.ok) {
            const config = await response.json();
            FEATURE_MATRIX = config.models || {};
            FEATURE_LABELS = config.feature_labels || {};
            CATEGORY_COLORS = config.category_colors || {};
            DEFAULT_MODEL = config.default_model || {};
            console.log(`📋 Loaded ${Object.keys(FEATURE_MATRIX).length} models from models.json`);
            return true;
        }
    } catch (error) {
        console.warn('Could not load models.json, using fallback:', error.message);
    }
    return false;
}

// Fetch models from backend and merge with JSON config
async function syncModelsFromBackend() {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/models`);
        if (response.ok) {
            const data = await response.json();
            const backendModels = data.models || {};
            
            // Merge backend models with local config
            Object.keys(backendModels).forEach(modelName => {
                if (!FEATURE_MATRIX[modelName]) {
                    // New model discovered by backend - use default with backend attrs
                    FEATURE_MATRIX[modelName] = {
                        ...DEFAULT_MODEL,
                        name: modelName,
                        backend_attrs: backendModels[modelName]
                    };
                    console.log(`🔍 New model discovered: ${modelName}`);
                } else {
                    // Update existing model with backend attributes
                    FEATURE_MATRIX[modelName].backend_attrs = backendModels[modelName];
                }
            });
            
            console.log(`🔄 Synced ${Object.keys(backendModels).length} models from backend`);
            return true;
        }
    } catch (error) {
        console.warn('Could not sync with backend:', error.message);
    }
    return false;
}

// Global statistics
let stats = {
    total_requests: 0,
    models: {},
    categories: {},
    last_update: new Date()
};

// Fallback configuration
let fallbackConfig = {
    fallbacks: {},
    timeout: 300
};

// AirLLM Configuration
let airllmConfig = {
    ollama_host: "ollama",
    ollama_port: 11434,
    airllm_enabled: false,
    airllm_host: "airllm",
    airllm_port: 9996,
    model_airllm_settings: {}
};

// Fetch statistics from server
async function updateStats() {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/stats`);
        if (response.ok) {
            stats = await response.json();
            renderDashboard();
        }
    } catch (error) {
        console.log('Stats not available yet');
    }
}

// Fetch fallback configuration from server
async function updateFallbackConfig() {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/config/fallbacks`);
        if (response.ok) {
            fallbackConfig = await response.json();
            renderFallbackConfig();
        }
    } catch (error) {
        console.log('Fallback config not available');
    }
}

// Render fallback configuration
function renderFallbackConfig() {
    const container = document.getElementById('fallback-config');
    if (!container) return;
    
    let html = `
        <div class="stats-header">
            <h3>🔄 Fallback Configuration</h3>
            <p>Timeout: <strong>${fallbackConfig.timeout}s</strong></p>
        </div>
        <div class="fallback-grid">
    `;
    
    const fallbacks = fallbackConfig.fallbacks || {};
    if (Object.keys(fallbacks).length === 0) {
        html += '<p style="text-align: center; color: #999;">No fallback models configured</p>';
    } else {
        Object.entries(fallbacks).forEach(([model, fallback]) => {
            const modelName = FEATURE_MATRIX[model]?.name || model;
            const fallbackName = FEATURE_MATRIX[fallback]?.name || fallback;
            html += `
                <div class="fallback-card">
                    <div class="fallback-model">${modelName}</div>
                    <div class="fallback-arrow">→</div>
                    <div class="fallback-target">${fallbackName}</div>
                </div>
            `;
        });
    }
    
    html += '</div>';
    container.innerHTML = html;
}

// Fetch AirLLM configuration from server
async function updateAirLLMConfig() {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/config/airllm`);
        if (response.ok) {
            airllmConfig = await response.json();
            renderAirLLMConfig();
        }
    } catch (error) {
        console.log('AirLLM config not available');
    }
}

// Refresh health status in header (called after saving config)
async function refreshHealthStatus() {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/api/health`);
        if (response.ok) {
            const data = await response.json();
            
            // Update header Ollama URL display
            const ollamaUrlTop = document.getElementById('ollama-url-top');
            if (ollamaUrlTop) {
                ollamaUrlTop.textContent = data.ollama_url || 'Not configured';
            }
            
            // Update main Ollama URL display
            const ollamaUrl = document.getElementById('ollama-url');
            if (ollamaUrl) {
                ollamaUrl.textContent = data.ollama_url || 'Not configured';
            }
            
            // Update status lights
            const ollamaStatusTop = document.getElementById('ollama-status-top');
            if (ollamaStatusTop) {
                ollamaStatusTop.style.background = data.status === 'healthy' ? '#22c55e' : '#ef4444';
            }
            
            const ollamaStatus = document.getElementById('ollama-status');
            if (ollamaStatus) {
                ollamaStatus.style.background = data.status === 'healthy' ? '#22c55e' : '#ef4444';
            }
            
            const routerStatusTop = document.getElementById('router-status-top');
            if (routerStatusTop) {
                routerStatusTop.style.background = data.api_status === 'running' ? '#22c55e' : '#ef4444';
            }
            
            const routerStatus = document.getElementById('router-status');
            if (routerStatus) {
                routerStatus.style.background = data.api_status === 'running' ? '#22c55e' : '#ef4444';
            }
            
            console.log('✅ Health status refreshed:', data.ollama_url);
        }
    } catch (error) {
        console.warn('Could not refresh health status:', error.message);
    }
}

// Render AirLLM configuration
function renderAirLLMConfig() {
    // Update form fields
    const hostInput = document.getElementById('ollama-target-host');
    const portInput = document.getElementById('ollama-target-port');
    const airllmEnabled = document.getElementById('airllm-enabled');
    const airllmHost = document.getElementById('airllm-host');
    const airllmPort = document.getElementById('airllm-port');
    
    if (hostInput) hostInput.value = airllmConfig.ollama_host || 'ollama';
    if (portInput) portInput.value = airllmConfig.ollama_port || 11434;
    if (airllmEnabled) airllmEnabled.value = airllmConfig.airllm_enabled ? 'true' : 'false';
    if (airllmHost) airllmHost.value = airllmConfig.airllm_host || 'airllm';
    if (airllmPort) airllmPort.value = airllmConfig.airllm_port || 9996;
    
    // Render model AirLLM toggles
    renderModelAirLLMToggles();
}

// Render model AirLLM toggles
function renderModelAirLLMToggles() {
    const container = document.getElementById('model-airllm-list');
    if (!container) return;
    
    let html = '';
    const modelSettings = airllmConfig.model_airllm_settings || {};
    
    // Get available models from stats or use default list
    const models = Object.keys(stats.models || {});
    
    if (models.length === 0) {
        // Use default model list from FEATURE_MATRIX
        Object.keys(FEATURE_MATRIX).forEach(modelName => {
            const isEnabled = modelSettings[modelName] || false;
            const modelInfo = FEATURE_MATRIX[modelName];
            html += `
                <div class="card" style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-weight:500;">${modelInfo?.name || modelName}</div>
                        <div style="font-size:11px;color:#999;">${modelName}</div>
                    </div>
                    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;">
                        <input type="checkbox" 
                            ${isEnabled ? 'checked' : ''} 
                            onchange="toggleModelAirLLM('${modelName}', this.checked)"
                            style="width:18px;height:18px;" />
                        <span style="font-size:12px;color:#666;">AirLLM</span>
                    </label>
                </div>
            `;
        });
    } else {
        models.forEach(modelName => {
            const isEnabled = modelSettings[modelName] || false;
            html += `
                <div class="card" style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-weight:500;">${modelName}</div>
                    </div>
                    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;">
                        <input type="checkbox" 
                            ${isEnabled ? 'checked' : ''} 
                            onchange="toggleModelAirLLM('${modelName}', this.checked)"
                            style="width:18px;height:18px;" />
                        <span style="font-size:12px;color:#666;">AirLLM</span>
                    </label>
                </div>
            `;
        });
    }
    
    if (html === '') {
        html = '<p style="color:#999;font-size:13px;">No models available. Models will appear here after requests are made.</p>';
    }
    
    container.innerHTML = html;
}

// Save Ollama target configuration
async function saveOllamaTarget() {
    const host = document.getElementById('ollama-target-host').value;
    const port = parseInt(document.getElementById('ollama-target-port').value);
    
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/config/ollama`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host, port })
        });
        
        if (response.ok) {
            const result = await response.json();
            document.getElementById('ollama-target-status').textContent = `✅ Saved: ${result.base_url}`;
            airllmConfig.ollama_host = host;
            airllmConfig.ollama_port = port;
            // Refresh the header display and status
            await refreshHealthStatus();
        } else {
            document.getElementById('ollama-target-status').textContent = '❌ Error saving configuration';
        }
    } catch (error) {
        document.getElementById('ollama-target-status').textContent = '❌ Error: ' + error.message;
    }
}

// Toggle AirLLM service
async function toggleAirLLMService() {
    const enabled = document.getElementById('airllm-enabled').value === 'true';
    const host = document.getElementById('airllm-host').value;
    const port = parseInt(document.getElementById('airllm-port').value);
    
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/config/airllm/service`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled, host, port })
        });
        
        if (response.ok) {
            const result = await response.json();
            document.getElementById('airllm-status').textContent = `✅ AirLLM ${enabled ? 'enabled' : 'disabled'}: ${result.base_url}`;
            airllmConfig.airllm_enabled = enabled;
            airllmConfig.airllm_host = host;
            airllmConfig.airllm_port = port;
            renderModelAirLLMToggles();
            // Refresh the header display and status
            await refreshHealthStatus();
        } else {
            document.getElementById('airllm-status').textContent = '❌ Error saving configuration';
        }
    } catch (error) {
        document.getElementById('airllm-status').textContent = '❌ Error: ' + error.message;
    }
}

// Save AirLLM configuration
async function saveAirLLMConfig() {
    await toggleAirLLMService();
}

// Toggle AirLLM for a specific model
async function toggleModelAirLLM(modelName, enabled) {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/config/model/airllm`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName, enabled })
        });
        
        if (response.ok) {
            const result = await response.json();
            airllmConfig.model_airllm_settings = result.model_airllm_settings;
        }
    } catch (error) {
        console.error('Error toggling model AirLLM:', error);
    }
}

// Render dashboard
function renderDashboard() {
    renderModelsGrid();
    renderUsageStats();
    renderCategoryBreakdown();
}

// Render models feature matrix
function renderModelsGrid() {
    const container = document.getElementById('models-grid');
    if (!container) return;
    
    container.innerHTML = '';
    
    Object.entries(FEATURE_MATRIX).forEach(([modelId, model]) => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.style.borderLeft = `4px solid ${getCategoryColor(model.category)}`;
        
        const usage = stats.models?.[modelId]?.count || 0;
        const percentage = stats.total_requests > 0 ? ((usage / stats.total_requests) * 100).toFixed(1) : 0;
        
        let featuresHtml = '';
        Object.entries(model.features).forEach(([feature, rating]) => {
            const stars = '⭐'.repeat(rating) + '☆'.repeat(5 - rating);
            featuresHtml += `<div class="feature-row">
                <span class="feature-label">${FEATURE_LABELS[feature]}</span>
                <span class="feature-rating">${stars}</span>
            </div>`;
        });
        
        card.innerHTML = `
            <div class="model-header">
                <h3>${model.name}</h3>
                <span class="category-badge" style="background: ${getCategoryColor(model.category)}">${model.category}</span>
            </div>
            <p class="model-description">${model.description}</p>
            <div class="model-meta">
                <span>📦 ${model.size}</span>
                <span>📊 ${usage} requests (${percentage}%)</span>
            </div>
            <div class="features-grid">
                ${featuresHtml}
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Render usage statistics
function renderUsageStats() {
    const container = document.getElementById('usage-stats');
    if (!container) return;
    
    const modelStats = Object.entries(stats.models || {})
        .map(([modelId, data]) => ({
            model: FEATURE_MATRIX[modelId]?.name || modelId,
            count: data.count,
            percentage: stats.total_requests > 0 ? ((data.count / stats.total_requests) * 100).toFixed(1) : 0
        }))
        .sort((a, b) => b.count - a.count);
    
    let html = `
        <div class="stats-header">
            <h3>📊 Overall Statistics</h3>
            <p>Total Requests: <strong>${stats.total_requests}</strong></p>
            <p>Last Updated: <strong>${new Date(stats.last_update).toLocaleTimeString()}</strong></p>
        </div>
        <div class="stats-table">
    `;
    
    if (modelStats.length === 0) {
        html += '<p style="text-align: center; color: #999;">No requests yet</p>';
    } else {
        modelStats.forEach(stat => {
            const barWidth = parseFloat(stat.percentage);
            html += `
                <div class="stat-row">
                    <div class="stat-label">${stat.model}</div>
                    <div class="stat-bar-container">
                        <div class="stat-bar" style="width: ${barWidth}%"></div>
                    </div>
                    <div class="stat-value">${stat.count} (${stat.percentage}%)</div>
                </div>
            `;
        });
    }
    
    html += '</div>';
    container.innerHTML = html;
}

// Render category breakdown
function renderCategoryBreakdown() {
    const container = document.getElementById('category-breakdown');
    if (!container) return;
    
    const categories = Object.entries(stats.categories || {})
        .map(([cat, count]) => ({
            category: cat,
            count: count,
            percentage: stats.total_requests > 0 ? ((count / stats.total_requests) * 100).toFixed(1) : 0
        }))
        .sort((a, b) => b.count - a.count);
    
    let html = '<div class="category-grid">';
    
    if (categories.length === 0) {
        html += '<p style="grid-column: 1/-1; text-align: center; color: #999;">No data yet</p>';
    } else {
        categories.forEach(cat => {
            html += `
                <div class="category-card" style="border-left-color: ${getCategoryColor(cat.category)}">
                    <div class="category-name">${cat.category.toUpperCase()}</div>
                    <div class="category-count">${cat.count}</div>
                    <div class="category-percentage">${cat.percentage}%</div>
                </div>
            `;
        });
    }
    
    html += '</div>';
    container.innerHTML = html;
}

// Get category color
function getCategoryColor(category) {
    // Use colors from JSON config first, fallback to defaults
    if (CATEGORY_COLORS[category]) {
        return CATEGORY_COLORS[category];
    }
    const colors = {
        'code': '#10b981',
        'vision': '#8b5cf6',
        'reasoning': '#f59e0b',
        'general': '#3b82f6',
        'uncensored': '#ef4444'
    };
    return colors[category] || '#6b7280';
}

// Fallback configuration styles (to be added to CSS)
const FALLBACK_STYLES = `
    .fallback-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    .fallback-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }
    
    .fallback-model {
        font-weight: 600;
        color: #1f2937;
        font-size: 0.9em;
        flex: 1;
    }
    
    .fallback-arrow {
        color: #f59e0b;
        font-size: 1.2em;
    }
    
    .fallback-target {
        font-weight: 600;
        color: #059669;
        font-size: 0.9em;
        flex: 1;
        text-align: right;
    }
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = FALLBACK_STYLES;
document.head.appendChild(styleSheet);

// Tab switching - simplified version
function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    // Hide all tab contents
    const allTabs = document.querySelectorAll('.tab-content');
    allTabs.forEach(tab => {
        tab.style.display = 'none';
    });
    
    // Remove active class from all buttons
    const allButtons = document.querySelectorAll('.tab-btn');
    allButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById('tab-' + tabName);
    if (selectedTab) {
        selectedTab.style.display = 'block';
    }
    
    // Add active class to clicked button (find by onclick attribute)
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        if (btn.getAttribute('onclick') && btn.getAttribute('onclick').indexOf(tabName) !== -1) {
            btn.classList.add('active');
        }
    });
}

// Code examples
function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    navigator.clipboard.writeText(element.textContent).then(() => {
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Load model configuration from JSON first
    const jsonLoaded = await loadModelConfig();
    
    if (!jsonLoaded) {
        // Fallback to embedded config if JSON fails
        console.warn('Using embedded model config as fallback');
    }
    
    // Then sync with backend to get auto-discovered models
    await syncModelsFromBackend();
    
    // Initial render
    renderDashboard();
    updateStats();
    updateFallbackConfig();
    updateAirLLMConfig();
    
    // Refresh health status in header (for IPs and traffic lights)
    await refreshHealthStatus();
    
    // Update stats every 5 seconds
    setInterval(updateStats, 5000);
    setInterval(updateFallbackConfig, 10000);
    setInterval(updateAirLLMConfig, 15000);
    
    // Refresh health status every 10 seconds (for traffic lights)
    setInterval(refreshHealthStatus, 10000);
    
    // Sync with backend every 30 seconds to detect new models
    setInterval(async () => {
        await syncModelsFromBackend();
        renderDashboard();  // Re-render if new models found
    }, 30000);
});
