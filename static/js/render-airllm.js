/**
 * AirLLM Configuration Rendering Module
 */

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