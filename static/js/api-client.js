/**
 * API Client Module
 * Handles all HTTP requests to the IntelliProxy backend
 */

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

// Manual refresh function - called by refresh button
async function refreshAllData() {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '⏳ Loading...';
    
    await updateStats();
    await updateFallbackConfig();
    await updateAirLLMConfig();
    
    btn.disabled = false;
    btn.innerHTML = '🔄 Refresh';
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

// Refresh health status in header (for traffic lights)
async function refreshHealthStatus() {
    try {
        const response = await fetch(`${APP_CONFIG.apiUrl}/health`);
        if (response.ok) {
            const data = await response.json();
            
            // Update header Ollama URL display
            const ollamaUrlTop = document.getElementById('ollama-url-top');
            if (ollamaUrlTop) {
                ollamaUrlTop.textContent = data.ollama_url || 'Not configured';
            }
            
            // Update header Router URL display  
            const routerUrlTop = document.getElementById('router-url-top');
            if (routerUrlTop) {
                routerUrlTop.textContent = data.router_url || 'Not configured';
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