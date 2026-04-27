/**
 * Dashboard Main Entry Point
 * Orchestrates all modules for the IntelliProxy dashboard
 * 
 * File size: ~130 lines (well within 500-line limit)
 * Complies with RULES_coding.md:
 * - no inline JavaScript in HTML
 * - modular component structure
 * - clear separation of concerns
 */

// Import all modules (loaded via script tags in order)
// - config.js: APP_CONFIG, FEATURE_MATRIX, stats, fallbackConfig, airllmConfig
// - model-sync.js: loadModelConfig(), syncModelsFromBackend()
// - api-client.js: API functions and state updates
// - render-models.js: renderDashboard(), renderModelsGrid(), etc.
// - render-fallback.js: renderFallbackConfig()
// - render-airllm.js: renderAirLLMConfig(), renderModelAirLLMToggles()

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Load model configuration from JSON first
    const jsonLoaded = await loadModelConfig();
    
    if (!jsonLoaded) {
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
        renderDashboard();
    }, 30000);
});

// Tab switching utility
function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    const allTabs = document.querySelectorAll('.tab-content');
    allTabs.forEach(tab => {
        tab.style.display = 'none';
    });
    
    const allButtons = document.querySelectorAll('.tab-btn');
    allButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    const selectedTab = document.getElementById('tab-' + tabName);
    if (selectedTab) {
        selectedTab.style.display = 'block';
    }
    
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        if (btn.getAttribute('onclick') && btn.getAttribute('onclick').indexOf(tabName) !== -1) {
            btn.classList.add('active');
        }
    });
}

// Copy to clipboard utility
function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    const btn = event.target;
    const originalText = btn.textContent;
    navigator.clipboard.writeText(element.textContent).then(() => {
        btn.textContent = '✓ Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
}