/**
 * Fallback Configuration Rendering Module
 */

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