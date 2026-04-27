/**
 * Model Rendering Module
 * Handles dashboard rendering for models and features
 */

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
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h3>📊 Overall Statistics</h3>
                <button onclick="refreshAllData()" class="btn btn-secondary">🔄 Refresh</button>
            </div>
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