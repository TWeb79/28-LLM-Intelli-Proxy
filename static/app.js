const APP_CONFIG = {
    apiUrl: 'http://localhost:9998'  // Changed from 8000 to 9998
};

// Models feature matrix
const FEATURE_MATRIX = {
    'qwen2.5:7b': {
        name: 'Qwen 2.5',
        size: '4.7 GB',
        category: 'general',
        features: { general: 4, code: 3, vision: 0, reasoning: 3, speed: 4, quality: 4, censored: 1 },
        description: 'Balanced general-purpose LLM'
    },
    'qwen2.5-coder:7b': {
        name: 'Qwen 2.5 Coder',
        size: '4.7 GB',
        category: 'code',
        features: { general: 3, code: 4, vision: 0, reasoning: 2, speed: 4, quality: 3, censored: 1 },
        description: 'Specialized for code generation and debugging'
    },
    'qwen3-coder:latest': {
        name: 'Qwen 3 Coder',
        size: '18 GB',
        category: 'code',
        features: { general: 4, code: 5, vision: 0, reasoning: 3, speed: 2, quality: 4, censored: 1 },
        description: 'Advanced code generation and analysis'
    },
    'deepseek-r1:latest': {
        name: 'DeepSeek R1',
        size: '5.2 GB',
        category: 'reasoning',
        features: { general: 4, code: 4, vision: 0, reasoning: 5, speed: 2, quality: 4, censored: 1 },
        description: 'Advanced reasoning and step-by-step analysis'
    },
    'llama2-uncensored:latest': {
        name: 'Llama 2 Uncensored',
        size: '3.8 GB',
        category: 'uncensored',
        features: { general: 3, code: 3, vision: 0, reasoning: 2, speed: 4, quality: 3, censored: 0 },
        description: 'Unrestricted content generation'
    },
    'nemotron-3-nano:latest': {
        name: 'Nemotron 3 Nano',
        size: '24 GB',
        category: 'general',
        features: { general: 5, code: 4, vision: 0, reasoning: 4, speed: 2, quality: 5, censored: 1 },
        description: 'Highest quality general-purpose LLM'
    },
    'llava:7b': {
        name: 'LLaVA 7B',
        size: '4.7 GB',
        category: 'vision',
        features: { general: 3, code: 3, vision: 3, reasoning: 2, speed: 4, quality: 3, censored: 1 },
        description: 'Vision and language understanding'
    },
    'goonsai/qwen2.5-3B-goonsai-nsfw-100k:latest': {
        name: 'Qwen 2.5 NSFW',
        size: '6.2 GB',
        category: 'uncensored',
        features: { general: 3, code: 2, vision: 0, reasoning: 2, speed: 4, quality: 2, censored: 0 },
        description: 'NSFW/Adult content generation'
    },
    'minimax-m2.1:cloud': {
        name: 'Minimax M2.1',
        size: 'Cloud',
        category: 'general',
        features: { general: 5, code: 4, vision: 0, reasoning: 5, speed: 2, quality: 5, censored: 1 },
        description: 'Premium cloud-based model'
    }
};

const FEATURE_LABELS = {
    general: '🧠 General',
    code: '💻 Code',
    vision: '👁️ Vision',
    reasoning: '🔍 Reasoning',
    speed: '⚡ Speed',
    quality: '✨ Quality',
    censored: '🔒 Censored'
};

// Global statistics
let stats = {
    total_requests: 0,
    models: {},
    categories: {},
    last_update: new Date()
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
    const colors = {
        'code': '#10b981',
        'vision': '#8b5cf6',
        'reasoning': '#f59e0b',
        'general': '#3b82f6',
        'uncensored': '#ef4444'
    };
    return colors[category] || '#6b7280';
}

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(`tab-${tabName}`).style.display = 'block';
    event.target.classList.add('active');
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
document.addEventListener('DOMContentLoaded', () => {
    renderDashboard();
    updateStats();
    
    // Update stats every 5 seconds
    setInterval(updateStats, 5000);
});
