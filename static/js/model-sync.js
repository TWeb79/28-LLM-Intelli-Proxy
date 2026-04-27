// Model Synchronization Module
// Fetch model configuration from JSON file
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