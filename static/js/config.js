// App Configuration Module
// Use current host if available, otherwise default to localhost
const currentHost = window.location.hostname;
const APP_CONFIG = {
    apiUrl: `http://${currentHost}:8128`  // Use dynamic hostname
};

// Feature matrix loaded from JSON config
let FEATURE_MATRIX = {};
let FEATURE_LABELS = {};
let CATEGORY_COLORS = {};
let DEFAULT_MODEL = {};

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
    airllm_port: 8128,
    model_airllm_settings: {}
};
