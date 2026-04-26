"""
Configuration loader for LLM IntelliProxy.

Loads configuration from config.yaml with environment variable substitution.
Supports ${VAR} syntax for environment variable interpolation.
"""
import os
import re
from typing import Any, Dict, Optional
import yaml


class ConfigLoader:
    """Loads and manages configuration from YAML file with env var substitution."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> 'ConfigLoader':
        """Singleton pattern to ensure config is loaded once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config loader."""
        if not self._config:
            self.load()
    
    def _substitute_env_vars(self, value: Any) -> Any:
        """Recursively substitute ${VAR} patterns with environment variables."""
        if isinstance(value, str):
            # Match ${VAR} pattern
            pattern = r'\$\{([^}]+)\}'
            
            def replace(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(pattern, replace, value)
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        return value
    
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml. Defaults to /app/config.yaml or ./config.yaml
            
        Returns:
            Loaded configuration dictionary
        """
        if config_path is None:
            # Try multiple locations
            for path in ['/app/config.yaml', './config.yaml', 'config.yaml']:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path is None or not os.path.exists(config_path):
            # Return default configuration
            self._config = self._get_default_config()
            return self._config
        
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Substitute environment variables
            self._config = self._substitute_env_vars(raw_config)
            
            # Merge with defaults for any missing keys
            self._config = self._merge_defaults(self._config, self._get_default_config())
            
            return self._config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            self._config = self._get_default_config()
            return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            'proxy': {
                'port': int(os.getenv('PROXY_PORT', '8128')),
                'log_level': os.getenv('LOG_LEVEL', 'info'),
                'fallback_model': os.getenv('FALLBACK_MODEL', 'qwen2.5:7b'),
            },
            'decision': {
                'model': os.getenv('DECISION_MODEL', ''),
                'refresh_registry_on_startup': True,
            },
            'providers': [
                {
                    'name': 'ollama',
                    'type': 'ollama',
                    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434'),
                    'refresh_interval_minutes': 15,
                    'enabled': True,
                },
            ],
            'storage': {
                'type': 'sqlite',
                'path': os.getenv('DATA_DIR', '/data') + '/llmproxy.db',
            },
            'dashboard': {
                'port': int(os.getenv('DASHBOARD_PORT', '8028')),
                'host': os.getenv('WEB_HOST', '0.0.0.0'),
            },
        }
    
    def _merge_defaults(self, config: Dict, defaults: Dict) -> Dict:
        """Recursively merge config with defaults."""
        result = defaults.copy()
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_defaults(value, result[key])
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'proxy.port')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config
    
    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration or None if not found
        """
        providers = self._config.get('providers', [])
        for provider in providers:
            if provider.get('name') == provider_name:
                return provider
        return None
    
    def get_enabled_providers(self) -> list:
        """Get list of enabled provider configurations."""
        providers = self._config.get('providers', [])
        return [p for p in providers if p.get('enabled', True)]


# Global config instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    loader = get_config()
    return loader.load(config_path)