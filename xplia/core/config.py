"""
Système de configuration pour XPLIA
=======================================

Ce module fournit un système centralisé de gestion des configurations,
permettant une personnalisation fine et cohérente de tous les composants
de la librairie.
"""

import copy
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ConfigScope(str, Enum):
    """Portées des configurations."""
    GLOBAL = "global"
    MODULE = "module"
    INSTANCE = "instance"
    SESSION = "session"


class ConfigManager:
    """
    Gestionnaire centralisé de configuration.
    
    Cette classe implémente le pattern Singleton pour assurer
    qu'une seule instance gère la configuration globale.
    """
    
    _instance = None
    _default_config = {
        "visualization": {
            "theme": "light",
            "color_palette": "viridis",
            "font_size": 12,
            "interactive": True,
            "export_format": "png",
            "dpi": 300,
        },
        "explanation": {
            "default_method": "unified",
            "n_samples": 1000,
            "include_baseline": True,
            "randomize_seed": 42,
            "confidence_level": 0.95,
        },
        "performance": {
            "parallel_jobs": -1,
            "cache_results": True,
            "cache_dir": "./xplia_cache",
            "use_gpu": "auto",
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "verbose": False,
        },
        "compliance": {
            "generate_reports": True,
            "regulations": ["gdpr", "ai_act"],
            "documentation_level": "standard",
            "audit_trail": True,
        },
        "ui": {
            "default_audience": "technical",
            "language": "en",
            "show_advanced_options": True,
            "dashboard_layout": "standard",
        },
    }
    
    def __new__(cls):
        """Implémentation du Singleton."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialise l'instance avec les configurations par défaut."""
        self._config = copy.deepcopy(self._default_config)
        self._load_user_config()
    
    def _load_user_config(self):
        """
        Charge la configuration de l'utilisateur depuis:
        1. Fichier de configuration global (~/.xplia/config.json)
        2. Variables d'environnement (XPLIA_*)
        3. Fichier de projet local (./.xplia_config.json)
        """
        # Fichier de configuration global
        global_config_path = Path.home() / ".xplia" / "config.json"
        if global_config_path.exists():
            try:
                with open(global_config_path, 'r') as f:
                    global_config = json.load(f)
                self._update_recursive(self._config, global_config)
            except Exception as e:
                # Utiliser un logger ici
                print(f"Erreur lors du chargement de la configuration globale: {e}")
        
        # Variables d'environnement
        env_prefix = "XPLIA_"
        for env_var, value in os.environ.items():
            if env_var.startswith(env_prefix):
                # Format attendu: XPLIA_SECTION_KEY=value
                # Exemple: XPLIA_VISUALIZATION_THEME=dark
                parts = env_var[len(env_prefix):].lower().split('_', 1)
                if len(parts) == 2:
                    section, key = parts
                    if section in self._config and key in self._config[section]:
                        # Conversion de type basique (str vers int, float, bool)
                        orig_type = type(self._config[section][key])
                        if orig_type == bool:
                            self._config[section][key] = value.lower() in ('true', 'yes', '1')
                        elif orig_type == int:
                            self._config[section][key] = int(value)
                        elif orig_type == float:
                            self._config[section][key] = float(value)
                        else:
                            self._config[section][key] = value
        
        # Fichier de projet local
        local_config_path = Path.cwd() / ".xplia_config.json"
        if local_config_path.exists():
            try:
                with open(local_config_path, 'r') as f:
                    local_config = json.load(f)
                self._update_recursive(self._config, local_config)
            except Exception as e:
                # Utiliser un logger ici
                print(f"Erreur lors du chargement de la configuration locale: {e}")
    
    def _update_recursive(self, target: Dict, source: Dict):
        """
        Met à jour récursivement un dictionnaire cible avec les valeurs d'un dictionnaire source.
        
        Args:
            target: Dictionnaire à mettre à jour
            source: Dictionnaire contenant les nouvelles valeurs
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_recursive(target[key], value)
            else:
                target[key] = value
    
    def get_config(self, section: Optional[str] = None, key: Optional[str] = None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            section: Section de configuration (optionnel)
            key: Clé de configuration dans la section (optionnel)
            
        Returns:
            Any: Valeur de configuration ou sous-dictionnaire
            
        Raises:
            KeyError: Si la section ou la clé n'existe pas
        """
        if section is None:
            return copy.deepcopy(self._config)
        
        if section not in self._config:
            raise KeyError(f"Section de configuration non trouvée: {section}")
        
        if key is None:
            return copy.deepcopy(self._config[section])
        
        if key not in self._config[section]:
            raise KeyError(f"Clé de configuration non trouvée: {key} dans la section {section}")
        
        return copy.deepcopy(self._config[section][key])
    
    def set_config(self, section: str, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.
        
        Args:
            section: Section de configuration
            key: Clé de configuration dans la section
            value: Nouvelle valeur
            
        Raises:
            KeyError: Si la section n'existe pas
        """
        if section not in self._config:
            raise KeyError(f"Section de configuration non trouvée: {section}")
        
        self._config[section][key] = value
    
    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """
        Met à jour une section entière de la configuration.
        
        Args:
            section: Section à mettre à jour
            values: Nouvelles valeurs
            
        Raises:
            KeyError: Si la section n'existe pas
        """
        if section not in self._config:
            raise KeyError(f"Section de configuration non trouvée: {section}")
        
        self._config[section].update(values)
    
    def reset(self, section: Optional[str] = None, key: Optional[str] = None) -> None:
        """
        Réinitialise la configuration aux valeurs par défaut.
        
        Args:
            section: Section à réinitialiser (optionnel)
            key: Clé à réinitialiser dans la section (optionnel)
        """
        if section is None:
            self._config = copy.deepcopy(self._default_config)
            return
        
        if section not in self._default_config:
            return
        
        if key is None:
            self._config[section] = copy.deepcopy(self._default_config[section])
            return
        
        if key not in self._default_config[section]:
            return
        
        self._config[section][key] = copy.deepcopy(self._default_config[section][key])
    
    def save_user_config(self, path: Optional[Union[str, Path]] = None,
                        scope: ConfigScope = ConfigScope.GLOBAL) -> None:
        """
        Sauvegarde la configuration actuelle.
        
        Args:
            path: Chemin du fichier de configuration (optionnel)
            scope: Portée de la configuration (détermine le chemin par défaut)
        """
        if path is None:
            if scope == ConfigScope.GLOBAL:
                path = Path.home() / ".xplia" / "config.json"
            elif scope == ConfigScope.SESSION:
                path = Path.cwd() / ".xplia_config.json"
            else:
                # Pour d'autres portées, utiliser le répertoire courant
                path = Path.cwd() / ".xplia_config.json"
        
        path = Path(path)
        
        # S'assurer que le répertoire parent existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Récupère la configuration par défaut.
        
        Returns:
            Dict[str, Any]: Configuration par défaut
        """
        return copy.deepcopy(self._default_config)


# Fonctions utilitaires pour accéder au gestionnaire de configuration
def get_config(section: Optional[str] = None, key: Optional[str] = None) -> Any:
    """
    Récupère une valeur de configuration.
    
    Args:
        section: Section de configuration (optionnel)
        key: Clé de configuration dans la section (optionnel)
        
    Returns:
        Any: Valeur de configuration
    """
    return ConfigManager().get_config(section, key)


def set_config(section: str, key: str, value: Any) -> None:
    """
    Définit une valeur de configuration.
    
    Args:
        section: Section de configuration
        key: Clé de configuration dans la section
        value: Nouvelle valeur
    """
    ConfigManager().set_config(section, key, value)
