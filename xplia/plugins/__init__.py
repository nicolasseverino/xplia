"""
Système de plugins XPLIA
========================

Permet l’extension dynamique de la librairie par des modules externes (explainers, visualizers, compliance, etc.).
"""

import importlib
import pkgutil

class PluginRegistry:
    """
    Registre global des plugins XPLIA.
    """
    _registry = {}

    @classmethod
    def register(cls, name, plugin):
        cls._registry[name] = plugin
    @classmethod
    def get(cls, name):
        return cls._registry.get(name)
    @classmethod
    def all(cls):
        return dict(cls._registry)

    @classmethod
    def auto_discover(cls, package='xplia.plugins'):
        """
        Découvre et enregistre automatiquement tous les plugins du package.
        """
        for loader, modname, ispkg in pkgutil.iter_modules(__import__(package).__path__):
            if modname != '__init__':
                module = importlib.import_module(f"{package}.{modname}")
                if hasattr(module, 'register_plugin'):
                    module.register_plugin(cls)
