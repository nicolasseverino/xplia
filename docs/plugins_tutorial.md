# Système de plugins XPLIA

XPLIA permet l’extension dynamique de ses fonctionnalités via un système de plugins (explainers, visualizers, compliance, etc.).

## 1. Créer un plugin

```python
# xplia/plugins/my_visualizer.py
from xplia.plugins import PluginRegistry

def register_plugin(registry):
    registry.register('my_visualizer', MyVisualizer)

class MyVisualizer:
    def render(self, explanation_result):
        # Code de visualisation custom
        pass
```

## 2. Découverte automatique des plugins

```python
from xplia.plugins import PluginRegistry
PluginRegistry.auto_discover()
print(PluginRegistry.all())
```

## 3. Utilisation dans XPLIA

```python
viz = PluginRegistry.get('my_visualizer')()
result = ... # résultat d’explication
viz.render(result)
```

## 4. Bonnes pratiques
- Préfixer les plugins par le domaine ou la société pour éviter les collisions.
- Documenter chaque plugin et fournir des exemples d’utilisation.
- Utiliser l’auto-discovery pour charger dynamiquement tous les plugins d’un dossier.
