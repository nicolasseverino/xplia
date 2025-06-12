"""
Exemple de plugin Visualizer pour XPLIA
"""
from . import PluginRegistry

def register_plugin(registry):
    registry.register('example_visualizer', ExampleVisualizer)

class ExampleVisualizer:
    def render(self, explanation_result):
        return f"Visualisation custom : {explanation_result}"
