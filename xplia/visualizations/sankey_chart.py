"""
Implémentation des diagrammes de flux Sankey pour XPLIA
"""

def create_sankey_chart(data, config, library="plotly", **kwargs):
    """
    Crée un diagramme de flux Sankey avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le diagramme (sources, cibles, valeurs)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_sankey_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_sankey_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_sankey_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les diagrammes Sankey")

# Implémentations spécifiques
def _create_plotly_sankey_chart(data, config, **kwargs):
    """Implémentation Plotly des diagrammes Sankey"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 1000)
    height = config.get('height', 600)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {source: [...], target: [...], value: [...], node: {label: [...]}}
        source = data.get('source', [])
        target = data.get('target', [])
        value = data.get('value', [])
        node_labels = data.get('node', {}).get('label', [])
        link_colors = data.get('link', {}).get('color', [])
        node_colors = data.get('node', {}).get('color', [])
    else:
        # Format [sources, targets, values, node_labels]
        source = data[0] if len(data) > 0 else []
        target = data[1] if len(data) > 1 else []
        value = data[2] if len(data) > 2 else [1] * len(source)
        node_labels = data[3] if len(data) > 3 else []
        link_colors = []
        node_colors = []
    
    # Configuration du diagramme Sankey
    sankey_config = dict(
        node=dict(
            pad=kwargs.get('node_pad', 15),
            thickness=kwargs.get('node_thickness', 20),
            line=dict(
                color=kwargs.get('node_line_color', "black"),
                width=kwargs.get('node_line_width', 0.5)
            ),
            label=node_labels if node_labels else None
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate=kwargs.get('hovertemplate', 'Source: %{source.label}<br>Target: %{target.label}<br>Value: %{value}<extra></extra>')
        )
    )
    
    # Ajout des couleurs personnalisées pour les nœuds si disponibles
    if node_colors:
        sankey_config['node']['color'] = node_colors
        
    # Ajout des couleurs personnalisées pour les liens si disponibles
    if link_colors:
        sankey_config['link']['color'] = link_colors
    elif 'link_colorscale' in kwargs:
        # Utiliser une échelle de couleurs pour les liens en fonction de leur valeur
        import numpy as np
        import plotly.colors as pc
        
        # Normaliser les valeurs entre 0 et 1
        norm_values = [(v - min(value)) / (max(value) - min(value) + 0.0001) for v in value]
        
        # Générer les couleurs à partir de l'échelle
        colorscale = kwargs.get('link_colorscale', 'Viridis')
        if isinstance(colorscale, str):
            colors = [pc.sample_colorscale(pc.get_colorscale(colorscale), v)[0] for v in norm_values]
        else:
            colors = [pc.sample_colorscale(colorscale, v)[0] for v in norm_values]
            
        sankey_config['link']['color'] = colors
    
    # Création du graphique
    fig = go.Figure(data=[go.Sankey(**sankey_config)])
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        font=dict(
            size=kwargs.get('font_size', 12)
        ),
        margin=dict(t=50, b=25, l=25, r=25)
    )
            
    return fig

def _create_mpl_sankey_chart(data, config, **kwargs):
    """Implémentation Matplotlib des diagrammes Sankey"""
    import matplotlib.pyplot as plt
    from matplotlib.sankey import Sankey
    
    # Extraction des paramètres
    title = config.get('title', '')
    
    # Note: La fonction Sankey de Matplotlib est limitée comparée à Plotly
    # Elle nécessite un format spécifique qui diffère du format Plotly
    # On va donc convertir les données
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        source = data.get('source', [])
        target = data.get('target', [])
        value = data.get('value', [])
        node_labels = data.get('node', {}).get('label', [])
    else:
        source = data[0] if len(data) > 0 else []
        target = data[1] if len(data) > 1 else []
        value = data[2] if len(data) > 2 else [1] * len(source)
        node_labels = data[3] if len(data) > 3 else []
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 1000) / 100,
        config.get('height', 600) / 100
    ))
    
    # Configuration des flux (adapté pour le format Matplotlib Sankey)
    # Note: Le format Matplotlib Sankey est assez différent et plus limité que celui de Plotly
    # Cette implémentation est simplifiée et ne prend en compte que le premier flux
    # Pour des diagrammes Sankey plus complexes, il est recommandé d'utiliser Plotly

    # Nous allons organiser les nœuds et les flux pour créer un diagramme simple
    # Identifier les nœuds uniques
    all_nodes = set()
    for s, t in zip(source, target):
        all_nodes.add(s)
        all_nodes.add(t)
    
    # Mappez les indices de nœuds vers des entiers consécutifs
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Créer des dictionnaires pour suivre les flux entrants et sortants
    flows = {}
    for i, (s, t, v) in enumerate(zip(source, target, value)):
        if node_map[s] not in flows:
            flows[node_map[s]] = []
        flows[node_map[s]].append((node_map[t], v))
    
    # Créer un objet Sankey
    sankey = Sankey(ax=ax, unit='', scale=0.01 * kwargs.get('scale', 1.0),
                   format='%.0f', gap=kwargs.get('gap', 0.25))
    
    # Ajouter les flux pour chaque nœud
    for node_idx, outflows in flows.items():
        # Calculer les flux entrants et sortants
        targets = [t for t, _ in outflows]
        values = [v for _, v in outflows]
        
        # Ajouter le diagramme au sankey
        sankey.add(flows=[values], labels=[node_labels[node_idx] if node_idx < len(node_labels) else f'Node {node_idx}'],
                 orientations=[0])  # 0 for horizontal
    
    # Mettre à jour le diagramme
    sankey.finish()
    
    # Ajouter le titre
    plt.title(title)
    
    # Ajuster les limites des axes
    plt.axis('equal')
    plt.axis('off')
    
    return fig
