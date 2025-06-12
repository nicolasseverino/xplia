"""
Implémentation des graphiques treemap (carte proportionnelle) pour XPLIA
"""

def create_treemap_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique treemap avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique (format hiérarchique)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_treemap_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_treemap_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_treemap_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les treemaps")

# Implémentations spécifiques
def _create_plotly_treemap_chart(data, config, **kwargs):
    """Implémentation Plotly des treemaps"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 800)
    height = config.get('height', 600)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {labels: [...], parents: [...], values: [...]} ou data hiérarchique
        if all(key in data for key in ['labels', 'parents']):
            # Format explicite Plotly
            labels = data.get('labels', [])
            parents = data.get('parents', [])
            values = data.get('values', [1] * len(labels))
            colors = data.get('colors', None)
        else:
            # Conversion du format hiérarchique vers format Plotly
            labels, parents, values, colors = _convert_hierarchical_to_plotly(data)
    else:
        # Supposer que c'est déjà au format attendu [labels, parents, values]
        labels = data[0] if len(data) > 0 else []
        parents = data[1] if len(data) > 1 else [''] * len(labels)
        values = data[2] if len(data) > 2 else [1] * len(labels)
        colors = data[3] if len(data) > 3 else None
    
    # Configuration du treemap
    treemap_config = dict(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues=kwargs.get('branchvalues', 'total'),  # 'total' ou 'remainder'
        hovertemplate=kwargs.get('hovertemplate', '%{label}<br>%{value}<br>%{percentParent:.1%} of parent<extra></extra>'),
        textinfo=kwargs.get('textinfo', 'label+value+percent parent+percent root'),
        marker=dict(
            pad=kwargs.get('pad', 3),
            line=dict(width=kwargs.get('line_width', 1))
        ),
        pathbar=dict(visible=kwargs.get('show_pathbar', True))
    )
    
    # Ajout de couleurs personnalisées si disponibles
    if colors:
        treemap_config['marker']['colors'] = colors
    elif 'colorscale' in kwargs:
        treemap_config['marker']['colorscale'] = kwargs['colorscale']
    
    # Création du graphique
    fig = go.Figure(go.Treemap(**treemap_config))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        margin=dict(t=50, b=25, l=25, r=25)
    )
            
    return fig

def _create_mpl_treemap_chart(data, config, **kwargs):
    """Implémentation Matplotlib des treemaps"""
    import matplotlib.pyplot as plt
    import squarify  # Nécessite pip install squarify
    import pandas as pd
    
    # Extraction des paramètres
    title = config.get('title', '')
    
    # La structure des données doit être adaptée pour squarify
    # Format pour squarify: valeurs (sizes), labels, et couleurs optionnelles
    if isinstance(data, dict):
        if all(key in data for key in ['labels', 'parents']):
            # Convertir le format Plotly en format squarify (simplifié, uniquement un niveau)
            df = pd.DataFrame({
                'labels': data.get('labels', []),
                'parents': data.get('parents', []),
                'values': data.get('values', [])
            })
            
            # Filtrer uniquement le premier niveau de hiérarchie
            root_level = df[df['parents'] == '']
            sizes = root_level['values'].tolist()
            labels = root_level['labels'].tolist()
            colors = data.get('colors', None)
            if colors and len(colors) == len(df):
                colors = [colors[i] for i in root_level.index]
        else:
            # Format hiérarchique - prendre uniquement les valeurs de premier niveau
            sizes = [item.get('value', 1) for item in data.get('children', [])]
            labels = [item.get('name', '') for item in data.get('children', [])]
            colors = [item.get('color', None) for item in data.get('children', [])]
            if all(c is None for c in colors):
                colors = None
    else:
        # Supposer que data est une liste de valeurs directe
        sizes = data if isinstance(data[0], (int, float)) else data[0]
        labels = kwargs.get('labels', [''] * len(sizes))
        colors = kwargs.get('colors', None)
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800) / 100,
        config.get('height', 600) / 100
    ))
    
    # Options de couleurs
    if colors is None:
        cmap = plt.cm.get_cmap(kwargs.get('colormap', 'viridis'))
        norm_sizes = [(size - min(sizes)) / (max(sizes) - min(sizes) + 0.0001) for size in sizes]
        colors = [cmap(i) for i in norm_sizes]
    
    # Création du treemap
    squarify.plot(
        sizes=sizes,
        label=labels,
        color=colors,
        alpha=kwargs.get('opacity', 0.8),
        pad=kwargs.get('pad', True),
        ax=ax
    )
    
    # Configuration des titres et labels
    ax.set_title(title)
    
    # Suppression des axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    return fig

def _convert_hierarchical_to_plotly(data):
    """
    Convertit des données hiérarchiques en format Plotly pour treemap.
    
    Args:
        data: Données hiérarchiques {name: "root", children: [{name: "A", value: 1}, ...]}
        
    Returns:
        tuple: (labels, parents, values, colors)
    """
    labels = []
    parents = []
    values = []
    colors = []
    
    # Fonction récursive pour parcourir la hiérarchie
    def _process_node(node, parent=""):
        name = node.get('name', '')
        value = node.get('value', None)
        color = node.get('color', None)
        children = node.get('children', [])
        
        # Ajouter le nœud actuel
        labels.append(name)
        parents.append(parent)
        colors.append(color)
        
        if value is not None:
            values.append(value)
        elif not children:
            values.append(1)  # Valeur par défaut pour les feuilles sans valeur
        else:
            # Pour les nœuds intermédiaires, la valeur est la somme des enfants
            sum_children = sum([child.get('value', 0) for child in children if 'value' in child])
            if sum_children > 0:
                values.append(sum_children)
            else:
                values.append(len(children))  # Si pas de valeurs, utiliser le nombre d'enfants
        
        # Traiter les enfants
        for child in children:
            _process_node(child, name)
    
    # Traiter à partir du nœud racine
    _process_node(data)
    
    # Si trop peu de couleurs sont spécifiées, remplir avec None
    if not all(c is None for c in colors) and len(colors) < len(labels):
        colors.extend([None] * (len(labels) - len(colors)))
    
    # Si trop peu de valeurs sont spécifiées, remplir avec 1
    if len(values) < len(labels):
        values.extend([1] * (len(labels) - len(values)))
    
    return labels, parents, values, colors
