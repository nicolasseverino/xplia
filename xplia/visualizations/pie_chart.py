"""
Implémentation des graphiques circulaires (pie charts) pour XPLIA
"""

def create_pie_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique circulaire avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique (dict, DataFrame, etc.)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_pie_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_pie_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_pie_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les graphiques circulaires")

# Implémentations spécifiques
def _create_plotly_pie_chart(data, config, **kwargs):
    """Implémentation Plotly des graphiques circulaires"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 800)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        labels = data.get('labels', [])
        values = data.get('values', [])
    else:
        # Format [labels, values]
        labels = data[0]
        values = data[1]
    
    # Récupération des options personnalisées
    hole = kwargs.get('hole', 0)  # 0 pour pie, >0 pour donut
    text_info = kwargs.get('text_info', 'percent')
    pull = kwargs.get('pull', None)  # [0, 0, 0.2, 0] pour extraire une portion
    
    # Création du graphique
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=hole,
        textinfo=text_info,
        pull=pull,
        marker_colors=kwargs.get('colors', config.get('color_palette', [])),
        textfont=dict(size=kwargs.get('text_size', 12)),
        hoverinfo=kwargs.get('hoverinfo', 'label+percent+value'),
        rotation=kwargs.get('rotation', 0)
    )])
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=kwargs.get('show_legend', True),
        legend=dict(
            orientation=kwargs.get('legend_orientation', 'v'),
            yanchor=kwargs.get('legend_yanchor', 'auto'),
            xanchor=kwargs.get('legend_xanchor', 'auto'),
            x=kwargs.get('legend_x', 1.02),
            y=kwargs.get('legend_y', 1)
        ),
        annotations=kwargs.get('annotations', [])
    )
    
    return fig

def _create_mpl_pie_chart(data, config, **kwargs):
    """Implémentation Matplotlib des graphiques circulaires"""
    import matplotlib.pyplot as plt
    
    # Création d'une nouvelle figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800)/100,
        config.get('height', 500)/100
    ))
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        labels = data.get('labels', [])
        values = data.get('values', [])
    else:
        # Format [labels, values]
        labels = data[0]
        values = data[1]
    
    # Récupération des options personnalisées
    hole = kwargs.get('hole', 0)  # Pour donut, équivalent à Matplotlib wedgeprops
    explode = kwargs.get('pull', None)  # [0, 0, 0.1, 0] pour extraire une portion
    
    # Options spécifiques pour Matplotlib
    wedgeprops = {}
    if hole > 0:
        wedgeprops = {'width': hole}
    
    # Création du graphique
    ax.pie(
        values, 
        labels=labels if kwargs.get('show_labels', True) else None,
        autopct=kwargs.get('autopct', '%1.1f%%'),
        startangle=kwargs.get('rotation', 0),
        shadow=kwargs.get('shadow', False),
        explode=explode,
        colors=kwargs.get('colors', config.get('color_palette', [])),
        textprops={'fontsize': kwargs.get('text_size', 12)},
        wedgeprops=wedgeprops
    )
    
    # Titre
    if config.get('title'):
        ax.set_title(config.get('title'))
    
    # Égaliser les axes pour un cercle parfait
    ax.axis('equal')
    
    # Légende
    if kwargs.get('show_legend', True):
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
        
    return fig
