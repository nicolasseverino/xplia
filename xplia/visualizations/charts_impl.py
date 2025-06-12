"""
Implémentations des générateurs de graphiques pour XPLIA
"""

def create_bar_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique à barres avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique (dict, DataFrame, etc.)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_bar_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_bar_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_bar_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les graphiques à barres")

# Implémentations spécifiques
def _create_plotly_bar_chart(data, config, **kwargs):
    """Implémentation Plotly des graphiques à barres"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    width = config.get('width', 800)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    x_data = data.get('x', []) if isinstance(data, dict) else data[0]
    y_data = data.get('y', []) if isinstance(data, dict) else data[1]
    
    # Création du graphique
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_data,
        y=y_data,
        marker_color=kwargs.get('color', config.get('color_palette', ['#1f77b4'])[0]),
        name=kwargs.get('name', ''),
        text=kwargs.get('text', None),
        textposition=kwargs.get('textposition', 'auto'),
    ))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        bargap=kwargs.get('bargap', 0.2),
        bargroupgap=kwargs.get('bargroupgap', 0.1)
    )
    
    return fig
