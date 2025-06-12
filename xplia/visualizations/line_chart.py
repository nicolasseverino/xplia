"""
Implémentation des graphiques en ligne pour XPLIA
"""

def create_line_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique en ligne avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique (dict, DataFrame, etc.)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_line_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_line_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_line_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les graphiques en ligne")

# Implémentations spécifiques
def _create_plotly_line_chart(data, config, **kwargs):
    """Implémentation Plotly des graphiques en ligne"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    width = config.get('width', 800)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {x: [...], y: [...], series: [...]}
        x_data = data.get('x', [])
        series = data.get('series', [{'name': '', 'y': data.get('y', [])}])
    else:
        # Format [[x], [y1], [y2], ...]
        x_data = data[0]
        series = [{'name': f'Série {i}', 'y': data[i]} for i in range(1, len(data))]
    
    # Création du graphique
    fig = go.Figure()
    
    # Ajout de chaque série
    for i, serie in enumerate(series):
        color = kwargs.get('colors', config.get('color_palette', []))[i % len(config.get('color_palette', ['#1f77b4']))]
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=serie.get('y', []),
            mode=kwargs.get('mode', 'lines+markers'),
            name=serie.get('name', f'Série {i}'),
            line=dict(
                color=color,
                width=kwargs.get('line_width', 2),
                dash=kwargs.get('dash', None)
            ),
            marker=dict(
                size=kwargs.get('marker_size', 6),
                symbol=kwargs.get('marker_symbol', 'circle')
            )
        ))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        hovermode=kwargs.get('hovermode', 'closest'),
        legend=dict(
            orientation=kwargs.get('legend_orientation', 'v'),
            yanchor=kwargs.get('legend_yanchor', 'auto'),
            xanchor=kwargs.get('legend_xanchor', 'auto'),
            x=kwargs.get('legend_x', 1.02),
            y=kwargs.get('legend_y', 1)
        )
    )
    
    # Si spécifié, ajouter des annotations
    if 'annotations' in kwargs:
        for annotation in kwargs['annotations']:
            fig.add_annotation(**annotation)
            
    # Configuration des axes
    if 'x_range' in kwargs:
        fig.update_xaxes(range=kwargs['x_range'])
    if 'y_range' in kwargs:
        fig.update_yaxes(range=kwargs['y_range'])
        
    return fig
