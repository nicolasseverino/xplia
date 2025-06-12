"""
Implémentation des histogrammes pour XPLIA
"""

def create_histogram_chart(data, config, library="plotly", **kwargs):
    """
    Crée un histogramme avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour l'histogramme
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_histogram_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_histogram_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_histogram_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les histogrammes")

# Implémentations spécifiques
def _create_plotly_histogram_chart(data, config, **kwargs):
    """Implémentation Plotly des histogrammes"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    width = config.get('width', 800)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {x: [...], name: "...", ...} ou {series: [{x: [...], name: "...", ...}, ...]}
        if 'series' in data:
            # Multiple séries
            series = data.get('series', [])
        else:
            # Une seule série
            series = [data]
    else:
        # Format [data] ou [[data1], [data2], ...]
        if all(isinstance(d, list) for d in data):
            # Format [[data1], [data2], ...]
            series = [{'x': d, 'name': f'Série {i+1}'} for i, d in enumerate(data)]
        else:
            # Format [data]
            series = [{'x': data, 'name': kwargs.get('name', 'Données')}]
    
    # Création de la figure
    fig = go.Figure()
    
    # Ajout de chaque série d'histogramme
    for i, serie in enumerate(series):
        color = serie.get('color', config.get('color_palette', [])[i % len(config.get('color_palette', ['#1f77b4']))])
        
        fig.add_trace(go.Histogram(
            x=serie.get('x', []),
            name=serie.get('name', f'Série {i}'),
            marker_color=color,
            opacity=kwargs.get('opacity', 0.7),
            nbinsx=kwargs.get('nbins', None),
            histnorm=kwargs.get('histnorm', None),  # '', 'percent', 'probability', 'density', 'probability density'
            cumulative_enabled=kwargs.get('cumulative', False),
            autobinx=True if kwargs.get('nbins') is None else False,
            xbins=serie.get('xbins', {}),
            marker_line_width=kwargs.get('marker_line_width', 1),
            marker_line_color=kwargs.get('marker_line_color', "#FFFFFF"),
            hoverinfo=kwargs.get('hoverinfo', 'x+y+name')
        ))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        barmode=kwargs.get('barmode', 'overlay'),  # 'overlay', 'stack', 'group', 'relative'
        bargap=kwargs.get('bargap', 0),
        bargroupgap=kwargs.get('bargroupgap', 0)
    )
    
    # Configuration des axes
    if 'x_range' in kwargs:
        fig.update_xaxes(range=kwargs['x_range'])
    if 'y_range' in kwargs:
        fig.update_yaxes(range=kwargs['y_range'])
        
    return fig

def _create_mpl_histogram_chart(data, config, **kwargs):
    """Implémentation Matplotlib des histogrammes"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    
    # Normalisation des données (similaire à Plotly)
    if isinstance(data, dict):
        if 'series' in data:
            series = data.get('series', [])
        else:
            series = [data]
    else:
        if all(isinstance(d, list) for d in data):
            series = [{'x': d, 'name': f'Série {i+1}'} for i, d in enumerate(data)]
        else:
            series = [{'x': data, 'name': kwargs.get('name', 'Données')}]
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800) / 100,
        config.get('height', 500) / 100
    ))
    
    # Options de l'histogramme
    bins = kwargs.get('nbins', 'auto')
    alpha = kwargs.get('opacity', 0.7)
    density = kwargs.get('histnorm', None) in ['density', 'probability', 'probability density']
    cumulative = kwargs.get('cumulative', False)
    histtype = 'bar'
    
    if kwargs.get('barmode') == 'stack':
        stacked = True
    else:
        stacked = False
    
    # Ajout de chaque série d'histogramme
    for i, serie in enumerate(series):
        color = serie.get('color', config.get('color_palette', [])[i % len(config.get('color_palette', ['#1f77b4']))])
        
        ax.hist(
            serie.get('x', []),
            bins=bins,
            alpha=alpha,
            density=density,
            cumulative=cumulative,
            histtype=histtype,
            color=color,
            label=serie.get('name', f'Série {i}'),
            edgecolor=kwargs.get('marker_line_color', 'white'),
            linewidth=kwargs.get('marker_line_width', 1),
            stacked=stacked
        )
    
    # Configuration des titres et labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Ajout de la grille
    if kwargs.get('grid', True):
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # Ajout de la légende
    if kwargs.get('show_legend', True) and len(series) > 1:
        ax.legend()
    
    # Configuration des axes
    if 'x_range' in kwargs:
        ax.set_xlim(kwargs['x_range'])
    if 'y_range' in kwargs:
        ax.set_ylim(kwargs['y_range'])
    
    return fig
