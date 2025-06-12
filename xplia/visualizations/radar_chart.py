"""
Implémentation des graphiques radar (diagrammes en toile d'araignée) pour XPLIA
"""

def create_radar_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique radar avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique 
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_radar_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_radar_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_radar_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les graphiques radar")

# Implémentations spécifiques
def _create_plotly_radar_chart(data, config, **kwargs):
    """Implémentation Plotly des graphiques radar"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 800)
    height = config.get('height', 600)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {categories: [...], series: [{name: "...", values: [...]}, ...]}
        categories = data.get('categories', [])
        series = data.get('series', [])
    else:
        # Format [categories, [values_series1, values_series2, ...], [names]]
        categories = data[0]
        values_series = data[1]
        names = data[2] if len(data) > 2 else [f"Série {i+1}" for i in range(len(values_series))]
        series = [{"name": names[i], "values": values} for i, values in enumerate(values_series)]
    
    # Création du graphique
    fig = go.Figure()
    
    # Ajout de chaque série
    for i, serie in enumerate(series):
        color = serie.get('color', config.get('color_palette', [])[i % len(config.get('color_palette', ['#1f77b4']))])
        
        fig.add_trace(go.Scatterpolar(
            r=serie.get('values', []),
            theta=categories,
            fill=kwargs.get('fill', 'toself'),
            name=serie.get('name', f'Série {i}'),
            line=dict(color=color, width=serie.get('line_width', 2)),
            marker=dict(
                size=serie.get('marker_size', 6),
                symbol=serie.get('marker_symbol', 'circle'),
                color=color,
                opacity=serie.get('opacity', 0.8)
            ),
            opacity=serie.get('fill_opacity', 0.6),
            hoverinfo=kwargs.get('hoverinfo', 'all')
        ))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=kwargs.get('range', [0, max([max(s.get('values', [0])) for s in series]) * 1.1])
            )
        ),
        showlegend=kwargs.get('show_legend', True)
    )
    
    return fig

def _create_mpl_radar_chart(data, config, **kwargs):
    """Implémentation Matplotlib des graphiques radar"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        categories = data.get('categories', [])
        series = data.get('series', [])
    else:
        # Format [categories, [values_series1, values_series2, ...], [names]]
        categories = data[0]
        values_series = data[1]
        names = data[2] if len(data) > 2 else [f"Série {i+1}" for i in range(len(values_series))]
        series = [{"name": names[i], "values": values} for i, values in enumerate(values_series)]
    
    # Nombre de variables
    N = len(categories)
    
    # Création d'un angle pour chaque variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le cercle
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800) / 100,
        config.get('height', 600) / 100
    ), subplot_kw=dict(polar=True))
    
    # Calculer la plage de valeurs pour toutes les séries
    max_value = max([max(s.get('values', [0])) for s in series])
    if kwargs.get('range'):
        ax.set_ylim(kwargs.get('range'))
    else:
        ax.set_ylim([0, max_value * 1.1])
    
    # Placer les axes à la bonne position
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Tracer chaque série
    for i, serie in enumerate(series):
        values = serie.get('values', [])
        values += values[:1]  # Fermer le cercle
        
        color = serie.get('color', config.get('color_palette', [])[i % len(config.get('color_palette', ['#1f77b4']))])
        
        ax.plot(angles, values, 'o-', linewidth=serie.get('line_width', 2), 
                label=serie.get('name', f'Série {i}'), color=color)
        
        if kwargs.get('fill', True):
            ax.fill(angles, values, color=color, alpha=serie.get('fill_opacity', 0.4))
    
    # Ajout du titre et de la légende
    ax.set_title(title)
    
    if kwargs.get('show_legend', True):
        ax.legend(loc=kwargs.get('legend_loc', 'best'))
    
    # Option pour personnaliser les grilles
    if kwargs.get('grid', True):
        ax.grid(True)
    
    return fig
