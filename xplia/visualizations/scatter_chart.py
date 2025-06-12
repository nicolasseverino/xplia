"""
Implémentation des graphiques de dispersion (scatter plots) pour XPLIA
"""

def create_scatter_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique de dispersion avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique (dict, DataFrame, etc.)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_scatter_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_scatter_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_scatter_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les graphiques de dispersion")

# Implémentations spécifiques
def _create_plotly_scatter_chart(data, config, **kwargs):
    """Implémentation Plotly des graphiques de dispersion"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    width = config.get('width', 800)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {x: [...], y: [...], z: [...]} ou [série1, série2, ...]
        if 'series' in data:
            # Multiple séries
            series = data.get('series', [])
        else:
            # Une seule série
            x_data = data.get('x', [])
            y_data = data.get('y', [])
            z_data = data.get('z', None)  # Pour scatter3d
            series = [{'x': x_data, 'y': y_data, 'z': z_data, 'name': kwargs.get('name', '')}]
    else:
        # Format [[x], [y]] ou [[x1, y1, ...], [x2, y2, ...], ...] pour multiple séries
        if len(data) == 2 and all(isinstance(d, list) for d in data):
            # Une seule série
            x_data = data[0]
            y_data = data[1]
            series = [{'x': x_data, 'y': y_data, 'name': kwargs.get('name', '')}]
        else:
            # Multiple séries
            series = []
            for i, serie in enumerate(data):
                if isinstance(serie, dict):
                    series.append(serie)
                else:
                    series.append({'x': serie[0], 'y': serie[1], 'name': f'Série {i}'})
    
    # Création du graphique
    fig = go.Figure()
    
    # Déterminer si c'est un scatter3D
    is_3d = any('z' in s and s['z'] is not None for s in series if isinstance(s, dict))
    
    # Ajout de chaque série
    for i, serie in enumerate(series):
        # Options par défaut
        marker_size = serie.get('size', kwargs.get('marker_size', 8))
        marker_symbol = serie.get('symbol', kwargs.get('marker_symbol', 'circle'))
        color = serie.get('color', config.get('color_palette', [])[i % len(config.get('color_palette', ['#1f77b4']))])
        
        if is_3d and 'z' in serie and serie['z'] is not None:
            fig.add_trace(go.Scatter3d(
                x=serie.get('x', []),
                y=serie.get('y', []),
                z=serie.get('z', []),
                mode=serie.get('mode', kwargs.get('mode', 'markers')),
                name=serie.get('name', f'Série {i}'),
                marker=dict(
                    size=marker_size,
                    symbol=marker_symbol,
                    color=color,
                    line=dict(width=serie.get('line_width', 0.5), color='DarkSlateGrey'),
                    opacity=serie.get('opacity', kwargs.get('opacity', 0.8))
                ),
                text=serie.get('text', None),
                hovertext=serie.get('hovertext', None),
                hoverinfo=serie.get('hoverinfo', kwargs.get('hoverinfo', 'all'))
            ))
        else:
            fig.add_trace(go.Scatter(
                x=serie.get('x', []),
                y=serie.get('y', []),
                mode=serie.get('mode', kwargs.get('mode', 'markers')),
                name=serie.get('name', f'Série {i}'),
                marker=dict(
                    size=marker_size,
                    symbol=marker_symbol,
                    color=color,
                    line=dict(width=serie.get('line_width', 0.5), color='DarkSlateGrey'),
                    opacity=serie.get('opacity', kwargs.get('opacity', 0.8))
                ),
                text=serie.get('text', None),
                hovertext=serie.get('hovertext', None),
                hoverinfo=serie.get('hoverinfo', kwargs.get('hoverinfo', 'all'))
            ))
    
    # Configuration du layout
    layout_args = dict(
        title=title,
        width=width,
        height=height,
        hovermode=kwargs.get('hovermode', 'closest'),
        showlegend=kwargs.get('show_legend', True)
    )
    
    if is_3d:
        layout_args.update({
            'scene': dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=kwargs.get('z_label', 'Z')
            )
        })
    else:
        layout_args.update({
            'xaxis_title': x_label,
            'yaxis_title': y_label
        })
    
    fig.update_layout(**layout_args)
    
    # Si spécifié, ajouter des annotations
    if 'annotations' in kwargs:
        for annotation in kwargs['annotations']:
            fig.add_annotation(**annotation)
            
    # Configuration des axes
    if 'x_range' in kwargs and not is_3d:
        fig.update_xaxes(range=kwargs['x_range'])
    if 'y_range' in kwargs and not is_3d:
        fig.update_yaxes(range=kwargs['y_range'])
        
    return fig

def _create_mpl_scatter_chart(data, config, **kwargs):
    """Implémentation Matplotlib des graphiques de dispersion"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    
    # Normalisation des données d'entrée (similaire à Plotly)
    if isinstance(data, dict):
        if 'series' in data:
            series = data.get('series', [])
        else:
            x_data = data.get('x', [])
            y_data = data.get('y', [])
            z_data = data.get('z', None)
            series = [{'x': x_data, 'y': y_data, 'z': z_data, 'name': kwargs.get('name', '')}]
    else:
        if len(data) == 2 and all(isinstance(d, list) for d in data):
            x_data = data[0]
            y_data = data[1]
            series = [{'x': x_data, 'y': y_data, 'name': kwargs.get('name', '')}]
        else:
            series = []
            for i, serie in enumerate(data):
                if isinstance(serie, dict):
                    series.append(serie)
                else:
                    series.append({'x': serie[0], 'y': serie[1], 'name': f'Série {i}'})
    
    # Déterminer si c'est un scatter3D
    is_3d = any('z' in s and s['z'] is not None for s in series if isinstance(s, dict))
    
    # Création de la figure
    fig = plt.figure(figsize=(
        config.get('width', 800) / 100,
        config.get('height', 500) / 100
    ))
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Ajout de chaque série
    for i, serie in enumerate(series):
        color = serie.get('color', config.get('color_palette', [])[i % len(config.get('color_palette', ['#1f77b4']))])
        marker = serie.get('symbol', kwargs.get('marker_symbol', 'o'))
        size = serie.get('size', kwargs.get('marker_size', 30))
        alpha = serie.get('opacity', kwargs.get('opacity', 0.8))
        
        if is_3d and 'z' in serie and serie['z'] is not None:
            ax.scatter(
                serie.get('x', []),
                serie.get('y', []),
                serie.get('z', []),
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                label=serie.get('name', f'Série {i}')
            )
        else:
            ax.scatter(
                serie.get('x', []),
                serie.get('y', []),
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                label=serie.get('name', f'Série {i}')
            )
    
    # Configuration des titres et labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if is_3d:
        ax.set_zlabel(kwargs.get('z_label', 'Z'))
    
    # Configuration des axes
    if 'x_range' in kwargs and not is_3d:
        ax.set_xlim(kwargs['x_range'])
    if 'y_range' in kwargs and not is_3d:
        ax.set_ylim(kwargs['y_range'])
    
    # Légende
    if kwargs.get('show_legend', True):
        ax.legend()
        
    # Grille
    if kwargs.get('grid', True):
        ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig
