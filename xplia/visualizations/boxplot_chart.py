"""
Implémentation des box plots (boîtes à moustaches) pour XPLIA
"""

def create_boxplot_chart(data, config, library="plotly", **kwargs):
    """
    Crée un box plot (boîte à moustaches) avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_boxplot_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_boxplot_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_boxplot_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les box plots")

# Implémentations spécifiques
def _create_plotly_boxplot_chart(data, config, **kwargs):
    """Implémentation Plotly des box plots"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    width = config.get('width', 800)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {x: [...], y: [...]} ou {data: [...], names: [...]}
        if 'data' in data and 'names' in data:
            # Multiple boîtes avec noms
            y_data = data.get('data', [])
            x_data = data.get('names', [])
            orientation = kwargs.get('orientation', 'v')
        elif 'y' in data:
            # Une seule série
            y_data = data.get('y', [])
            x_data = data.get('x', [])
            orientation = kwargs.get('orientation', 'v')
        else:
            # Dictionnaire de séries {nom_serie: données}
            y_data = list(data.values())
            x_data = list(data.keys())
            orientation = kwargs.get('orientation', 'v')
    else:
        # Format [data1, data2, ...] ou [[data1], [data2], ...]
        if all(isinstance(d, list) for d in data):
            # Chaque sous-liste est un dataset
            y_data = data
            x_data = kwargs.get('names', [f"Série {i+1}" for i in range(len(data))])
        else:
            # Une seule série
            y_data = [data]
            x_data = kwargs.get('names', ["Données"])
        orientation = kwargs.get('orientation', 'v')
    
    # Création du graphique
    fig = go.Figure()
    
    # Ajout des box plots
    if orientation == 'v':
        for i, (y, name) in enumerate(zip(y_data if isinstance(y_data[0], list) else [y_data], 
                                        x_data if isinstance(x_data, list) else [x_data])):
            color = kwargs.get('colors', config.get('color_palette', []))[i % len(config.get('color_palette', ['#1f77b4']))]
            fig.add_trace(go.Box(
                y=y,
                name=name,
                boxmean=kwargs.get('boxmean', True),
                marker_color=color,
                line_color=kwargs.get('line_color', color),
                fillcolor=kwargs.get('fillcolor', color),
                opacity=kwargs.get('opacity', 0.7),
                boxpoints=kwargs.get('boxpoints', 'outliers'),
                jitter=kwargs.get('jitter', 0.3),
                pointpos=kwargs.get('pointpos', -1.8),
                orientation=orientation,
                quartilemethod=kwargs.get('quartilemethod', 'linear')
            ))
    else:
        for i, (x, name) in enumerate(zip(y_data if isinstance(y_data[0], list) else [y_data], 
                                        x_data if isinstance(x_data, list) else [x_data])):
            color = kwargs.get('colors', config.get('color_palette', []))[i % len(config.get('color_palette', ['#1f77b4']))]
            fig.add_trace(go.Box(
                x=x,
                name=name,
                boxmean=kwargs.get('boxmean', True),
                marker_color=color,
                line_color=kwargs.get('line_color', color),
                fillcolor=kwargs.get('fillcolor', color),
                opacity=kwargs.get('opacity', 0.7),
                boxpoints=kwargs.get('boxpoints', 'outliers'),
                jitter=kwargs.get('jitter', 0.3),
                pointpos=kwargs.get('pointpos', -1.8),
                orientation=orientation,
                quartilemethod=kwargs.get('quartilemethod', 'linear')
            ))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label if orientation == 'v' else y_label,
        yaxis_title=y_label if orientation == 'v' else x_label,
        width=width,
        height=height,
        boxmode=kwargs.get('boxmode', 'group'),
        boxgap=kwargs.get('boxgap', 0.1),
        boxgroupgap=kwargs.get('boxgroupgap', 0.1)
    )
    
    return fig

def _create_mpl_boxplot_chart(data, config, **kwargs):
    """Implémentation Matplotlib des box plots"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    
    # Normalisation des données (similaire à Plotly)
    if isinstance(data, dict):
        if 'data' in data and 'names' in data:
            y_data = data.get('data', [])
            x_data = data.get('names', [])
        elif 'y' in data:
            y_data = [data.get('y', [])]
            x_data = [data.get('x', "")]
        else:
            y_data = list(data.values())
            x_data = list(data.keys())
    else:
        if all(isinstance(d, list) for d in data):
            y_data = data
            x_data = kwargs.get('names', [f"Série {i+1}" for i in range(len(data))])
        else:
            y_data = [data]
            x_data = kwargs.get('names', ["Données"])
    
    # Orientation
    orientation = kwargs.get('orientation', 'v')
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800) / 100,
        config.get('height', 500) / 100
    ))
    
    # Options de style
    boxprops = dict(linestyle='-', linewidth=1.5)
    flierprops = dict(marker='o', markersize=6, markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick')
    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    
    # Ajout des couleurs personnalisées si disponibles
    colors = kwargs.get('colors', config.get('color_palette', []))
    if colors:
        patches = []
        for i in range(len(y_data)):
            color = colors[i % len(colors)]
            bp_dict = {
                'boxprops': dict(facecolor=color, **boxprops),
                'flierprops': flierprops,
                'medianprops': medianprops,
                'meanprops': meanprops,
                'patch_artist': True
            }
            patches.append(bp_dict)
    else:
        patches = [dict(boxprops=boxprops, flierprops=flierprops, 
                        medianprops=medianprops, meanprops=meanprops)] * len(y_data)
    
    # Création du box plot
    if orientation == 'v':
        bp = ax.boxplot(y_data, 
                       labels=x_data, 
                       notch=kwargs.get('notch', False),
                       vert=True, 
                       patch_artist=True,
                       showmeans=kwargs.get('boxmean', True),
                       showfliers=kwargs.get('boxpoints', 'outliers') != 'False',
                       widths=kwargs.get('width', 0.5))
    else:
        bp = ax.boxplot(y_data,
                       labels=x_data,
                       notch=kwargs.get('notch', False),
                       vert=False,
                       patch_artist=True, 
                       showmeans=kwargs.get('boxmean', True),
                       showfliers=kwargs.get('boxpoints', 'outliers') != 'False',
                       widths=kwargs.get('width', 0.5))
    
    # Application des couleurs
    for i, box in enumerate(bp['boxes']):
        box_color = colors[i % len(colors)] if colors else 'lightblue'
        box.set(facecolor=box_color, alpha=kwargs.get('opacity', 0.7))
    
    # Configuration des titres et labels
    ax.set_title(title)
    if orientation == 'v':
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    
    # Ajout de la grille
    if kwargs.get('grid', True):
        ax.grid(axis='y' if orientation == 'v' else 'x', linestyle='--', alpha=0.7)
    
    return fig
