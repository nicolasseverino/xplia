"""
Implémentation des cartes thermiques (heatmaps) pour XPLIA
"""

def create_heatmap(data, config, library="plotly", **kwargs):
    """
    Crée une carte thermique avec la bibliothèque spécifiée.
    
    Args:
        data: Données pour le graphique (matrice 2D)
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_heatmap(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_heatmap(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_heatmap(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les cartes thermiques")

# Implémentations spécifiques
def _create_plotly_heatmap(data, config, **kwargs):
    """Implémentation Plotly des cartes thermiques"""
    import plotly.graph_objects as go
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    width = config.get('width', 800)
    height = config.get('height', 600)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {z: [...], x: [...], y: [...]}
        z_data = data.get('z', [])
        x_data = data.get('x', list(range(len(z_data[0]) if z_data and len(z_data) > 0 else 0)))
        y_data = data.get('y', list(range(len(z_data) if z_data else 0)))
    else:
        # Format matrice 2D [z]
        z_data = data
        x_data = kwargs.get('x_labels', list(range(len(z_data[0]) if z_data and len(z_data) > 0 else 0)))
        y_data = kwargs.get('y_labels', list(range(len(z_data) if z_data else 0)))
    
    # Vérification pour assurer que z_data est une liste de listes (matrice 2D)
    if not all(isinstance(row, list) for row in z_data):
        if isinstance(z_data, np.ndarray):
            z_data = z_data.tolist()
        else:
            raise ValueError("Les données z doivent être une matrice 2D (liste de listes)")
    
    # Récupération des options personnalisées
    colorscale = kwargs.get('colorscale', 'Viridis')
    showscale = kwargs.get('showscale', True)
    text_matrix = kwargs.get('text', None)
    
    # Création du graphique
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_data,
        colorscale=colorscale,
        showscale=showscale,
        text=text_matrix,
        hovertemplate=kwargs.get('hovertemplate', 
                                 'x: %{x}<br>' +
                                 'y: %{y}<br>' +
                                 'z: %{z}<br>' +
                                 '%{text}<extra></extra>'),
        colorbar=dict(
            title=kwargs.get('colorbar_title', ''),
            titleside=kwargs.get('colorbar_titleside', 'right'),
            ticks=kwargs.get('colorbar_ticks', "outside"),
            tickfont=dict(
                size=kwargs.get('colorbar_tickfont_size', 12)
            )
        ),
        zauto=kwargs.get('zauto', True),
        zmin=kwargs.get('zmin', None),
        zmax=kwargs.get('zmax', None)
    ))
    
    # Configuration du layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        xaxis=dict(
            tickangle=kwargs.get('x_tickangle', -45)
        )
    )
            
    return fig

def _create_mpl_heatmap(data, config, **kwargs):
    """Implémentation Matplotlib des cartes thermiques"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        z_data = np.array(data.get('z', []))
        x_data = data.get('x', list(range(z_data.shape[1] if z_data.size > 0 else 0)))
        y_data = data.get('y', list(range(z_data.shape[0] if z_data.size > 0 else 0)))
    else:
        z_data = np.array(data)
        x_data = kwargs.get('x_labels', list(range(z_data.shape[1] if z_data.size > 0 else 0)))
        y_data = kwargs.get('y_labels', list(range(z_data.shape[0] if z_data.size > 0 else 0)))
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800) / 100,
        config.get('height', 600) / 100
    ))
    
    # Options de coloration
    cmap = kwargs.get('colorscale', kwargs.get('cmap', 'viridis'))
    vmin = kwargs.get('zmin', None)
    vmax = kwargs.get('zmax', None)
    
    # Création de la carte thermique
    im = ax.imshow(z_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                   aspect=kwargs.get('aspect', 'auto'))
    
    # Ajouter une barre de couleur
    if kwargs.get('showscale', True):
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(kwargs.get('colorbar_title', ''))
    
    # Configuration des titres et labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Configurer les ticks pour x et y si spécifiés
    if len(x_data) > 0:
        if len(x_data) <= 30:  # Éviter trop de labels
            ax.set_xticks(range(len(x_data)))
            ax.set_xticklabels(x_data, rotation=kwargs.get('x_tickangle', 45))
    
    if len(y_data) > 0:
        if len(y_data) <= 30:  # Éviter trop de labels
            ax.set_yticks(range(len(y_data)))
            ax.set_yticklabels(y_data)
    
    # Afficher les valeurs dans les cellules si demandé
    if kwargs.get('show_values', False):
        text_kw = kwargs.get('text_kw', {})
        text_format = kwargs.get('text_format', '{:.2f}')
        
        # Pour chaque cellule
        for i in range(z_data.shape[0]):
            for j in range(z_data.shape[1]):
                value = z_data[i, j]
                text_color = 'black' if im.norm(value) > 0.5 else 'white'
                ax.text(j, i, text_format.format(value), ha="center", va="center", 
                        color=text_color, **text_kw)
    
    return fig
