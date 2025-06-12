"""
Implémentation des graphiques de type jauge (gauge charts) pour XPLIA
"""

def create_gauge_chart(data, config, library="plotly", **kwargs):
    """
    Crée un graphique de type jauge avec la bibliothèque spécifiée.
    
    Args:
        data: Valeur ou données pour la jauge
        config: Configuration du graphique
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de graphique
        
    Returns:
        Objet graphique selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_gauge_chart(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_gauge_chart(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_gauge_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les jauges")

# Implémentations spécifiques
def _create_plotly_gauge_chart(data, config, **kwargs):
    """Implémentation Plotly des jauges"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 500)
    height = config.get('height', 500)
    
    # Normalisation des données d'entrée
    if isinstance(data, dict):
        # Format {value: X, min: Y, max: Z, ...}
        value = data.get('value', 0)
        minimum = data.get('min', 0)
        maximum = data.get('max', 100)
        threshold = data.get('threshold', None)
        steps = data.get('steps', None)
    else:
        # Format simple: valeur directe
        value = data if isinstance(data, (int, float)) else data[0]
        minimum = kwargs.get('min', 0)
        maximum = kwargs.get('max', 100)
        threshold = kwargs.get('threshold', None)
        steps = kwargs.get('steps', None)
    
    # Configuration des étapes de couleur
    if steps is None:
        # Valeurs par défaut pour une jauge à 3 zones
        steps = [
            {'range': [minimum, maximum * 0.6], 'color': 'lightgreen'},
            {'range': [maximum * 0.6, maximum * 0.8], 'color': 'gold'},
            {'range': [maximum * 0.8, maximum], 'color': 'firebrick'}
        ]
    
    # Configuration de l'indicateur
    gauge_bar_color = kwargs.get('gauge_bar_color', 'royalblue')
    gauge_bar_thickness = kwargs.get('gauge_bar_thickness', 0.6)
    show_axis_tick_labels = kwargs.get('show_axis_tick_labels', True)
    
    # Texte à afficher au centre de la jauge
    delta_reference = kwargs.get('delta_reference', None)
    delta_config = {}
    if delta_reference is not None:
        delta = value - delta_reference
        delta_config = {
            'reference': delta_reference,
            'valueformat': kwargs.get('delta_format', '.1f'),
            'relative': kwargs.get('delta_relative', False)
        }
    
    # Configuration du mode de la jauge
    gauge_mode = kwargs.get('gauge_mode', 'gauge+number')
    
    # Création du graphique
    fig = go.Figure(go.Indicator(
        mode=gauge_mode,
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': kwargs.get('title_font_size', 24)}},
        delta=delta_config,
        gauge={
            'axis': {
                'range': [minimum, maximum],
                'tickwidth': 1,
                'tickcolor': 'darkblue',
                'visible': show_axis_tick_labels
            },
            'bar': {'color': gauge_bar_color, 'thickness': gauge_bar_thickness},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': steps,
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    # Configuration du layout
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(t=50, b=25, l=25, r=25)
    )
            
    return fig

def _create_mpl_gauge_chart(data, config, **kwargs):
    """Implémentation Matplotlib des jauges"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extraction des paramètres
    title = config.get('title', '')
    
    # Normalisation des données
    if isinstance(data, dict):
        value = data.get('value', 0)
        minimum = data.get('min', 0)
        maximum = data.get('max', 100)
        threshold = data.get('threshold', None)
        steps = data.get('steps', None)
    else:
        value = data if isinstance(data, (int, float)) else data[0]
        minimum = kwargs.get('min', 0)
        maximum = kwargs.get('max', 100)
        threshold = kwargs.get('threshold', None)
        steps = kwargs.get('steps', None)
    
    # Configuration des étapes de couleur
    if steps is None:
        # Valeurs par défaut pour une jauge à 3 zones
        steps = [
            {'range': [minimum, maximum * 0.6], 'color': 'lightgreen'},
            {'range': [maximum * 0.6, maximum * 0.8], 'color': 'gold'},
            {'range': [maximum * 0.8, maximum], 'color': 'firebrick'}
        ]
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(
        config.get('width', 500) / 100,
        config.get('height', 500) / 100
    ), subplot_kw={'projection': 'polar'})
    
    # Configuration de base pour la jauge
    # La jauge est représentée par un diagramme polaire partiel
    start_angle = np.pi/6
    end_angle = 5*np.pi/6
    
    # Rendre la jauge semi-circulaire
    ax.set_thetamin(90)  # commencer à 90° (bas)
    ax.set_thetamax(270)  # terminer à 270° (haut)
    
    # Limites du rayon
    ax.set_ylim(0, 1)
    
    # Dessiner les segments de couleur pour les étapes
    for step in steps:
        step_min, step_max = step['range']
        color = step['color']
        
        # Normaliser les valeurs
        norm_min = (step_min - minimum) / (maximum - minimum)
        norm_max = (step_max - minimum) / (maximum - minimum)
        
        # Convertir en angles
        angle_min = np.pi/2 + norm_min * np.pi
        angle_max = np.pi/2 + norm_max * np.pi
        
        # Créer un set de points pour le secteur
        theta = np.linspace(angle_min, angle_max, 30)
        radii = np.ones_like(theta) * 0.8
        
        # Dessiner le segment
        ax.fill_between(theta, 0, radii, color=color, alpha=0.8)
    
    # Dessiner la valeur actuelle
    norm_value = (value - minimum) / (maximum - minimum)
    value_angle = np.pi/2 + norm_value * np.pi
    
    # Flèche pointant vers la valeur actuelle
    ax.plot([value_angle, value_angle], [0, 0.9], 'k-', lw=2)
    ax.plot([value_angle, value_angle], [0.7, 0.9], 'k-', lw=6)
    
    # Ajouter le texte de la valeur au centre
    ax.text(np.pi, 0.2, f"{value:.1f}", ha='center', va='center', fontsize=20)
    
    # Cacher les graduations et les étiquettes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Cacher les lignes de grille
    ax.grid(False)
    
    # Cacher l'axe
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Ajouter le titre
    plt.title(title, y=0.1)
    
    return fig
