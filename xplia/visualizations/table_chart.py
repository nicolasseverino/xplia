"""
Implémentation des tableaux de données pour XPLIA
"""
import pandas as pd

def create_table_chart(data, config, library="plotly", **kwargs):
    """
    Crée un tableau de données avec la bibliothèque spécifiée.
    
    Args:
        data: Données tabulaires (DataFrame pandas, dict, liste de listes)
        config: Configuration du tableau
        library: Bibliothèque à utiliser
        **kwargs: Arguments spécifiques au type de tableau
        
    Returns:
        Objet tableau selon la bibliothèque utilisée
    """
    if library == "plotly":
        return _create_plotly_table(data, config, **kwargs)
    elif library == "matplotlib":
        return _create_mpl_table(data, config, **kwargs)
    elif library == "bokeh":
        return _create_bokeh_table(data, config, **kwargs)
    else:
        raise ValueError(f"Bibliothèque {library} non supportée pour les tableaux")

# Implémentations spécifiques
def _create_plotly_table(data, config, **kwargs):
    """Implémentation Plotly des tableaux de données"""
    import plotly.graph_objects as go
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 800)
    height = config.get('height', None)  # Hauteur automatique par défaut
    
    # Normalisation des données d'entrée en DataFrame pandas
    df = _normalize_table_data(data)
    
    # Configuration des colonnes
    headers = kwargs.get('headers', df.columns.tolist())
    
    # Configuration des couleurs
    header_color = kwargs.get('header_color', '#2a3f5f')
    header_font_color = kwargs.get('header_font_color', 'white')
    
    # Alternance de couleurs pour les lignes
    use_alternating_colors = kwargs.get('alternating_rows', True)
    if use_alternating_colors:
        odd_color = kwargs.get('odd_color', 'white')
        even_color = kwargs.get('even_color', 'lightgrey')
        fill_color = [odd_color, even_color] * (len(df) // 2 + 1)
        fill_color = fill_color[:len(df)]
    else:
        fill_color = kwargs.get('fill_color', 'white')
    
    # Création du tableau
    table = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=header_color,
            font=dict(color=header_font_color, size=kwargs.get('header_font_size', 14)),
            align='left',
            height=kwargs.get('header_height', 40)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[fill_color] * len(df.columns) if isinstance(fill_color, str) else fill_color,
            font=dict(
                color=kwargs.get('cell_font_color', 'darkslategray'),
                size=kwargs.get('cell_font_size', 12)
            ),
            align=kwargs.get('align', ['left'] * len(df.columns)),
            height=kwargs.get('cell_height', 30),
            format=kwargs.get('format', None)
        )
    )])
    
    # Configuration du layout
    layout_config = {
        'title': title,
        'width': width,
        'margin': dict(t=50, b=25, l=25, r=25),
    }
    
    # Ajouter la hauteur si spécifiée
    if height:
        layout_config['height'] = height
        
    table.update_layout(**layout_config)
            
    return table

def _create_mpl_table(data, config, **kwargs):
    """Implémentation Matplotlib des tableaux de données"""
    import matplotlib.pyplot as plt
    
    # Extraction des paramètres
    title = config.get('title', '')
    
    # Normalisation des données
    df = _normalize_table_data(data)
    
    # Configuration des colonnes
    headers = kwargs.get('headers', df.columns.tolist())
    
    # Création de la figure et de la table
    fig, ax = plt.subplots(figsize=(
        config.get('width', 800) / 100,
        config.get('height', len(df) * 30 + 50) / 100
    ))
    
    # Cacher les axes
    ax.axis('off')
    
    # Configuration des couleurs
    header_colors = kwargs.get('header_color', '#2a3f5f')
    header_font_color = kwargs.get('header_font_color', 'white')
    
    # Alternance de couleurs pour les lignes
    use_alternating_colors = kwargs.get('alternating_rows', True)
    if use_alternating_colors:
        odd_color = kwargs.get('odd_color', 'white')
        even_color = kwargs.get('even_color', 'lightgrey')
        row_colors = [odd_color, even_color] * (len(df) // 2 + 1)
        row_colors = row_colors[:len(df)]
    else:
        row_colors = None
    
    # Créer le tableau
    table = ax.table(
        cellText=df.values,
        colLabels=headers,
        loc='center',
        cellLoc=kwargs.get('align', 'left'),
        colColours=[header_colors] * len(headers),
        rowColours=row_colors
    )
    
    # Configuration du style du tableau
    table.auto_set_font_size(False)
    table.set_fontsize(kwargs.get('cell_font_size', 12))
    
    # Ajuster la largeur des colonnes
    table.auto_set_column_width(col=list(range(len(headers))))
    
    # Appliquer la couleur des textes d'en-tête
    for i, cell in table._cells.items():
        if i[0] == 0:  # En-tête
            cell.set_text_props(color=header_font_color)
    
    # Ajouter le titre
    plt.title(title, y=1.08)
    
    # Ajuster la figure
    plt.tight_layout()
    
    return fig

def _create_bokeh_table(data, config, **kwargs):
    """Implémentation Bokeh des tableaux de données"""
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter
    from bokeh.io import show, output_notebook
    
    # Normalisation des données
    df = _normalize_table_data(data)
    
    # Extraction des paramètres
    title = config.get('title', '')
    width = config.get('width', 800)
    height = config.get('height', None)
    
    # Création de la source de données
    source = ColumnDataSource(df)
    
    # Configuration des colonnes
    headers = kwargs.get('headers', df.columns.tolist())
    
    # Alternance de couleurs pour les lignes si activée
    use_alternating_colors = kwargs.get('alternating_rows', True)
    formatter = None
    if use_alternating_colors:
        odd_color = kwargs.get('odd_color', 'white')
        even_color = kwargs.get('even_color', 'lightgrey')
        formatter = HTMLTemplateFormatter(template=f"""
        <div style='background:<% if (index % 2) {{ return '{even_color}'; }} else {{ return '{odd_color}'; }} %>; color: black;'><%= value %></div>
        """)
    
    # Création des colonnes du tableau
    columns = []
    for header in headers:
        if header in df.columns:
            col_config = {
                'field': header,
                'title': header,
                'width': kwargs.get('column_width', 150)
            }
            
            # Ajouter le formateur si disponible
            if formatter:
                col_config['formatter'] = formatter
            
            columns.append(TableColumn(**col_config))
    
    # Création du tableau
    data_table = DataTable(
        source=source, 
        columns=columns, 
        width=width,
        height=height if height else len(df) * 40 + 30,
        index_position=None,
        header_row=True,
        editable=kwargs.get('editable', False),
        selectable=kwargs.get('selectable', True),
        sortable=kwargs.get('sortable', True),
        fit_columns=kwargs.get('fit_columns', False)
    )
    
    # Bokeh nécessite l'utilisation de l'objet DataTable directement
    # pour une utilisation dans un notebook ou une application Bokeh
    return data_table

def _normalize_table_data(data):
    """
    Normalise les données d'entrée sous forme de DataFrame pandas.
    
    Args:
        data: Données à normaliser (DataFrame, dict, liste de listes, etc.)
        
    Returns:
        pandas.DataFrame: Données normalisées
    """
    if isinstance(data, pd.DataFrame):
        return data
    
    elif isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], pd.DataFrame):
            return data['data']
        elif 'data' in data and isinstance(data['data'], dict):
            return pd.DataFrame(data['data'])
        elif all(isinstance(v, list) for v in data.values()):
            # Format {col1: [values], col2: [values], ...}
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    
    elif isinstance(data, list):
        if all(isinstance(row, dict) for row in data):
            # Liste de dictionnaires [{col1: val1, col2: val2}, ...]
            return pd.DataFrame(data)
        elif len(data) > 0 and isinstance(data[0], list):
            # Format [[header1, header2, ...], [row1val1, row1val2, ...], ...]
            headers = data[0]
            rows = data[1:]
            return pd.DataFrame(rows, columns=headers)
        else:
            # Format simple [val1, val2, ...]
            return pd.DataFrame(data)
    
    else:
        raise ValueError("Format de données non pris en charge pour les tableaux")
