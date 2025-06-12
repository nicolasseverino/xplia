"""
Démonstrateur de visualisations en HTML pur
=========================================

Ce script génère un rapport HTML avec des visualisations
en utilisant uniquement HTML, JavaScript (Chart.js via CDN) et Python standard.
"""

import os
import json
import numpy as np
from datetime import datetime

def generate_html_report():
    """Génère un rapport HTML avec des visualisations en Chart.js."""
    
    # Création de données d'exemple
    np.random.seed(42)
    
    # Données pour bar chart
    categories = ['A', 'B', 'C', 'D', 'E']
    values_bar = np.random.randint(10, 100, size=len(categories)).tolist()
    
    # Données pour line chart
    dates = [f"2024-{month:02d}-01" for month in range(1, 13)]
    values_line = np.cumsum(np.random.normal(10, 3, 12)).tolist()
    
    # Données pour pie chart
    pie_labels = categories
    pie_values = np.random.randint(10, 100, size=len(categories)).tolist()
    
    # JavaScript pour la transformation des couleurs - ES5 compatible
    js_color_transform = """
        function transformColors(colors) {
            var result = [];
            for (var i = 0; i < colors.length; i++) {
                result.push(colors[i].replace('0.7', '1.0'));
            }
            return result;
        }
    """
    
    # Création du HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport de démonstration avec Chart.js</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .report-header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 20px;
            }}
            .visualization-container {{
                margin-bottom: 40px;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #fff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .visualization-title {{
                font-size: 1.4em;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            .visualization-description {{
                color: #7f8c8d;
                margin-bottom: 20px;
            }}
            .chart-container {{
                position: relative; 
                height: 400px;
                width: 100%;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="report-header">
            <h1>Rapport de démonstration avec visualisations Chart.js</h1>
            <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="report-content">
            <div class="visualization-container">
                <h2 class="visualization-title">Répartition par catégorie</h2>
                <p class="visualization-description">Ce graphique montre la distribution des valeurs par catégorie.</p>
                <div class="chart-container">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
            
            <div class="visualization-container">
                <h2 class="visualization-title">Évolution temporelle</h2>
                <p class="visualization-description">Ce graphique montre l'évolution des valeurs au cours du temps.</p>
                <div class="chart-container">
                    <canvas id="lineChart"></canvas>
                </div>
            </div>
            
            <div class="visualization-container">
                <h2 class="visualization-title">Distribution en camembert</h2>
                <p class="visualization-description">Ce graphique montre la répartition proportionnelle des catégories.</p>
                <div class="chart-container">
                    <canvas id="pieChart"></canvas>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Rapport généré par le démonstrateur en HTML pur XPLIA</p>
        </footer>
        
        <script>
            // Configuration des couleurs
            var colors = [
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 99, 132, 0.7)',
                'rgba(255, 206, 86, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(153, 102, 255, 0.7)',
            ];
            
            // Fonction pour transformer les couleurs (ES5 compatible)
            {js_color_transform}
            
            // Bar Chart
            var barCtx = document.getElementById('barChart').getContext('2d');
            new Chart(barCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(categories)},
                    datasets: [{{
                        label: 'Valeurs',
                        data: {json.dumps(values_bar)},
                        backgroundColor: colors,
                        borderColor: transformColors(colors),
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Valeurs'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Catégories'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Line Chart
            var lineCtx = document.getElementById('lineChart').getContext('2d');
            new Chart(lineCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(dates)},
                    datasets: [{{
                        label: 'Évolution',
                        data: {json.dumps(values_line)},
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.2,
                        fill: true,
                        pointRadius: 5,
                        pointBackgroundColor: 'rgba(75, 192, 192, 1)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            title: {{
                                display: true,
                                text: 'Valeur'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Date'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Pie Chart
            var pieCtx = document.getElementById('pieChart').getContext('2d');
            new Chart(pieCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {json.dumps(pie_labels)},
                    datasets: [{{
                        data: {json.dumps(pie_values)},
                        backgroundColor: colors,
                        borderColor: transformColors(colors),
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'right'
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html

def main():
    """Fonction principale."""
    print("Génération d'un rapport HTML avec des visualisations Chart.js...")
    
    # Création du dossier de sortie s'il n'existe pas
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Génération du HTML
    html_content = generate_html_report()
    
    # Sauvegarde du rapport
    output_path = os.path.join(output_dir, "pure_html_visualization_demo.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Rapport généré avec succès: {output_path}")
    print("Ce rapport contient 3 visualisations basées sur Chart.js: un graphique à barres, un graphique linéaire et un camembert.")
    print("Ouvrez le fichier dans un navigateur pour visualiser le rapport.")

if __name__ == "__main__":
    main()
