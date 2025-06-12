"""
Générateur de rapports HTML pour XPLIA
====================================

Ce module implémente un générateur de rapports au format HTML pour le système
avancé de génération de rapports de conformité XPLIA.
"""

import logging
import os
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import base64
from dataclasses import asdict

from ..report_base import BaseReportGenerator, ReportConfig, ReportContent
from ...visualizations import ChartGenerator, ChartType, ChartLibrary, OutputContext

logger = logging.getLogger(__name__)

class HTMLReportGenerator(BaseReportGenerator):
    """
    Générateur de rapports au format HTML.
    
    Cette classe génère des rapports de conformité au format HTML interactif,
    avec support pour les styles personnalisés, les graphiques et les visualisations.
    """
    
    def __init__(self, config: ReportConfig):
        """
        Initialise le générateur de rapports HTML.
        
        Args:
            config: Configuration du générateur
        """
        super().__init__(config)
        self.template_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "templates", "html"
        )
        
        # Initialisation du générateur de visualisations
        self.chart_generator = ChartGenerator(
            library=ChartLibrary.PLOTLY,
            theme=getattr(config, "chart_theme", "light"),
            output_context=OutputContext.WEB,
            interactive=True,
            responsive=True
        )
    
    def generate(self, content: ReportContent, output_path: Optional[str] = None) -> Optional[str]:
        """
        Génère un rapport au format HTML.
        
        Args:
            content: Contenu du rapport
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Contenu HTML si output_path est None, sinon None
        """
        try:
            # Génération du HTML
            html_content = self._generate_html(content)
            
            # Sauvegarde ou retour
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"Rapport HTML généré avec succès: {output_path}")
                return None
            else:
                return html_content
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport HTML: {e}")
            raise
    
    def _generate_html(self, content: ReportContent) -> str:
        """
        Génère le contenu HTML du rapport.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Contenu HTML du rapport
        """
        # Chargement du template
        template_path = self._get_template_path()
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            # Si le template n'existe pas, utilisation du template par défaut
            template = self._get_default_template()
        
        # Préparation des données
        report_data = self._prepare_template_data(content)
        
        # Remplacement des variables dans le template
        html = template
        for key, value in report_data.items():
            placeholder = f"{{{{ {key} }}}}"
            if isinstance(value, str):
                html = html.replace(placeholder, value)
        
        # Traitement des sections conditionnelles et des listes
        html = self._process_sections(html, report_data)
        
        return html
    
    def _get_template_path(self) -> str:
        """
        Détermine le chemin du template à utiliser.
        
        Returns:
            Chemin du template
        """
        # Utilisation du template spécifié dans la configuration ou du template par défaut
        template_name = self.config.template_name or "standard"
        
        # Recherche du template dans le répertoire des templates
        template_path = os.path.join(self.template_dir, f"{template_name}.html")
        
        # Si le template spécifié n'existe pas, utilisation du template standard
        if not os.path.exists(template_path):
            template_path = os.path.join(self.template_dir, "standard.html")
        
        return template_path
    
    def _get_default_template(self) -> str:
        """
        Renvoie le template HTML par défaut.
        
        Returns:
            Template HTML par défaut
        """
        return """<!DOCTYPE html>
<html lang="{{ language }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .logo { max-width: 200px; }
        h1, h2, h3 { color: #0056b3; }
        h1 { font-size: 24px; }
        h2 { font-size: 20px; margin-top: 30px; }
        h3 { font-size: 18px; }
        .metadata {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .score {
            display: flex;
            align-items: center;
            margin: 20px 0;
        }
        .score-value {
            font-size: 48px;
            font-weight: bold;
            margin-right: 20px;
        }
        .score-label {
            font-size: 18px;
        }
        .chart-container {
            height: 300px;
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
        .issue {
            margin-bottom: 15px;
            padding: 15px;
            border-left: 4px solid #dc3545;
            background-color: #fff9f9;
        }
        .recommendation {
            margin-bottom: 15px;
            padding: 15px;
            border-left: 4px solid #28a745;
            background-color: #f9fff9;
        }
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #777;
            font-size: 14px;
        }
        .qr-code {
            text-align: center;
            margin: 20px 0;
        }
        .signature {
            margin: 30px 0;
            padding: 15px;
            border: 1px solid #ddd;
            background: #f9f9f9;
        }
        @media print {
            body { padding: 0; }
            .container { box-shadow: none; }
        }
        .content {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .summary {
            margin-bottom: 20px;
        }
        .card {
            background-color: #fff;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .visualizations {
            margin-bottom: 20px;
        }
        .viz-container {
            margin-bottom: 20px;
        }
        .viz-description {
            font-size: 14px;
            color: #666;
        }
        .viz-content {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="content">
        <!-- Section Synthèse -->
        <section class="summary">
            <h2>Synthèse</h2>
            <div class="card">
                {{ summary }}
            </div>
        </section>
        
        {% if has_visualizations %}
        <!-- Section Visualisations -->
        <section class="visualizations">
            <h2>Visualisations et Analyses</h2>
            {% for viz in visualizations %}
            <div class="viz-container card">
                {% if viz.title %}
                <h3>{{ viz.title }}</h3>
                {% endif %}
                {% if viz.description %}
                <p class="viz-description">{{ viz.description }}</p>
                {% endif %}
                <div class="viz-content">
                    {{ viz.html }}
                </div>
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        <header>
            <div>
                <h1>{{ title }}</h1>
                <p>{{ timestamp }}</p>
            </div>
            {% if logo_path %}
            <img src="{{ logo_path }}" alt="Logo" class="logo">
            {% endif %}
        </header>
        
        <section class="metadata">
            <p><strong>{{ organization_label }}:</strong> {{ organization }}</p>
            <p><strong>{{ responsible_label }}:</strong> {{ responsible }}</p>
            <p><strong>{{ date_label }}:</strong> {{ formatted_date }}</p>
        </section>
        
        {% if summary %}
        <section>
            <h2>{{ summary_label }}</h2>
            <p>{{ summary }}</p>
        </section>
        {% endif %}
        
        {% if compliance_score %}
        <section>
            <h2>{{ compliance_score_label }}</h2>
            <div class="score">
                <div class="score-value">{{ score }}</div>
                <div class="score-label">{{ compliance_status }}</div>
            </div>
            
            {% if score_details %}
            <div class="chart-container" id="scoreChart">
                <!-- Chart will be inserted here by JavaScript -->
            </div>
            <table>
                <tr>
                    <th>{{ regulation_label }}</th>
                    <th>{{ score_label }}</th>
                </tr>
                {% for regulation, score in score_details.items() %}
                <tr>
                    <td>{{ regulation }}</td>
                    <td>{{ score }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </section>
        {% endif %}
        
        {% if audit_trail %}
        <section>
            <h2>{{ audit_trail_label }}</h2>
            <table>
                <tr>
                    {% for header in audit_trail_headers %}
                    <th>{{ header }}</th>
                    {% endfor %}
                </tr>
                {% for entry in audit_trail %}
                <tr>
                    {% for field, value in entry.items() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        </section>
        {% endif %}
        
        {% if decision_log %}
        <section>
            <h2>{{ decision_log_label }}</h2>
            <table>
                <tr>
                    {% for header in decision_log_headers %}
                    <th>{{ header }}</th>
                    {% endfor %}
                </tr>
                {% for entry in decision_log %}
                <tr>
                    {% for field, value in entry.items() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        </section>
        {% endif %}
        
        {% if issues %}
        <section>
            <h2>{{ issues_label }}</h2>
            {% for issue in issues %}
            <div class="issue">
                <h3>{{ issue.title }}</h3>
                <p><strong>{{ severity_label }}:</strong> {{ issue.severity }}</p>
                <p><strong>{{ description_label }}:</strong> {{ issue.description }}</p>
                {% if issue.remediation %}
                <p><strong>{{ remediation_label }}:</strong> {{ issue.remediation }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        {% if recommendations %}
        <section>
            <h2>{{ recommendations_label }}</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation">
                <p>{{ recommendation }}</p>
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        {% if include_verification_qr and verification_url %}
        <section class="qr-code">
            <h2>{{ verification_label }}</h2>
            <img src="{{ qr_code }}" alt="QR Code for verification">
            <p>{{ verification_instruction }}</p>
        </section>
        {% endif %}
        
        {% if include_signatures %}
        <section class="signature">
            <h2>{{ signature_label }}</h2>
            <p>{{ signature_value }}</p>
            <p>{{ signature_timestamp }}</p>
        </section>
        {% endif %}
        
        <footer>
            <p>{{ footer_text }}</p>
            <p>{{ generator_info }}</p>
        </footer>
    </div>
    
    {% if include_charts %}
    <script>
        // Script qui sera injecté pour les graphiques
        // Exemple avec Chart.js (à inclure comme dépendance)
        document.addEventListener('DOMContentLoaded', function() {
            if (document.getElementById('scoreChart')) {
                // Code pour générer les graphiques
            }
        });
    </script>
    {% endif %}
</body>
</html>"""
    
    def _process_visualizations(self, visualizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Traite les visualisations pour le rapport HTML.
        
        Cette méthode convertit les définitions abstraites de visualisations en HTML intégrable
        en utilisant le générateur de graphiques configuré.
        
        Args:
            visualizations: Liste de descriptions de visualisations à traiter
                Chaque élément doit contenir au minimum:
                - type: Type de graphique (correspond aux valeurs de ChartType)
                - data: Données pour le graphique
                - config: Configuration du graphique (optionnel)
                - title: Titre de la visualisation (optionnel)
                - description: Description de la visualisation (optionnel)
                
        Returns:
            Liste de visualisations traitées avec le HTML généré
        """
        processed_visualizations = []
        
        if not visualizations:
            return processed_visualizations
            
        for viz in visualizations:
            try:
                # Si le HTML est déjà fourni, on l'utilise directement
                if "html" in viz:
                    processed_visualizations.append(viz)
                    continue
                
                # Sinon, on génère le graphique
                chart_type = getattr(ChartType, viz.get("type", "BAR").upper())
                data = viz.get("data", {})
                config = viz.get("config", {})
                
                # Ajout du titre et de la description dans la configuration si fournis
                if "title" in viz and "title" not in config:
                    config["title"] = viz["title"]
                if "description" in viz and "subtitle" not in config:
                    config["subtitle"] = viz["description"]
                
                # Création du graphique
                chart = self.chart_generator.create_chart(chart_type, data, config)
                
                # Conversion en HTML
                html_content = self.chart_generator.to_html(chart)
                
                # Ajout à la liste des visualisations traitées
                processed_viz = viz.copy()
                processed_viz["html"] = html_content
                processed_visualizations.append(processed_viz)
                
                logger.info(f"Visualisation générée: {viz.get('title', 'Sans titre')}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la génération d'une visualisation: {e}")
                # On ajoute une visualisation d'erreur
                processed_viz = viz.copy()
                processed_viz["html"] = f"<div class='error-visualization'><p>Erreur: {e}</p></div>"
                processed_visualizations.append(processed_viz)
        
        return processed_visualizations
    
    def _prepare_template_data(self, content: ReportContent) -> Dict[str, Any]:
        """
        Prépare les données pour le template HTML.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Dictionnaire avec les données pour le template
        """
        # Données de base
        data = {
            "title": content.title,
            "timestamp": content.timestamp,
            "language": self.config.language,
            "organization": self.config.organization,
            "responsible": self.config.responsible,
            "formatted_date": self._format_date(content.timestamp),
            "summary": content.summary or "",
            "footer_text": f"© {datetime.datetime.now().year} XPLIA - Rapport généré le {self._format_date(content.timestamp)}",
            "generator_info": f"Généré par XPLIA Report Generator v1.0.0"
        }
        
        # Traductions
        translations = {
            "organization_label": self._get_translation("organization"),
            "responsible_label": self._get_translation("responsible"),
            "date_label": self._get_translation("date"),
            "summary_label": self._get_translation("summary"),
            "compliance_score_label": self._get_translation("compliance_score"),
            "regulation_label": self._get_translation("regulation"),
            "score_label": self._get_translation("score"),
            "audit_trail_label": self._get_translation("audit_trail"),
            "decision_log_label": self._get_translation("decision_log"),
            "issues_label": self._get_translation("issues"),
            "severity_label": self._get_translation("severity"),
            "description_label": self._get_translation("description"),
            "remediation_label": self._get_translation("remediation"),
            "recommendations_label": self._get_translation("recommendations"),
            "verification_label": self._get_translation("verification"),
            "verification_instruction": self._get_translation("verification_instruction"),
            "signature_label": self._get_translation("signature"),
        }
        data.update(translations)
        
        # Logo
        if self.config.logo_path and os.path.exists(self.config.logo_path):
            data["logo_path"] = self.config.logo_path
        
        # Score de conformité
        if content.compliance_score:
            data["compliance_score"] = True
            data["score"] = content.compliance_score.get("score", "N/A")
            data["compliance_status"] = content.compliance_score.get("status", "")
            
            # Détails du score
            if "details" in content.compliance_score:
                data["score_details"] = content.compliance_score["details"]
        
        # Journal d'audit
        if content.audit_trail:
            data["audit_trail"] = content.audit_trail
            data["audit_trail_headers"] = list(content.audit_trail[0].keys()) if content.audit_trail else []
        
        # Journal des décisions
        if content.decision_log:
            data["decision_log"] = content.decision_log
            data["decision_log_headers"] = list(content.decision_log[0].keys()) if content.decision_log else []
        
        # Problèmes identifiés
        if content.issues:
            data["issues"] = content.issues
        
        # Recommandations
        if content.recommendations:
            data["recommendations"] = content.recommendations
            
        # Visualisations
        if hasattr(content, "visualizations") and content.visualizations:
            # Traitement des visualisations
            processed_visualizations = self._process_visualizations(content.visualizations)
            data["visualizations"] = processed_visualizations
            data["has_visualizations"] = len(processed_visualizations) > 0
        
        # QR Code de vérification
        if self.config.include_verification_qr and self.config.verification_url:
            try:
                import qrcode
                from io import BytesIO
                import base64
                
                # Création du QR Code avec l'URL de vérification
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(self.config.verification_url)
                qr.make(fit=True)
                
                # Conversion en image et puis en base64
                img = qr.make_image(fill_color="black", back_color="white")
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data["qr_code"] = f"data:image/png;base64,{img_str}"
            except ImportError:
                logger.warning("Impossible de générer le QR Code: qrcode non installé")
        
        # Signature numérique si demandée
        if self.config.include_signatures:
            data["signature_value"] = content.sign('xplia_secret')  # À remplacer par une vraie clé
            data["signature_timestamp"] = datetime.datetime.now().isoformat()
        
        # Ajout des métadonnées
        data.update(content.metadata)
        
        return data
    
    def _process_sections(self, html: str, data: Dict[str, Any]) -> str:
        """
        Traite les sections conditionnelles et les listes dans le template.
        
        Args:
            html: Template HTML
            data: Données pour le template
            
        Returns:
            HTML avec les sections conditionnelles et les listes traitées
        """
        # Sections conditionnelles
        for key, value in data.items():
            # Conditions {% if key %}...{% endif %}
            if_marker = f"{{% if {key} %}}"
            endif_marker = "{% endif %}"
            
            if if_marker in html and endif_marker in html:
                # S'il y a une valeur, garder le contenu entre les marqueurs
                if value:
                    html = html.replace(if_marker, "")
                    html = html.replace(endif_marker, "")
                # Sinon, supprimer le contenu entre les marqueurs
                else:
                    start_idx = html.find(if_marker)
                    end_idx = html.find(endif_marker, start_idx) + len(endif_marker)
                    if start_idx != -1 and end_idx != -1:
                        html = html[:start_idx] + html[end_idx:]
        
        # TODO: Traitement des listes {% for item in items %}...{% endfor %}
        # Cette partie nécessite un parser plus élaboré qui pourrait être implémenté ultérieurement
        
        return html
    
    def _format_date(self, timestamp: str) -> str:
        """
        Formate une date pour l'affichage.
        
        Args:
            timestamp: Timestamp ISO
            
        Returns:
            Date formatée
        """
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
            return dt.strftime("%d/%m/%Y %H:%M")
        except (ValueError, TypeError):
            return timestamp
