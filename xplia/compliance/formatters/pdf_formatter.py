"""
Générateur de rapports PDF pour XPLIA
===================================

Ce module implémente un générateur de rapports au format PDF pour le système
avancé de génération de rapports de conformité XPLIA.
"""

import logging
import os
import datetime
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import io
import base64

# Import des modules de visualisation
try:
    from ...visualizations import ChartType, ChartGenerator
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Module de visualisations non disponible, les graphiques personnalisés ne seront pas pris en charge")
    ChartType = None
    ChartGenerator = None

from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ..report_base import BaseReportGenerator, ReportConfig, ReportContent

logger = logging.getLogger(__name__)

class XPLIAReport(FPDF):
    """
    Extension de FPDF adaptée aux besoins spécifiques de XPLIA.
    
    Cette classe ajoute des fonctionnalités spécifiques pour les rapports
    de conformité comme les en-têtes, pieds de page, et mise en forme.
    """
    
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.title = "Rapport de conformité XPLIA"
        self.organization = ""
        self.language = "fr"
        self.footer_text = ""
    
    def header(self):
        # Police Arial gras 15
        self.set_font('Arial', 'B', 15)
        # Titre
        w = self.get_string_width(self.title) + 6
        self.set_x((210 - w) / 2)
        # Couleurs du cadre, du fond et du texte
        self.set_draw_color(0, 80, 180)
        self.set_fill_color(230, 230, 230)
        self.set_text_color(0, 0, 0)
        # Épaisseur du cadre
        self.set_line_width(1)
        # Titre
        self.cell(w, 9, self.title, 1, 1, 'C', 1)
        # Saut de ligne
        self.ln(10)
    
    def footer(self):
        # Positionnement à 1.5 cm du bas
        self.set_y(-15)
        # Police Arial italique 8
        self.set_font('Arial', 'I', 8)
        # Couleur du texte en gris
        self.set_text_color(128)
        # Numéro de page et footer text
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - {self.footer_text}', 0, 0, 'C')

class PDFReportGenerator(BaseReportGenerator):
    """
    Générateur de rapports au format PDF.
    
    Cette classe génère des rapports de conformité au format PDF,
    avec support pour les styles personnalisés, les graphiques et les visualisations.
    """
    
    def __init__(self, config: ReportConfig):
        """
        Initialise le générateur de rapports PDF.
        
        Args:
            config: Configuration du générateur
        """
        super().__init__(config)
        self.template_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "templates", "pdf"
        )
    
    def generate(self, content: ReportContent, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Génère un rapport au format PDF.
        
        Args:
            content: Contenu du rapport
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Contenu PDF en bytes si output_path est None, sinon None
        """
        try:
            # Création du PDF
            pdf = self._create_pdf(content)
            
            # Génération du contenu
            self._generate_content(pdf, content)
            
            # Sauvegarde ou retour
            if output_path:
                pdf.output(output_path)
                logger.info(f"Rapport PDF généré avec succès: {output_path}")
                return None
            else:
                # Génération en mémoire
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_path = temp_file.name
                temp_file.close()
                
                pdf.output(temp_path)
                
                with open(temp_path, 'rb') as f:
                    pdf_content = f.read()
                
                # Nettoyage
                os.unlink(temp_path)
                
                return pdf_content
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport PDF: {e}")
            raise
    
    def _create_pdf(self, content: ReportContent) -> XPLIAReport:
        """
        Crée l'objet PDF avec les paramètres de base.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Objet PDF initialisé
        """
        pdf = XPLIAReport()
        pdf.title = content.title
        pdf.organization = self.config.organization
        pdf.language = self.config.language
        pdf.footer_text = f"XPLIA - {self._format_date(content.timestamp)}"
        
        # Configuration de base
        pdf.set_author(self.config.responsible)
        pdf.set_creator("XPLIA Report Generator")
        pdf.set_title(content.title)
        pdf.set_subject(f"Rapport de conformité pour {self.config.organization}")
        pdf.alias_nb_pages()  # Pour {nb} dans le pied de page
        
        # Ajout de la première page
        pdf.add_page()
        
        return pdf
    
    def _generate_content(self, pdf: XPLIAReport, content: ReportContent) -> None:
        """
        Génère le contenu du rapport PDF.
        
        Args:
            pdf: Objet PDF
            content: Contenu du rapport
        """
        # Ajout du logo si disponible
        if self.config.logo_path and os.path.exists(self.config.logo_path):
            pdf.image(self.config.logo_path, x=160, y=8, w=30)
        
        # Informations de base
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"{self._get_translation('organization')}: {self.config.organization}", ln=1)
        pdf.cell(0, 10, f"{self._get_translation('responsible')}: {self.config.responsible}", ln=1)
        pdf.cell(0, 10, f"{self._get_translation('date')}: {self._format_date(content.timestamp)}", ln=1)
        pdf.ln(5)
        
        # Résumé
        if content.summary:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('summary'), ln=1)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 8, content.summary)
            pdf.ln(5)
        
        # Score de conformité
        if content.compliance_score:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('compliance_score'), ln=1)
            
            # Score global
            score = content.compliance_score.get("score", "N/A")
            status = content.compliance_score.get("status", "")
            
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(30, 15, str(score), ln=0)
            pdf.set_font('Arial', '', 14)
            pdf.cell(0, 15, status, ln=1)
            
            # Détails des scores
            if 'score_details' in content and content['score_details']:
                self._add_chart(pdf, content['score_details'])
                
                # Tableau détaillé des scores
                pdf.ln(10)
                pdf.set_font('Arial', '', 10)
                
                col_width = 60
                row_height = 10
                
                # En-têtes
                pdf.set_fill_color(230, 230, 230)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(col_width, row_height, self._get_translation('regulation'), 1, 0, 'C', 1)
                pdf.cell(col_width, row_height, self._get_translation('score'), 1, 1, 'C', 1)
                
                # Données
                pdf.set_font('Arial', '', 10)
                for regulation, score in content['score_details'].items():
                    pdf.cell(col_width, row_height, regulation, 1, 0, 'L')
                    pdf.cell(col_width, row_height, f"{score}", 1, 1, 'C')
            
            # Traitement des visualisations personnalisées
            if 'visualizations' in content and content['visualizations']:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, self._get_translation('visualizations_title'), ln=1)
                pdf.ln(5)
                
                # Ajout des visualisations
                self._process_visualizations(pdf, content['visualizations'])
            
            pdf.ln(10)
        
        # Journal d'audit
        if content.audit_trail:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('audit_trail'), ln=1)
            
            # En-têtes
            if content.audit_trail:
                headers = content.audit_trail[0].keys()
                col_width = 190 / len(headers)
                
                pdf.set_font('Arial', 'B', 10)
                for header in headers:
                    pdf.cell(col_width, 10, str(header).capitalize(), 1, 0, 'C')
                pdf.ln()
                
                # Données
                pdf.set_font('Arial', '', 9)
                for entry in content.audit_trail:
                    for key, value in entry.items():
                        pdf.cell(col_width, 10, str(value), 1, 0)
                    pdf.ln()
            
            pdf.ln(10)
        
        # Journal des décisions
        if content.decision_log:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('decision_log'), ln=1)
            
            # En-têtes
            if content.decision_log:
                headers = content.decision_log[0].keys()
                col_width = 190 / len(headers)
                
                pdf.set_font('Arial', 'B', 10)
                for header in headers:
                    pdf.cell(col_width, 10, str(header).capitalize(), 1, 0, 'C')
                pdf.ln()
                
                # Données
                pdf.set_font('Arial', '', 9)
                for entry in content.decision_log:
                    for key, value in entry.items():
                        pdf.cell(col_width, 10, str(value), 1, 0)
                    pdf.ln()
            
            pdf.ln(10)
        
        # Problèmes identifiés
        if content.issues:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('issues'), ln=1)
            
            for issue in content.issues:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, issue.get("title", "Problème"), ln=1)
                
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(40, 8, f"{self._get_translation('severity')}:", 0, 0)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 8, issue.get("severity", ""), ln=1)
                
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(40, 8, f"{self._get_translation('description')}:", 0, 0)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 8, issue.get("description", ""))
                
                if "remediation" in issue:
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(40, 8, f"{self._get_translation('remediation')}:", 0, 0)
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 8, issue.get("remediation", ""))
                
                pdf.ln(5)
        
        # Recommandations
        if content.recommendations:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('recommendations'), ln=1)
            
            pdf.set_font('Arial', '', 11)
            for i, recommendation in enumerate(content.recommendations, 1):
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(10, 8, f"{i}.", 0, 0)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 8, recommendation)
                pdf.ln(2)
        
        # QR Code de vérification
        if self.config.include_verification_qr and self.config.verification_url:
            try:
                import qrcode
                from PIL import Image
                
                # Création du QR Code
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(self.config.verification_url)
                qr.make(fit=True)
                
                # Sauvegarde temporaire
                img = qr.make_image(fill_color="black", back_color="white")
                qr_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                img.save(qr_path)
                
                # Ajout au PDF
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, self._get_translation('verification'), ln=1)
                pdf.image(qr_path, x=75, y=60, w=60)
                pdf.set_y(130)
                pdf.set_font('Arial', '', 11)
                pdf.cell(0, 10, self._get_translation('verification_instruction'), 0, 1, 'C')
                
                # Nettoyage
                os.unlink(qr_path)
            except ImportError:
                logger.warning("Impossible de générer le QR Code: qrcode non installé")
        
        # Signature numérique si demandée
        if self.config.include_signatures:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self._get_translation('signature'), ln=1)
            
            signature = content.sign('xplia_secret')  # À remplacer par une vraie clé
            
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 8, f"{self._get_translation('signature_value')}: {signature}")
            pdf.cell(0, 8, f"{self._get_translation('signature_timestamp')}: {datetime.datetime.now().isoformat()}", ln=1)
    
    def _process_visualizations(self, pdf: XPLIAReport, visualizations: List[Dict[str, Any]]) -> None:
        """
        Traite les visualisations pour le rapport PDF.
        
        Cette méthode convertit les définitions abstraites de visualisations en images
        en utilisant le générateur de graphiques configuré, et les intègre au PDF.
        
        Args:
            pdf: Objet PDF
            visualizations: Liste de descriptions de visualisations à traiter
                Chaque élément doit contenir au minimum:
                - type: Type de graphique (correspond aux valeurs de ChartType)
                - data: Données pour le graphique
                - config: Configuration du graphique (optionnel)
                - title: Titre de la visualisation (optionnel)
                - description: Description de la visualisation (optionnel)
        """
        if not visualizations or not ChartGenerator:
            return
            
        # Initialisation du générateur de graphiques si nécessaire
        if not hasattr(self, 'chart_generator'):
            try:
                self.chart_generator = ChartGenerator()
                logger.info("Générateur de graphiques initialisé pour le PDF")
            except Exception as e:
                logger.error(f"Impossible d'initialiser le générateur de graphiques: {e}")
                return
        
        for viz in visualizations:
            try:
                # Traitement du titre et de la description
                title = viz.get('title', '')
                description = viz.get('description', '')
                
                # Ajout du titre et de la description
                if title:
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, title, ln=1)
                
                if description:
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 5, description)
                    pdf.ln(5)
                
                # Si du HTML est fourni directement, on ne peut pas le traiter dans un PDF
                # On ajoute simplement une note
                if "html" in viz:
                    pdf.set_font('Arial', 'I', 10)
                    pdf.multi_cell(0, 5, "[Visualisation HTML non disponible en PDF]")
                    pdf.ln(5)
                    continue
                
                # Sinon, on génère la visualisation
                chart_type = getattr(ChartType, viz.get("type", "BAR").upper())
                data = viz.get("data", {})
                config = viz.get("config", {})
                
                # Ajout du titre et de la description dans la configuration si fournis
                if "title" in viz and "title" not in config:
                    config["title"] = viz["title"]
                if "description" in viz and "subtitle" not in config:
                    config["subtitle"] = viz["description"]
                
                # Configuration spécifique pour PDF (désactiver l'interactivité, etc.)
                config["static"] = True
                
                # Création du graphique
                chart = self.chart_generator.create_chart(chart_type, data, config)
                
                # Conversion en image pour intégration dans le PDF
                img_path, width, height = self._chart_to_image(chart)
                if img_path:
                    # Calcul de la largeur adaptée à la page PDF
                    pdf_width = min(170, width / 2)  # Largeur max de 170 mm
                    
                    # Ajout de l'image au PDF
                    pdf.image(img_path, x=20, w=pdf_width)
                    pdf.ln(5)
                    
                    # Nettoyage
                    os.unlink(img_path)
                
                logger.info(f"Visualisation PDF générée: {title}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la génération d'une visualisation PDF: {e}")
                pdf.set_font('Arial', 'I', 10)
                pdf.multi_cell(0, 5, f"[Erreur de visualisation: {e}]")
                pdf.ln(5)
    
    def _chart_to_image(self, chart) -> Tuple[Optional[str], int, int]:
        """
        Convertit un objet de graphique en image pour intégration dans un PDF.
        
        Args:
            chart: Objet graphique généré par ChartGenerator
            
        Returns:
            Tuple (chemin de l'image temporaire, largeur, hauteur)
        """
        try:
            # Extraction du type de graphique et de la figure
            fig = None
            if hasattr(chart, 'get_figure'):
                # Pour matplotlib
                fig = chart.get_figure()
            elif hasattr(chart, 'to_image'):
                # Pour plotly
                img_bytes = chart.to_image(format='png', width=800, height=500)
                img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                with open(img_path, 'wb') as f:
                    f.write(img_bytes)
                return img_path, 800, 500
            elif hasattr(chart, '_repr_png_'):
                # Pour d'autres bibliothèques avec représentation PNG
                img_data = chart._repr_png_()
                img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                return img_path, 800, 500
            else:
                logger.warning(f"Type de graphique non pris en charge pour l'export PDF")
                return None, 0, 0
                
            # Traitement pour matplotlib
            if fig is not None:
                img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                fig.savefig(img_path, dpi=100, bbox_inches='tight')
                width, height = fig.get_size_inches() * fig.dpi
                plt.close(fig)
                return img_path, int(width), int(height)
                
            return None, 0, 0
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion du graphique en image: {e}")
            return None, 0, 0
            
    def _add_chart(self, pdf: XPLIAReport, score_details: Dict[str, Any]) -> None:
        """
        Ajoute un graphique des scores de conformité au PDF.
        
        Args:
            pdf: Objet PDF
            score_details: Détails des scores par réglementation
        """
        try:
            # Création du graphique
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Données
            regulations = list(score_details.keys())
            scores = list(score_details.values())
            
            # Graphique en barres
            bars = ax.bar(regulations, scores, color='skyblue')
            ax.set_ylim(0, 100)
            ax.set_xlabel(self._get_translation('regulation'))
            ax.set_ylabel(self._get_translation('score'))
            ax.set_title(self._get_translation('compliance_score_chart'))
            
            # Rotation des labels
            plt.xticks(rotation=45, ha='right')
            
            # Ajout des valeurs au-dessus des barres
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Sauvegarde temporaire
            chart_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
            plt.savefig(chart_path, dpi=100)
            plt.close()
            
            # Ajout au PDF
            pdf.image(chart_path, x=20, w=170)
            
            # Nettoyage
            os.unlink(chart_path)
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique: {e}")
    
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
            
    def _get_translation(self, key: str) -> str:
        """
        Obtient une traduction pour la clé donnée.
        
        Args:
            key: Clé de traduction
            
        Returns:
            Texte traduit
        """
        translations = {
            'fr': {
                'verification': 'Vérification numérique',
                'verification_instruction': 'Scanner ce QR code pour vérifier ce rapport',
                'signature': 'Signature numérique',
                'signature_value': 'Signature',
                'signature_timestamp': 'Horodatage',
                'regulation': 'Réglementation',
                'score': 'Score',
                'compliance_score_chart': 'Scores de conformité',
                'date_created': 'Date de création',
                'authors': 'Auteurs',
                'issues': 'Problèmes identifiés',
                'recommendations': 'Recommandations',
                'audit_trail': 'Journal d\'audit',
                'severity': 'Sévérité',
                'description': 'Description',
                'remediation': 'Correction',
                'visualizations_title': 'Visualisations et Analyses'
            },
            'en': {
                'verification': 'Digital Verification',
                'verification_instruction': 'Scan this QR code to verify this report',
                'signature': 'Digital Signature',
                'signature_value': 'Signature',
                'signature_timestamp': 'Timestamp',
                'regulation': 'Regulation',
                'score': 'Score',
                'compliance_score_chart': 'Compliance Scores',
                'date_created': 'Creation Date',
                'authors': 'Authors',
                'issues': 'Identified Issues',
                'recommendations': 'Recommendations',
                'audit_trail': 'Audit Trail',
                'severity': 'Severity',
                'description': 'Description',
                'remediation': 'Remediation',
                'visualizations_title': 'Visualizations and Analysis'
            }
        }
        
        # Récupération de la langue du rapport
        lang = getattr(self, 'language', 'fr')
        if lang not in translations:
            lang = 'fr'  # Langue par défaut
            
        # Récupération de la traduction
        return translations[lang].get(key, key)
