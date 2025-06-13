"""
Extension du formatter PDF avec support des métriques de confiance
==================================================================

Ce module étend le générateur de rapports PDF standard pour y intégrer
les métriques de confiance issues des modules d'évaluation de confiance XPLIA.
"""

import logging
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .pdf_formatter import PDFReportGenerator, XPLIAReport
from .trust_formatter_mixin import TrustFormatterMixin
from ..report_base import ReportContent

logger = logging.getLogger(__name__)

class TrustPDFReportGenerator(PDFReportGenerator, TrustFormatterMixin):
    """
    Générateur de rapports PDF avec support des métriques de confiance.
    
    Cette classe étend le générateur de rapports PDF standard pour y intégrer
    les métriques de confiance issues des modules d'évaluation de confiance XPLIA.
    """
    
    def _generate_content(self, pdf: XPLIAReport, content: ReportContent) -> None:
        """
        Génère le contenu du rapport PDF avec métriques de confiance.
        
        Args:
            pdf: Objet PDF
            content: Contenu du rapport
        """
        # Génération du contenu de base
        super()._generate_content(pdf, content)
        
        # Vérification de la présence d'explications avec métriques de confiance
        if not hasattr(content, "explanations") or not content.explanations:
            return
            
        # Traitement des métriques de confiance pour chaque explication
        for i, explanation in enumerate(content.explanations):
            # Traitement des métriques de confiance
            trust_data = self._process_trust_metrics(explanation)
            
            # Si des métriques sont disponibles, les intégrer au rapport
            if trust_data["has_trust_metrics"]:
                # Ajout d'une nouvelle page pour les métriques de confiance
                pdf.add_page()
                
                # Titre de la section
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, f"Évaluation de Confiance - Explication {i+1}", 0, 1, 'C')
                pdf.ln(5)
                
                # Ajout du score global et du niveau de confiance
                self._add_trust_score_section(pdf, trust_data)
                
                # Ajout des métriques détaillées
                self._add_trust_metrics_details(pdf, trust_data)
                
                # Ajout des recommandations si disponibles
                if trust_data["trust_recommendations"]:
                    self._add_trust_recommendations(pdf, trust_data["trust_recommendations"])
                    
                # Ajout du graphique radar des scores de confiance
                self._add_trust_radar_chart(pdf, trust_data)
                
                logger.info(f"Métriques de confiance intégrées pour l'explication {i+1}")
    
    def _add_trust_score_section(self, pdf: XPLIAReport, trust_data: Dict[str, Any]) -> None:
        """
        Ajoute la section de score global de confiance au PDF.
        
        Args:
            pdf: Objet PDF
            trust_data: Données de confiance formatées
        """
        # Score global
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Score Global de Confiance", 0, 1)
        
        # Niveau de confiance
        pdf.set_font('Arial', '', 12)
        
        # Couleurs selon le niveau de confiance
        level_colors = {
            "very-low": (211, 47, 47),  # Rouge
            "low": (245, 124, 0),       # Orange
            "moderate": (251, 192, 45),  # Jaune
            "high": (124, 179, 66),     # Vert clair
            "very-high": (56, 142, 60)  # Vert foncé
        }
        
        level_class = trust_data["trust_level_class"]
        color = level_colors.get(level_class, (128, 128, 128))  # Gris par défaut
        
        # Affichage du score et du niveau
        pdf.set_text_color(*color)
        pdf.cell(0, 10, f"Score: {trust_data['trust_score']}/10 - Niveau: {trust_data['trust_level_label']}", 0, 1)
        pdf.set_text_color(0, 0, 0)  # Retour à la couleur noire
        
        # Résumé
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 11)
        pdf.multi_cell(0, 5, trust_data["trust_summary"])
        pdf.ln(5)
    
    def _add_trust_metrics_details(self, pdf: XPLIAReport, trust_data: Dict[str, Any]) -> None:
        """
        Ajoute les métriques détaillées de confiance au PDF.
        
        Args:
            pdf: Objet PDF
            trust_data: Données de confiance formatées
        """
        # Titre de la section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Métriques Détaillées", 0, 1)
        
        # Définition des métriques à afficher
        metrics = [
            {"name": "Certitude", "score": trust_data["uncertainty_score"], 
             "class": trust_data["uncertainty_class"], "percentage": trust_data["uncertainty_percentage"]},
            {"name": "Anti-Fairwashing", "score": trust_data["fairwashing_score"], 
             "class": trust_data["fairwashing_class"], "percentage": trust_data["fairwashing_percentage"]},
            {"name": "Cohérence", "score": trust_data["consistency_score"], 
             "class": trust_data["consistency_class"], "percentage": trust_data["consistency_percentage"]},
            {"name": "Robustesse", "score": trust_data["robustness_score"], 
             "class": trust_data["robustness_class"], "percentage": trust_data["robustness_percentage"]}
        ]
        
        # Couleurs selon la classe
        class_colors = {
            "excellent": (76, 175, 80),  # Vert
            "good": (139, 195, 74),      # Vert clair
            "moderate": (255, 193, 7),   # Jaune
            "poor": (255, 152, 0),       # Orange
            "critical": (244, 67, 54)    # Rouge
        }
        
        # Affichage des métriques
        for metric in metrics:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(40, 10, metric["name"], 0, 0)
            
            # Score avec couleur
            color = class_colors.get(metric["class"], (128, 128, 128))
            pdf.set_text_color(*color)
            pdf.cell(30, 10, metric["score"], 0, 0)
            pdf.set_text_color(0, 0, 0)
            
            # Barre de progression
            self._draw_progress_bar(pdf, 70, pdf.get_y() + 5, 100, 5, metric["percentage"])
            
            pdf.ln(15)
        
        # Affichage des types d'incertitude si disponibles
        if trust_data["uncertainty_types"]:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Types d'Incertitude:", 0, 1)
            
            pdf.set_font('Arial', '', 11)
            for type_name, value in trust_data["uncertainty_types"]:
                pdf.cell(0, 6, f"- {type_name}: {value}", 0, 1)
            
            pdf.ln(5)
        
        # Affichage des types de fairwashing détectés si disponibles
        if trust_data["detected_fairwashing_types"]:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Types de Fairwashing Détectés:", 0, 1)
            
            pdf.set_font('Arial', '', 11)
            for type_name in trust_data["detected_fairwashing_types"]:
                pdf.cell(0, 6, f"- {type_name}", 0, 1)
            
            pdf.ln(5)
    
    def _draw_progress_bar(self, pdf: XPLIAReport, x: float, y: float, 
                          w: float, h: float, percentage: float) -> None:
        """
        Dessine une barre de progression dans le PDF.
        
        Args:
            pdf: Objet PDF
            x: Position X
            y: Position Y
            w: Largeur
            h: Hauteur
            percentage: Pourcentage de remplissage (0-100)
        """
        # Sauvegarde de la position et des couleurs
        fill_color = pdf.fill_color
        text_color = pdf.text_color
        draw_color = pdf.draw_color
        
        # Fond de la barre
        pdf.set_fill_color(240, 240, 240)
        pdf.set_draw_color(200, 200, 200)
        pdf.rect(x, y, w, h, 'FD')
        
        # Partie remplie de la barre
        if percentage > 0:
            # Couleur selon le pourcentage
            if percentage >= 80:
                pdf.set_fill_color(76, 175, 80)  # Vert
            elif percentage >= 60:
                pdf.set_fill_color(139, 195, 74)  # Vert clair
            elif percentage >= 40:
                pdf.set_fill_color(255, 193, 7)  # Jaune
            elif percentage >= 20:
                pdf.set_fill_color(255, 152, 0)  # Orange
            else:
                pdf.set_fill_color(244, 67, 54)  # Rouge
                
            fill_width = w * percentage / 100
            pdf.rect(x, y, fill_width, h, 'F')
        
        # Restauration des couleurs
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*text_color)
        pdf.set_draw_color(*draw_color)
    
    def _add_trust_recommendations(self, pdf: XPLIAReport, recommendations: List[str]) -> None:
        """
        Ajoute les recommandations de confiance au PDF.
        
        Args:
            pdf: Objet PDF
            recommendations: Liste des recommandations
        """
        # Titre de la section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Recommandations", 0, 1)
        
        # Liste des recommandations
        pdf.set_font('Arial', '', 11)
        for i, recommendation in enumerate(recommendations, 1):
            pdf.cell(10, 6, f"{i}.", 0, 0)
            pdf.multi_cell(0, 6, recommendation)
        
        pdf.ln(5)
    
    def _add_trust_radar_chart(self, pdf: XPLIAReport, trust_data: Dict[str, Any]) -> None:
        """
        Ajoute un graphique radar des scores de confiance au PDF.
        
        Args:
            pdf: Objet PDF
            trust_data: Données de confiance formatées
        """
        try:
            # Création du graphique radar
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
            
            # Catégories et valeurs
            categories = ['Certitude', 'Anti-Fairwashing', 'Cohérence', 'Robustesse']
            values = [
                trust_data["uncertainty_percentage"] / 100,
                trust_data["fairwashing_percentage"] / 100,
                trust_data["consistency_percentage"] / 100,
                trust_data["robustness_percentage"] / 100
            ]
            
            # Nombre de variables
            N = len(categories)
            
            # Angles pour chaque axe (divisé uniformément)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Fermer le graphique
            
            # Valeurs (répéter la première valeur pour fermer le graphique)
            values += values[:1]
            
            # Tracé du graphique
            ax.plot(angles, values, linewidth=2, linestyle='solid', label="Scores de confiance")
            ax.fill(angles, values, alpha=0.25)
            
            # Ajout des axes
            plt.xticks(angles[:-1], categories)
            
            # Ajout des labels pour chaque axe
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
            plt.ylim(0, 1)
            
            # Titre
            plt.title("Radar des Scores de Confiance", size=14, y=1.1)
            
            # Sauvegarde temporaire
            chart_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Ajout au PDF
            pdf.image(chart_path, x=30, w=150)
            
            # Nettoyage
            os.unlink(chart_path)
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique radar: {e}")
            # Ajout d'un message d'erreur dans le PDF
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, "Erreur lors de la génération du graphique radar", 0, 1)
