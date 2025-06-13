"""
Mixin pour l'intégration des métriques de confiance dans les formatters
========================================================================

Ce module fournit un mixin pour intégrer facilement les métriques de confiance
dans les différents formatters de rapports XPLIA.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class TrustFormatterMixin:
    """
    Mixin pour l'intégration des métriques de confiance dans les formatters.
    
    Cette classe fournit des méthodes communes pour traiter et formater
    les métriques de confiance dans les différents types de rapports.
    """
    
    def _process_trust_metrics(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite les métriques de confiance d'une explication.
        
        Args:
            explanation: Dictionnaire contenant l'explication et les métriques de confiance
            
        Returns:
            Dictionnaire avec les métriques de confiance formatées pour l'affichage
        """
        # Initialisation des valeurs par défaut
        trust_data = {
            "has_trust_metrics": False,
            "trust_score": 0,
            "trust_score_percentage": 0,
            "trust_level": "unknown",
            "trust_level_label": "Inconnu",
            "trust_level_class": "moderate",
            "trust_summary": "Aucune métrique de confiance disponible.",
            "uncertainty_score": "N/A",
            "uncertainty_percentage": 0,
            "uncertainty_class": "moderate",
            "uncertainty_types": [],
            "fairwashing_score": "N/A",
            "fairwashing_percentage": 0,
            "fairwashing_class": "moderate",
            "detected_fairwashing_types": [],
            "consistency_score": "N/A",
            "consistency_percentage": 0,
            "consistency_class": "moderate",
            "robustness_score": "N/A",
            "robustness_percentage": 0,
            "robustness_class": "moderate",
            "trust_recommendations": []
        }
        
        # Vérification de la présence des métriques de confiance
        if not explanation or "confidence_report" not in explanation:
            return trust_data
            
        confidence_report = explanation.get("confidence_report", {})
        if not confidence_report:
            return trust_data
            
        # Extraction des métriques de base
        trust_score = confidence_report.get("trust_score", {})
        if not trust_score:
            return trust_data
            
        # Mise à jour des données avec les métriques disponibles
        trust_data["has_trust_metrics"] = True
        
        # Score global et niveau de confiance
        global_trust = trust_score.get("global_trust", 0)
        trust_data["trust_score"] = round(global_trust * 10, 1)  # Échelle 0-10
        trust_data["trust_score_percentage"] = round(global_trust * 100)  # Pourcentage
        
        # Niveau de confiance
        trust_level = trust_score.get("trust_level", "MODERATE")
        trust_data["trust_level"] = trust_level.lower()
        
        # Mapping des niveaux de confiance pour l'affichage
        level_mapping = {
            "very_low": {"label": "Très Faible", "class": "very-low"},
            "low": {"label": "Faible", "class": "low"},
            "moderate": {"label": "Modéré", "class": "moderate"},
            "high": {"label": "Élevé", "class": "high"},
            "very_high": {"label": "Très Élevé", "class": "very-high"}
        }
        
        level_info = level_mapping.get(trust_level.lower(), {"label": "Modéré", "class": "moderate"})
        trust_data["trust_level_label"] = level_info["label"]
        trust_data["trust_level_class"] = level_info["class"]
        
        # Résumé
        trust_data["trust_summary"] = confidence_report.get("summary", "Aucun résumé disponible.")
        
        # Métriques détaillées
        self._process_uncertainty_metrics(trust_data, trust_score, confidence_report)
        self._process_fairwashing_metrics(trust_data, trust_score, confidence_report)
        self._process_consistency_metrics(trust_data, trust_score)
        self._process_robustness_metrics(trust_data, trust_score)
        
        # Recommandations
        trust_data["trust_recommendations"] = confidence_report.get("recommendations", [])
        
        return trust_data
    
    def _process_uncertainty_metrics(self, trust_data: Dict[str, Any], 
                                    trust_score: Dict[str, Any], 
                                    confidence_report: Dict[str, Any]) -> None:
        """
        Traite les métriques d'incertitude.
        
        Args:
            trust_data: Dictionnaire des données de confiance à mettre à jour
            trust_score: Score de confiance global
            confidence_report: Rapport de confiance complet
        """
        uncertainty_trust = trust_score.get("uncertainty_trust", 0)
        trust_data["uncertainty_score"] = f"{round(uncertainty_trust * 10, 1)}/10"
        trust_data["uncertainty_percentage"] = round(uncertainty_trust * 100)
        
        # Classe CSS basée sur le score
        if uncertainty_trust >= 0.8:
            trust_data["uncertainty_class"] = "excellent"
        elif uncertainty_trust >= 0.6:
            trust_data["uncertainty_class"] = "good"
        elif uncertainty_trust >= 0.4:
            trust_data["uncertainty_class"] = "moderate"
        elif uncertainty_trust >= 0.2:
            trust_data["uncertainty_class"] = "poor"
        else:
            trust_data["uncertainty_class"] = "critical"
            
        # Types d'incertitude
        uncertainty_metrics = confidence_report.get("uncertainty_metrics", {})
        if uncertainty_metrics:
            types = []
            
            # Extraction des différents types d'incertitude
            for type_name, value in [
                ("Aléatoire", uncertainty_metrics.get("aleatoric_uncertainty", 0)),
                ("Épistémique", uncertainty_metrics.get("epistemic_uncertainty", 0)),
                ("Structurelle", uncertainty_metrics.get("structural_uncertainty", 0)),
                ("Approximation", uncertainty_metrics.get("approximation_uncertainty", 0))
            ]:
                if value > 0:
                    types.append((type_name, f"{round(value * 100)}%"))
                    
            trust_data["uncertainty_types"] = types
    
    def _process_fairwashing_metrics(self, trust_data: Dict[str, Any], 
                                    trust_score: Dict[str, Any], 
                                    confidence_report: Dict[str, Any]) -> None:
        """
        Traite les métriques de fairwashing.
        
        Args:
            trust_data: Dictionnaire des données de confiance à mettre à jour
            trust_score: Score de confiance global
            confidence_report: Rapport de confiance complet
        """
        fairwashing_trust = trust_score.get("fairwashing_trust", 0)
        trust_data["fairwashing_score"] = f"{round(fairwashing_trust * 10, 1)}/10"
        trust_data["fairwashing_percentage"] = round(fairwashing_trust * 100)
        
        # Classe CSS basée sur le score
        if fairwashing_trust >= 0.8:
            trust_data["fairwashing_class"] = "excellent"
        elif fairwashing_trust >= 0.6:
            trust_data["fairwashing_class"] = "good"
        elif fairwashing_trust >= 0.4:
            trust_data["fairwashing_class"] = "moderate"
        elif fairwashing_trust >= 0.2:
            trust_data["fairwashing_class"] = "poor"
        else:
            trust_data["fairwashing_class"] = "critical"
            
        # Types de fairwashing détectés
        fairwashing_audit = confidence_report.get("fairwashing_audit", {})
        if fairwashing_audit:
            detected_types = fairwashing_audit.get("detected_types", [])
            
            # Mapping des types pour l'affichage
            type_mapping = {
                "FEATURE_MASKING": "Masquage de Features",
                "IMPORTANCE_SHIFT": "Déplacement d'Importance",
                "BIAS_HIDING": "Dissimulation de Biais",
                "CHERRY_PICKING": "Sélection Biaisée",
                "THRESHOLD_MANIPULATION": "Manipulation de Seuils"
            }
            
            formatted_types = [type_mapping.get(t, t) for t in detected_types]
            trust_data["detected_fairwashing_types"] = formatted_types
    
    def _process_consistency_metrics(self, trust_data: Dict[str, Any], 
                                    trust_score: Dict[str, Any]) -> None:
        """
        Traite les métriques de cohérence.
        
        Args:
            trust_data: Dictionnaire des données de confiance à mettre à jour
            trust_score: Score de confiance global
        """
        consistency_trust = trust_score.get("consistency_trust", 0)
        trust_data["consistency_score"] = f"{round(consistency_trust * 10, 1)}/10"
        trust_data["consistency_percentage"] = round(consistency_trust * 100)
        
        # Classe CSS basée sur le score
        if consistency_trust >= 0.8:
            trust_data["consistency_class"] = "excellent"
        elif consistency_trust >= 0.6:
            trust_data["consistency_class"] = "good"
        elif consistency_trust >= 0.4:
            trust_data["consistency_class"] = "moderate"
        elif consistency_trust >= 0.2:
            trust_data["consistency_class"] = "poor"
        else:
            trust_data["consistency_class"] = "critical"
    
    def _process_robustness_metrics(self, trust_data: Dict[str, Any], 
                                   trust_score: Dict[str, Any]) -> None:
        """
        Traite les métriques de robustesse.
        
        Args:
            trust_data: Dictionnaire des données de confiance à mettre à jour
            trust_score: Score de confiance global
        """
        robustness_trust = trust_score.get("robustness_trust", 0)
        trust_data["robustness_score"] = f"{round(robustness_trust * 10, 1)}/10"
        trust_data["robustness_percentage"] = round(robustness_trust * 100)
        
        # Classe CSS basée sur le score
        if robustness_trust >= 0.8:
            trust_data["robustness_class"] = "excellent"
        elif robustness_trust >= 0.6:
            trust_data["robustness_class"] = "good"
        elif robustness_trust >= 0.4:
            trust_data["robustness_class"] = "moderate"
        elif robustness_trust >= 0.2:
            trust_data["robustness_class"] = "poor"
        else:
            trust_data["robustness_class"] = "critical"
    
    def _get_trust_metrics_template(self) -> str:
        """
        Récupère le template HTML pour les métriques de confiance.
        
        Returns:
            Contenu du template HTML
        """
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "templates", "html", "trust_metrics.html"
        )
        
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.warning(f"Template de métriques de confiance non trouvé: {template_path}")
            return "<!-- Template de métriques de confiance non disponible -->"
