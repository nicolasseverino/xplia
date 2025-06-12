"""
Générateur de rapports JSON pour XPLIA
====================================

Ce module implémente un générateur de rapports au format JSON pour le système
avancé de génération de rapports de conformité XPLIA.
"""

import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..report_base import BaseReportGenerator, ReportConfig, ReportContent

logger = logging.getLogger(__name__)

class JSONReportGenerator(BaseReportGenerator):
    """
    Générateur de rapports au format JSON.
    
    Cette classe génère des rapports de conformité au format JSON structuré,
    avec support pour les métadonnées, la signature et l'internationalisation.
    """
    
    def __init__(self, config: ReportConfig):
        """
        Initialise le générateur de rapports JSON.
        
        Args:
            config: Configuration du générateur
        """
        super().__init__(config)
    
    def generate(self, content: ReportContent, output_path: Optional[str] = None) -> Optional[str]:
        """
        Génère un rapport au format JSON.
        
        Args:
            content: Contenu du rapport
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Contenu JSON si output_path est None, sinon None
        """
        try:
            # Préparation des données
            report_data = self._prepare_json_data(content)
            
            # Conversion en JSON
            json_content = json.dumps(report_data, indent=2, ensure_ascii=False)
            
            # Sauvegarde ou retour
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(json_content)
                logger.info(f"Rapport JSON généré avec succès: {output_path}")
                return None
            else:
                return json_content
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport JSON: {e}")
            raise
    
    def _prepare_json_data(self, content: ReportContent) -> Dict[str, Any]:
        """
        Prépare les données pour le rapport JSON.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Dictionnaire avec données préparées pour le JSON
        """
        # Données de base du rapport
        report_data = content.to_dict()
        
        # Ajout des métadonnées du générateur
        report_data["generator"] = {
            "name": "XPLIA Compliance Report Generator",
            "version": "1.0.0",
            "format": "JSON",
            "timestamp": report_data.get("timestamp", ""),
            "config": {
                "organization": self.config.organization,
                "responsible": self.config.responsible,
                "language": self.config.language
            }
        }
        
        # Ajout des traductions
        translation_keys = [
            "report_title", "organization", "responsible", "date",
            "compliance_score", "compliance_status", "issues_title", 
            "recommendations_title"
        ]
        
        report_data["translations"] = {
            key: self._get_translation(key) for key in translation_keys
        }
        
        # Signature si demandée
        if self.config.include_signatures:
            report_data["signature"] = content.sign('xplia_secret')  # À remplacer par une vraie clé
            
        return report_data
