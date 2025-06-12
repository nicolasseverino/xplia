"""
Générateur de rapports CSV pour XPLIA
===================================

Ce module implémente un générateur de rapports au format CSV pour le système
avancé de génération de rapports de conformité XPLIA.
"""

import csv
import io
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..report_base import BaseReportGenerator, ReportConfig, ReportContent

logger = logging.getLogger(__name__)

class CSVReportGenerator(BaseReportGenerator):
    """
    Générateur de rapports au format CSV.
    
    Cette classe génère des rapports de conformité au format CSV,
    adaptés pour l'analyse de données et l'importation dans des outils de BI.
    """
    
    def __init__(self, config: ReportConfig):
        """
        Initialise le générateur de rapports CSV.
        
        Args:
            config: Configuration du générateur
        """
        super().__init__(config)
    
    def generate(self, content: ReportContent, output_path: Optional[str] = None) -> Optional[str]:
        """
        Génère un rapport au format CSV.
        
        Args:
            content: Contenu du rapport
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Contenu CSV si output_path est None, sinon None
        """
        try:
            # Création des différentes sections CSV
            metadata_csv = self._generate_metadata_csv(content)
            audit_trail_csv = self._generate_audit_trail_csv(content)
            issues_csv = self._generate_issues_csv(content)
            
            # Combinaison des différentes sections
            output = io.StringIO()
            output.write(metadata_csv + "\n\n" + audit_trail_csv + "\n\n" + issues_csv)
            
            csv_content = output.getvalue()
            
            # Sauvegarde ou retour
            if output_path:
                with open(output_path, "w", encoding="utf-8", newline='') as f:
                    f.write(csv_content)
                logger.info(f"Rapport CSV généré avec succès: {output_path}")
                return None
            else:
                return csv_content
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport CSV: {e}")
            raise
    
    def _generate_metadata_csv(self, content: ReportContent) -> str:
        """
        Génère la section des métadonnées au format CSV.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Section CSV des métadonnées
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # En-tête
        writer.writerow([self._get_translation("report_title")])
        writer.writerow([])
        
        # Informations générales
        writer.writerow([self._get_translation("organization"), self.config.organization])
        writer.writerow([self._get_translation("responsible"), self.config.responsible])
        writer.writerow([self._get_translation("date"), content.timestamp])
        
        # Score de conformité
        if content.compliance_score:
            writer.writerow([
                self._get_translation("compliance_score"),
                content.compliance_score.get("score", "N/A")
            ])
            
            # Détails du score par réglementations
            if "details" in content.compliance_score:
                writer.writerow([])
                writer.writerow([self._get_translation("details")])
                
                for regulation, score in content.compliance_score["details"].items():
                    writer.writerow([regulation, score])
        
        return output.getvalue()
    
    def _generate_audit_trail_csv(self, content: ReportContent) -> str:
        """
        Génère la section de l'audit trail au format CSV.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Section CSV de l'audit trail
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([self._get_translation("audit_trail_title")])
        writer.writerow([])
        
        if not content.audit_trail:
            writer.writerow([self._get_translation("no_data")])
            return output.getvalue()
        
        # Détermination des en-têtes en fonction des clés du premier élément
        if content.audit_trail:
            headers = list(content.audit_trail[0].keys())
            writer.writerow(headers)
            
            # Écriture des lignes
            for entry in content.audit_trail:
                writer.writerow([str(entry.get(h, "")) for h in headers])
        
        return output.getvalue()
    
    def _generate_issues_csv(self, content: ReportContent) -> str:
        """
        Génère la section des problèmes identifiés au format CSV.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Section CSV des problèmes
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([self._get_translation("issues_title")])
        writer.writerow([])
        
        if not content.issues:
            writer.writerow([self._get_translation("no_data")])
            return output.getvalue()
        
        # Détermination des en-têtes en fonction des clés du premier élément
        if content.issues:
            headers = list(content.issues[0].keys())
            writer.writerow(headers)
            
            # Écriture des lignes
            for issue in content.issues:
                writer.writerow([str(issue.get(h, "")) for h in headers])
        
        # Ajout des recommandations si présentes
        if content.recommendations:
            writer.writerow([])
            writer.writerow([self._get_translation("recommendations_title")])
            
            for i, recommendation in enumerate(content.recommendations, 1):
                writer.writerow([f"{i}", recommendation])
        
        return output.getvalue()
