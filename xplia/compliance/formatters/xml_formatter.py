"""
Générateur de rapports XML pour XPLIA
===================================

Ce module implémente un générateur de rapports au format XML pour le système
avancé de génération de rapports de conformité XPLIA.
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from ..report_base import BaseReportGenerator, ReportConfig, ReportContent

logger = logging.getLogger(__name__)

class XMLReportGenerator(BaseReportGenerator):
    """
    Générateur de rapports au format XML.
    
    Cette classe génère des rapports de conformité au format XML structuré,
    adaptés pour l'échange de données et l'intégration avec d'autres systèmes.
    """
    
    def __init__(self, config: ReportConfig):
        """
        Initialise le générateur de rapports XML.
        
        Args:
            config: Configuration du générateur
        """
        super().__init__(config)
    
    def generate(self, content: ReportContent, output_path: Optional[str] = None) -> Optional[str]:
        """
        Génère un rapport au format XML.
        
        Args:
            content: Contenu du rapport
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Contenu XML si output_path est None, sinon None
        """
        try:
            # Création de l'arbre XML
            root = self._create_xml_tree(content)
            
            # Conversion en chaîne XML bien formatée
            xml_string = self._prettify_xml(root)
            
            # Sauvegarde ou retour
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(xml_string)
                logger.info(f"Rapport XML généré avec succès: {output_path}")
                return None
            else:
                return xml_string
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport XML: {e}")
            raise
    
    def _create_xml_tree(self, content: ReportContent) -> ET.Element:
        """
        Crée l'arbre XML du rapport.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Élément racine de l'arbre XML
        """
        # Élément racine du rapport
        root = ET.Element("ComplianceReport")
        
        # Attributs de l'élément racine
        root.set("timestamp", content.timestamp)
        root.set("language", self.config.language)
        root.set("version", "1.0")
        
        # Informations générales
        info = ET.SubElement(root, "ReportInfo")
        ET.SubElement(info, "Title").text = content.title
        ET.SubElement(info, "Organization").text = self.config.organization
        ET.SubElement(info, "Responsible").text = self.config.responsible
        ET.SubElement(info, "GenerationDate").text = datetime.now().isoformat()
        
        # Résumé
        if content.summary:
            ET.SubElement(root, "Summary").text = content.summary
        
        # Score de conformité
        if content.compliance_score:
            score_elem = ET.SubElement(root, "ComplianceScore")
            ET.SubElement(score_elem, "Overall").text = str(content.compliance_score.get("score", 0))
            
            # Détails du score par réglementations
            if "details" in content.compliance_score:
                details = ET.SubElement(score_elem, "Details")
                for regulation, score in content.compliance_score["details"].items():
                    reg_elem = ET.SubElement(details, "Regulation")
                    reg_elem.set("name", regulation)
                    reg_elem.text = str(score)
        
        # Journal d'audit
        if content.audit_trail:
            audit_elem = ET.SubElement(root, "AuditTrail")
            for entry in content.audit_trail:
                entry_elem = ET.SubElement(audit_elem, "Entry")
                for key, value in entry.items():
                    ET.SubElement(entry_elem, key.capitalize()).text = str(value)
        
        # Journal des décisions
        if content.decision_log:
            decisions_elem = ET.SubElement(root, "DecisionLog")
            for entry in content.decision_log:
                decision_elem = ET.SubElement(decisions_elem, "Decision")
                for key, value in entry.items():
                    ET.SubElement(decision_elem, key.capitalize()).text = str(value)
        
        # Problèmes identifiés
        if content.issues:
            issues_elem = ET.SubElement(root, "Issues")
            for issue in content.issues:
                issue_elem = ET.SubElement(issues_elem, "Issue")
                for key, value in issue.items():
                    ET.SubElement(issue_elem, key.capitalize()).text = str(value)
        
        # Recommandations
        if content.recommendations:
            recommendations_elem = ET.SubElement(root, "Recommendations")
            for recommendation in content.recommendations:
                ET.SubElement(recommendations_elem, "Recommendation").text = recommendation
        
        # Métadonnées
        if content.metadata:
            metadata_elem = ET.SubElement(root, "Metadata")
            for key, value in content.metadata.items():
                if isinstance(value, dict):
                    sub_elem = ET.SubElement(metadata_elem, key.capitalize())
                    for k, v in value.items():
                        ET.SubElement(sub_elem, k.capitalize()).text = str(v)
                else:
                    ET.SubElement(metadata_elem, key.capitalize()).text = str(value)
        
        # Signature si demandée
        if self.config.include_signatures:
            signature = content.sign('xplia_secret')  # À remplacer par une vraie clé
            ET.SubElement(root, "Signature").text = signature
        
        return root
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """
        Formate un élément XML en une chaîne bien formatée.
        
        Args:
            elem: Élément XML à formater
            
        Returns:
            Chaîne XML bien formatée
        """
        # Conversion en chaîne
        rough_string = ET.tostring(elem, 'utf-8')
        
        # Utilisation de minidom pour le formatage
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
