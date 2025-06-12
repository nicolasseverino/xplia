"""
Générateur avancé de rapports de conformité pour XPLIA
======================================================

Ce module implémente un système de génération de rapports de conformité
multi-format, multilingue et hautement personnalisable, avec intégration
du système de registre XPLIA pour une découverte dynamique des composants.

Caractéristiques avancées:
- Support multi-format (PDF, HTML, JSON, CSV, XML)
- Multilinguisme complet avec fichiers de traduction
- Templates personnalisables par secteur d'activité
- Signatures numériques et vérification
- Intégration avec le système de registre XPLIA
- Visualisations et métriques de conformité
"""

import logging
import os
import datetime
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union, Type, Set
from pathlib import Path
from enum import Enum

from ..core.registry import component_registry
from .report_base import ReportConfig, ReportContent, ReportFormat, BaseReportGenerator

# Import conditionnel pour ne pas casser si tous les formateurs ne sont pas encore implémentés
try:
    from .formatters import FORMATTER_REGISTRY
    HAS_ALL_FORMATTERS = True
except (ImportError, AttributeError):
    # Fallback si les formateurs ne sont pas encore tous implémentés
    HAS_ALL_FORMATTERS = False
    FORMATTER_REGISTRY = {}

logger = logging.getLogger(__name__)

class ComplianceReportGenerator:
    """
    Générateur avancé de rapports de conformité multi-format pour XPLIA.
    
    Cette classe implémente un système sophistiqué de génération de rapports
    de conformité avec support multi-format, multilinguisme, et templates
    personnalisables. Elle s'intègre avec le système de registre XPLIA
    pour une découverte dynamique des composants et des extensions.
    """
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                organization: str = "",
                responsible: str = "",
                language: str = "fr"):
        """
        Initialise le générateur de rapports de conformité.
        
        Args:
            config: Configuration avancée (optionnelle)
            organization: Nom de l'organisation (pour rétrocompatibilité)
            responsible: Nom du responsable (pour rétrocompatibilité)
            language: Code de langue (pour rétrocompatibilité)
        """
        # Construction de la configuration
        self._config = ReportConfig(
            organization=config.get("organization", organization) if config else organization,
            responsible=config.get("responsible", responsible) if config else responsible,
            language=config.get("language", language) if config else language,
            logo_path=config.get("logo_path") if config else None,
            template_name=config.get("template_name", "standard") if config else "standard",
            include_charts=config.get("include_charts", True) if config else True,
            include_signatures=config.get("include_signatures", True) if config else True,
            include_verification_qr=config.get("include_verification_qr", False) if config else False,
            metadata=config.get("metadata", {}) if config else {},
            style_config=config.get("style_config", {}) if config else {},
            verification_url=config.get("verification_url") if config else None
        )
        
        # Pour compatibilité avec l'ancienne API
        self.organization = self._config.organization
        self.responsible = self._config.responsible
        self.language = self._config.language
        
        # Recherche et initialisation des générateurs de format
        self._format_generators = {}
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialise les générateurs de format disponibles."""
        if HAS_ALL_FORMATTERS:
            # Utilisation des générateurs officiels si disponibles
            for format_type, generator_class in FORMATTER_REGISTRY.items():
                self._format_generators[format_type] = generator_class(self._config)
        else:
            # Recherche des générateurs enregistrés dans le registre XPLIA
            try:
                formatter_components = component_registry.get_components_by_tag("report_formatter")
                
                for name, (component_class, metadata) in formatter_components.items():
                    try:
                        format_type = getattr(ReportFormat, metadata.get("format", "").upper(), None)
                        if format_type:
                            self._format_generators[format_type] = component_class(self._config)
                            logger.debug(f"Générateur de rapports {format_type.value} chargé depuis le registre")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'initialisation du générateur {name}: {e}")
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération des formateurs depuis le registre: {e}")
        
        # Fallback pour la rétrocompatibilité
        if not self._format_generators:
            logger.warning("Aucun générateur de format disponible. Utilisation des méthodes de compatibilité.")
    
    def generate_report(self, 
                       compliance_data: Dict[str, Any],
                       format: Union[str, ReportFormat] = "pdf",
                       output_path: Optional[str] = None) -> Optional[Union[str, bytes]]:
        """
        Génère un rapport de conformité dans le format spécifié.
        
        Args:
            compliance_data: Données de conformité
            format: Format du rapport (pdf, html, json, csv, xml)
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Contenu du rapport si output_path est None, sinon None
        """
        # Conversion du format en ReportFormat si nécessaire
        if isinstance(format, str):
            try:
                format = ReportFormat(format.lower())
            except ValueError:
                supported = ", ".join([f.value for f in ReportFormat])
                raise ValueError(f"Format non supporté: {format}. Formats supportés: {supported}")
        
        # Préparation du contenu
        content = self._prepare_content(compliance_data)
        
        # Génération du rapport avec le générateur approprié
        if format in self._format_generators:
            return self._format_generators[format].generate(content, output_path)
        
        # Fallback pour la rétrocompatibilité
        if format == ReportFormat.PDF and output_path:
            logger.warning("Utilisation du générateur PDF de compatibilité")
            self._legacy_generate_pdf(
                compliance_data.get("audit_trail", []), 
                compliance_data.get("decision_log", []),
                output_path
            )
            return None
        elif format == ReportFormat.HTML:
            logger.warning("Utilisation du générateur HTML de compatibilité")
            return self._legacy_generate_html(
                compliance_data.get("audit_trail", []),
                compliance_data.get("decision_log", [])
            )
        
        raise NotImplementedError(f"Format {format.value} non supporté")
    
    def _prepare_content(self, compliance_data: Dict[str, Any]) -> ReportContent:
        """
        Convertit les données de conformité en objet ReportContent.
        
        Args:
            compliance_data: Données de conformité
            
        Returns:
            Objet ReportContent
        """
        return ReportContent(
            title=compliance_data.get("title", "Rapport de conformité XPLIA"),
            timestamp=compliance_data.get("timestamp", datetime.datetime.now().isoformat()),
            summary=compliance_data.get("summary"),
            compliance_score=compliance_data.get("compliance_score"),
            audit_trail=compliance_data.get("audit_trail", []),
            decision_log=compliance_data.get("decision_log", []),
            issues=compliance_data.get("issues", []),
            recommendations=compliance_data.get("recommendations", []),
            metadata=compliance_data.get("metadata", {})
        )
    
    # Méthodes de compatibilité avec l'ancienne API
    
    def generate_pdf(self, audit_trail: List[Dict], decision_log: List[Dict], output_path: str):
        """Méthode de compatibilité avec l'ancienne API."""
        logger.warning("Méthode 'generate_pdf' dépréciée. Utiliser 'generate_report' à la place.")
        self._legacy_generate_pdf(audit_trail, decision_log, output_path)
    
    def generate_html(self, audit_trail: List[Dict], decision_log: List[Dict]) -> str:
        """Méthode de compatibilité avec l'ancienne API."""
        logger.warning("Méthode 'generate_html' dépréciée. Utiliser 'generate_report' à la place.")
        return self._legacy_generate_html(audit_trail, decision_log)
    
    def _legacy_generate_pdf(self, audit_trail: List[Dict], decision_log: List[Dict], output_path: str):
        """Implémentation de compatibilité pour generate_pdf."""
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Rapport de conformité XPLIA", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Organisation : {self.organization}", ln=True)
        pdf.cell(200, 10, txt=f"Responsable : {self.responsible}", ln=True)
        pdf.cell(200, 10, txt=f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)
        
        pdf.cell(200, 10, txt="--- Journal des demandes d'explication (RGPD) ---", ln=True)
        for entry in audit_trail:
            pdf.multi_cell(0, 10, txt=str(entry))
        
        pdf.ln(10)
        pdf.cell(200, 10, txt="--- Journal des décisions (AI Act) ---", ln=True)
        for entry in decision_log:
            pdf.multi_cell(0, 10, txt=str(entry))
        
        pdf.output(output_path)
    
    def _legacy_generate_html(self, audit_trail: List[Dict], decision_log: List[Dict]) -> str:
        """Implémentation de compatibilité pour generate_html."""
        html = f"<h1>Rapport de conformité XPLIA</h1>"
        html += f"<p>Organisation : {self.organization}<br>Responsable : {self.responsible}<br>Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
        
        html += "<h2>Journal des demandes d'explication (RGPD)</h2><ul>"
        for entry in audit_trail:
            html += f"<li>{entry}</li>"
        html += "</ul>"
        
        html += "<h2>Journal des décisions (AI Act)</h2><ul>"
        for entry in decision_log:
            html += f"<li>{entry}</li>"
        html += "</ul>"
        
        return html
