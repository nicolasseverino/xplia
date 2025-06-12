"""
Base du système de génération de rapports de conformité pour XPLIA
=================================================================

Ce module fournit les classes et fonctions de base pour le système avancé
de génération de rapports de conformité XPLIA, avec support multi-format,
multilinguisme, et personnalisation avancée.
"""

import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import os

logger = logging.getLogger(__name__)

class ReportFormat(Enum):
    """Types de formats supportés pour les rapports."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XML = "xml"


@dataclass
class ReportConfig:
    """Configuration pour la génération de rapports."""
    organization: str = ""
    responsible: str = ""
    language: str = "fr"
    logo_path: Optional[str] = None
    template_name: str = "standard"
    include_charts: bool = True
    include_signatures: bool = True
    include_verification_qr: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    style_config: Dict[str, Any] = field(default_factory=dict)
    verification_url: Optional[str] = None


@dataclass
class ReportContent:
    """Contenu d'un rapport de conformité."""
    title: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    summary: Optional[str] = None
    compliance_score: Optional[Dict[str, Any]] = None
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    attachments: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le contenu en dictionnaire."""
        return {
            'title': self.title,
            'timestamp': self.timestamp,
            'summary': self.summary,
            'compliance_score': self.compliance_score,
            'audit_trail': self.audit_trail,
            'decision_log': self.decision_log,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'attachments': self.attachments,
            'metadata': self.metadata
        }
    
    def sign(self, secret_key: str) -> str:
        """Génère une signature cryptographique du contenu."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256((content + secret_key).encode()).hexdigest()


class BaseReportGenerator:
    """
    Classe de base pour les générateurs de rapports spécifiques au format.
    
    Cette classe fournit les fonctionnalités communes à tous les générateurs
    de rapports, quel que soit le format de sortie.
    """
    
    def __init__(self, config: ReportConfig):
        self.config = config
        
        # Chemin des templates
        self._template_dir = Path(__file__).parent / "templates"
        if not self._template_dir.exists():
            self._template_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, content: ReportContent, output_path: Optional[str] = None) -> Any:
        """
        Génère un rapport dans le format spécifique implémenté par la sous-classe.
        
        Args:
            content: Contenu du rapport
            output_path: Chemin de sortie pour le rapport
            
        Returns:
            Le contenu du rapport ou None si sauvegardé dans output_path
        """
        raise NotImplementedError("Les sous-classes doivent implémenter cette méthode")
    
    def _get_translation(self, key: str, language: Optional[str] = None) -> str:
        """
        Récupère la traduction d'une clé dans la langue spécifiée ou celle configurée.
        
        Args:
            key: Clé de traduction
            language: Langue (si None, utilise celle de la configuration)
            
        Returns:
            Texte traduit ou la clé originale si non trouvée
        """
        from .translations import get_translation
        lang = language or self.config.language
        return get_translation(key, lang)
    
    def _get_template_path(self, template_name: Optional[str] = None) -> Path:
        """
        Récupère le chemin du template spécifié ou celui configuré.
        
        Args:
            template_name: Nom du template (si None, utilise celui de la configuration)
            
        Returns:
            Chemin du template
        """
        name = template_name or self.config.template_name
        template_path = self._template_dir / f"{name}.template"
        
        if not template_path.exists():
            logger.warning(f"Template {name} non trouvé, utilisation du template standard")
            template_path = self._template_dir / "standard.template"
            
            # Création du template standard s'il n'existe pas
            if not template_path.exists():
                self._create_standard_template(template_path)
        
        return template_path
    
    def _create_standard_template(self, path: Path) -> None:
        """
        Crée un template standard.
        
        Args:
            path: Chemin où créer le template
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# {{title}}\n\n")
            f.write("**{{organization}}**  \n")
            f.write("{{responsible}}  \n")
            f.write("{{date}}  \n\n")
            f.write("## {{compliance_score_title}}\n\n")
            f.write("{{compliance_score_value}}  \n\n")
            f.write("## {{audit_trail_title}}\n\n")
            f.write("{{audit_trail_content}}  \n\n")
            f.write("## {{decision_log_title}}\n\n")
            f.write("{{decision_log_content}}  \n\n")
            f.write("## {{issues_title}}\n\n")
            f.write("{{issues_content}}  \n\n")
            f.write("## {{recommendations_title}}\n\n")
            f.write("{{recommendations_content}}  \n\n")
    
    def _prepare_report_data(self, content: ReportContent) -> Dict[str, Any]:
        """
        Prépare les données pour le rapport avec traductions.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Dictionnaire avec données préparées pour le template
        """
        data = {
            'title': self._get_translation('report_title'),
            'organization': self.config.organization,
            'responsible': self.config.responsible,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'compliance_score_title': self._get_translation('compliance_score'),
            'audit_trail_title': self._get_translation('audit_trail_title'),
            'decision_log_title': self._get_translation('decision_log_title'),
            'issues_title': self._get_translation('issues_title'),
            'recommendations_title': self._get_translation('recommendations_title'),
            'verification_code': self._get_translation('verification_code'),
        }
        
        # Données de contenu
        if content.compliance_score:
            data['compliance_score_value'] = f"{content.compliance_score.get('score', 0):.2f}"
        else:
            data['compliance_score_value'] = "N/A"
        
        # Signature si demandée
        if self.config.include_signatures:
            data['signature'] = content.sign('xplia_secret')  # À remplacer par une vraie clé
        
        return data
