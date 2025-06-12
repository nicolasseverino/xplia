"""
Système de traduction multilingue pour les rapports de conformité XPLIA
======================================================================

Ce module gère les traductions pour l'ensemble du système de rapports de conformité,
permettant une internationalisation complète et l'ajout facile de nouvelles langues.
"""

from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Dictionnaire des traductions par défaut
_DEFAULT_TRANSLATIONS = {
    "fr": {
        "report_title": "Rapport de conformité XPLIA",
        "organization": "Organisation",
        "responsible": "Responsable",
        "date": "Date",
        "audit_trail_title": "Journal des demandes d'explication (RGPD)",
        "decision_log_title": "Journal des décisions (AI Act)",
        "compliance_score": "Score de conformité",
        "compliance_status": "Statut de conformité",
        "issues_title": "Problèmes identifiés",
        "recommendations_title": "Recommandations",
        "verification_code": "Code de vérification",
        "compliant": "Conforme",
        "non_compliant": "Non conforme",
        "partially_compliant": "Partiellement conforme",
        "generated_on": "Généré le",
        "page": "Page",
        "of": "sur",
        "confidential": "CONFIDENTIEL",
        "gdpr_section": "Conformité RGPD",
        "ai_act_section": "Conformité AI Act",
        "hipaa_section": "Conformité HIPAA",
        "details": "Détails",
        "severity": "Gravité",
        "critical": "Critique",
        "high": "Élevée",
        "medium": "Moyenne",
        "low": "Faible",
        "metadata": "Métadonnées",
        "report_id": "Identifiant du rapport",
        "report_version": "Version du rapport",
        "model_name": "Nom du modèle",
        "model_version": "Version du modèle",
        "data_types": "Types de données",
        "legal_disclaimer": "Ce rapport est généré automatiquement et ne constitue pas un avis juridique."
    },
    "en": {
        "report_title": "XPLIA Compliance Report",
        "organization": "Organization",
        "responsible": "Responsible Person",
        "date": "Date",
        "audit_trail_title": "Explanation Request Log (GDPR)",
        "decision_log_title": "Decision Log (AI Act)",
        "compliance_score": "Compliance Score",
        "compliance_status": "Compliance Status",
        "issues_title": "Identified Issues",
        "recommendations_title": "Recommendations",
        "verification_code": "Verification Code",
        "compliant": "Compliant",
        "non_compliant": "Non-compliant",
        "partially_compliant": "Partially compliant",
        "generated_on": "Generated on",
        "page": "Page",
        "of": "of",
        "confidential": "CONFIDENTIAL",
        "gdpr_section": "GDPR Compliance",
        "ai_act_section": "AI Act Compliance",
        "hipaa_section": "HIPAA Compliance",
        "details": "Details",
        "severity": "Severity",
        "critical": "Critical",
        "high": "High",
        "medium": "Medium", 
        "low": "Low",
        "metadata": "Metadata",
        "report_id": "Report ID",
        "report_version": "Report Version",
        "model_name": "Model Name",
        "model_version": "Model Version",
        "data_types": "Data Types",
        "legal_disclaimer": "This report is automatically generated and does not constitute legal advice."
    },
    "de": {
        "report_title": "XPLIA Compliance-Bericht",
        "organization": "Organisation",
        "responsible": "Verantwortliche Person",
        "date": "Datum",
        "audit_trail_title": "Erklärungsanforderungsprotokoll (DSGVO)",
        "decision_log_title": "Entscheidungsprotokoll (KI-Verordnung)",
        "compliance_score": "Compliance-Bewertung",
        "compliance_status": "Compliance-Status",
        "issues_title": "Identifizierte Probleme",
        "recommendations_title": "Empfehlungen",
        "verification_code": "Verifizierungscode",
        "compliant": "Konform",
        "non_compliant": "Nicht konform",
        "partially_compliant": "Teilweise konform",
        "generated_on": "Erstellt am",
        "page": "Seite",
        "of": "von",
        "confidential": "VERTRAULICH",
        "gdpr_section": "DSGVO-Konformität",
        "ai_act_section": "KI-Verordnungskonformität",
        "hipaa_section": "HIPAA-Konformität",
        "details": "Details",
        "severity": "Schweregrad",
        "critical": "Kritisch",
        "high": "Hoch",
        "medium": "Mittel", 
        "low": "Niedrig",
        "metadata": "Metadaten",
        "report_id": "Bericht-ID",
        "report_version": "Berichtversion",
        "model_name": "Modellname",
        "model_version": "Modellversion",
        "data_types": "Datentypen",
        "legal_disclaimer": "Dieser Bericht wird automatisch generiert und stellt keine Rechtsberatung dar."
    },
    "es": {
        "report_title": "Informe de Cumplimiento XPLIA",
        "organization": "Organización",
        "responsible": "Persona Responsable",
        "date": "Fecha",
        "audit_trail_title": "Registro de Solicitudes de Explicación (RGPD)",
        "decision_log_title": "Registro de Decisiones (Ley de IA)",
        "compliance_score": "Puntuación de Cumplimiento",
        "compliance_status": "Estado de Cumplimiento",
        "issues_title": "Problemas Identificados",
        "recommendations_title": "Recomendaciones",
        "verification_code": "Código de Verificación",
        "compliant": "Cumple",
        "non_compliant": "No cumple",
        "partially_compliant": "Cumple parcialmente",
        "generated_on": "Generado el",
        "page": "Página",
        "of": "de",
        "confidential": "CONFIDENCIAL",
        "gdpr_section": "Cumplimiento RGPD",
        "ai_act_section": "Cumplimiento Ley de IA",
        "hipaa_section": "Cumplimiento HIPAA",
        "details": "Detalles",
        "severity": "Gravedad",
        "critical": "Crítica",
        "high": "Alta",
        "medium": "Media", 
        "low": "Baja",
        "metadata": "Metadatos",
        "report_id": "ID del Informe",
        "report_version": "Versión del Informe",
        "model_name": "Nombre del Modelo",
        "model_version": "Versión del Modelo",
        "data_types": "Tipos de Datos",
        "legal_disclaimer": "Este informe se genera automáticamente y no constituye asesoramiento legal."
    }
}

# Dictionnaire des traductions chargées depuis les fichiers
_LOADED_TRANSLATIONS = {}

def _load_translations():
    """Charge les traductions depuis les fichiers dans le répertoire des traductions."""
    translations_dir = Path(__file__).parent / "translations"
    
    if not translations_dir.exists():
        logger.warning("Répertoire de traductions non trouvé. Création...")
        translations_dir.mkdir(parents=True, exist_ok=True)
        
        # Création des fichiers de traduction par défaut
        for lang, translations in _DEFAULT_TRANSLATIONS.items():
            lang_file = translations_dir / f"{lang}.json"
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
        
        return _DEFAULT_TRANSLATIONS
    
    translations = {}
    
    # Chargement des fichiers de traduction
    for lang_file in translations_dir.glob("*.json"):
        lang = lang_file.stem
        
        try:
            with open(lang_file, "r", encoding="utf-8") as f:
                translations[lang] = json.load(f)
            logger.debug(f"Traductions chargées pour {lang}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des traductions pour {lang}: {e}")
            # Utilisation des traductions par défaut si disponibles
            if lang in _DEFAULT_TRANSLATIONS:
                translations[lang] = _DEFAULT_TRANSLATIONS[lang]
    
    # Ajout des langues par défaut manquantes
    for lang, default_translations in _DEFAULT_TRANSLATIONS.items():
        if lang not in translations:
            translations[lang] = default_translations
            
            # Création du fichier de traduction manquant
            lang_file = translations_dir / f"{lang}.json"
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(default_translations, f, ensure_ascii=False, indent=2)
    
    return translations


def get_translation(key: str, language: str = "en", default: Optional[str] = None) -> str:
    """
    Récupère la traduction d'une clé dans la langue spécifiée.
    
    Args:
        key: Clé de traduction
        language: Code de langue (fr, en, de, es, etc.)
        default: Valeur par défaut si la traduction n'est pas trouvée
        
    Returns:
        Texte traduit ou valeur par défaut ou clé originale si non trouvée
    """
    # Chargement des traductions si nécessaire
    if not _LOADED_TRANSLATIONS:
        global _LOADED_TRANSLATIONS
        _LOADED_TRANSLATIONS = _load_translations()
    
    # Utilisation de l'anglais comme fallback si la langue n'existe pas
    if language not in _LOADED_TRANSLATIONS:
        language = "en"
        logger.warning(f"Langue {language} non supportée. Utilisation de l'anglais.")
    
    # Récupération de la traduction
    translation = _LOADED_TRANSLATIONS[language].get(key)
    
    if translation is None:
        # Si la clé n'existe pas dans cette langue, essayons l'anglais
        if language != "en":
            translation = _LOADED_TRANSLATIONS["en"].get(key)
        
        # Si toujours pas trouvé, utilisation de la valeur par défaut ou de la clé
        if translation is None:
            translation = default if default is not None else key
            logger.warning(f"Traduction non trouvée pour '{key}' en {language}")
    
    return translation


def get_available_languages() -> Dict[str, str]:
    """
    Récupère la liste des langues disponibles avec leur nom dans la langue locale.
    
    Returns:
        Dictionnaire avec code de langue comme clé et nom de la langue comme valeur
    """
    # Chargement des traductions si nécessaire
    if not _LOADED_TRANSLATIONS:
        global _LOADED_TRANSLATIONS
        _LOADED_TRANSLATIONS = _load_translations()
    
    language_names = {
        "fr": "Français",
        "en": "English",
        "de": "Deutsch",
        "es": "Español",
    }
    
    return {lang: language_names.get(lang, lang) for lang in _LOADED_TRANSLATIONS.keys()}


def add_translation(language: str, key: str, value: str) -> None:
    """
    Ajoute ou met à jour une traduction pour une langue et une clé données.
    
    Args:
        language: Code de langue (fr, en, de, es, etc.)
        key: Clé de traduction
        value: Valeur traduite
    """
    # Chargement des traductions si nécessaire
    if not _LOADED_TRANSLATIONS:
        global _LOADED_TRANSLATIONS
        _LOADED_TRANSLATIONS = _load_translations()
    
    # Création d'une nouvelle langue si nécessaire
    if language not in _LOADED_TRANSLATIONS:
        _LOADED_TRANSLATIONS[language] = {}
    
    # Ajout ou mise à jour de la traduction
    _LOADED_TRANSLATIONS[language][key] = value
    
    # Sauvegarde dans le fichier
    translations_dir = Path(__file__).parent / "translations"
    translations_dir.mkdir(parents=True, exist_ok=True)
    
    lang_file = translations_dir / f"{language}.json"
    with open(lang_file, "w", encoding="utf-8") as f:
        json.dump(_LOADED_TRANSLATIONS[language], f, ensure_ascii=False, indent=2)
