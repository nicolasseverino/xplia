"""
Profils d'Audience et Gestion des Utilisateurs
===========================================

Ce module implémente la gestion des profils d'utilisateurs et d'audience
pour permettre une adaptation personnalisée des explications.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from enum import Enum, auto
from dataclasses import dataclass, field
import json
import os
import time

from ...core.base import AudienceLevel


class ExplanationPreference(Enum):
    """Préférences d'explication des utilisateurs."""
    VISUAL = auto()         # Préfère les explications visuelles
    TEXTUAL = auto()        # Préfère les explications textuelles
    TECHNICAL = auto()      # Préfère les détails techniques
    SIMPLIFIED = auto()     # Préfère les explications simplifiées
    COMPREHENSIVE = auto()  # Préfère les explications exhaustives
    CONCISE = auto()        # Préfère les explications concises
    INTERACTIVE = auto()    # Préfère les explications interactives
    COMPARATIVE = auto()    # Préfère les comparaisons


class DomainExpertise(Enum):
    """Niveaux d'expertise dans différents domaines."""
    NOVICE = 1
    BASIC = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


@dataclass
class UserProfile:
    """
    Profil d'un utilisateur avec ses préférences et son niveau d'expertise.
    """
    user_id: str
    name: Optional[str] = None
    audience_level: AudienceLevel = AudienceLevel.TECHNICAL
    preferences: Set[ExplanationPreference] = field(default_factory=set)
    domain_expertise: Dict[str, DomainExpertise] = field(default_factory=dict)
    language: str = "fr"
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def update_preference(self, preference: ExplanationPreference, add: bool = True) -> None:
        """
        Ajoute ou supprime une préférence d'explication.
        
        Args:
            preference: La préférence à modifier
            add: Si True, ajoute la préférence; sinon la supprime
        """
        if add:
            self.preferences.add(preference)
        elif preference in self.preferences:
            self.preferences.remove(preference)
        
        self.updated_at = time.time()
    
    def set_expertise(self, domain: str, level: DomainExpertise) -> None:
        """
        Définit le niveau d'expertise dans un domaine spécifique.
        
        Args:
            domain: Le domaine concerné
            level: Le niveau d'expertise
        """
        self.domain_expertise[domain] = level
        self.updated_at = time.time()
    
    def record_interaction(self, 
                          explanation_id: str, 
                          feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Enregistre une interaction avec une explication.
        
        Args:
            explanation_id: Identifiant de l'explication
            feedback: Retour d'information sur l'explication (optionnel)
        """
        self.interaction_history.append({
            "explanation_id": explanation_id,
            "timestamp": time.time(),
            "feedback": feedback or {}
        })
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le profil en dictionnaire pour la sérialisation.
        
        Returns:
            Dictionnaire représentant le profil
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "audience_level": self.audience_level.value,
            "preferences": [pref.name for pref in self.preferences],
            "domain_expertise": {domain: level.value for domain, level in self.domain_expertise.items()},
            "language": self.language,
            "interaction_history": self.interaction_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """
        Crée un profil à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire contenant les données du profil
            
        Returns:
            Instance de UserProfile
        """
        preferences = set()
        for pref_name in data.get("preferences", []):
            try:
                preferences.add(ExplanationPreference[pref_name])
            except KeyError:
                logging.warning(f"Préférence inconnue ignorée: {pref_name}")
        
        domain_expertise = {}
        for domain, level_value in data.get("domain_expertise", {}).items():
            try:
                domain_expertise[domain] = DomainExpertise(level_value)
            except ValueError:
                logging.warning(f"Niveau d'expertise invalide ignoré pour {domain}: {level_value}")
        
        return cls(
            user_id=data["user_id"],
            name=data.get("name"),
            audience_level=AudienceLevel(data.get("audience_level", AudienceLevel.TECHNICAL.value)),
            preferences=preferences,
            domain_expertise=domain_expertise,
            language=data.get("language", "fr"),
            interaction_history=data.get("interaction_history", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time())
        )


class AudienceProfileManager:
    """
    Gestionnaire de profils d'utilisateurs et d'audience.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialise le gestionnaire de profils.
        
        Args:
            storage_path: Chemin vers le stockage des profils (optionnel)
        """
        self.profiles: Dict[str, UserProfile] = {}
        self.storage_path = storage_path
        
        # Charger les profils existants si un chemin de stockage est spécifié
        if storage_path and os.path.exists(storage_path):
            self._load_profiles()
    
    def get_profile(self, user_id: str) -> UserProfile:
        """
        Récupère le profil d'un utilisateur, ou en crée un nouveau si inexistant.
        
        Args:
            user_id: Identifiant de l'utilisateur
            
        Returns:
            Profil de l'utilisateur
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        
        return self.profiles[user_id]
    
    def update_profile(self, profile: UserProfile) -> None:
        """
        Met à jour un profil d'utilisateur.
        
        Args:
            profile: Profil à mettre à jour
        """
        profile.updated_at = time.time()
        self.profiles[profile.user_id] = profile
        
        # Sauvegarder les profils si un chemin de stockage est spécifié
        if self.storage_path:
            self._save_profiles()
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Supprime un profil d'utilisateur.
        
        Args:
            user_id: Identifiant de l'utilisateur
            
        Returns:
            True si le profil a été supprimé, False sinon
        """
        if user_id in self.profiles:
            del self.profiles[user_id]
            
            # Sauvegarder les profils si un chemin de stockage est spécifié
            if self.storage_path:
                self._save_profiles()
            
            return True
        
        return False
    
    def _load_profiles(self) -> None:
        """Charge les profils depuis le stockage."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
                
                for profile_data in profiles_data:
                    try:
                        profile = UserProfile.from_dict(profile_data)
                        self.profiles[profile.user_id] = profile
                    except Exception as e:
                        logging.error(f"Erreur lors du chargement du profil: {str(e)}")
                
                logging.info(f"Chargement de {len(self.profiles)} profils utilisateur")
        except Exception as e:
            logging.error(f"Erreur lors du chargement des profils: {str(e)}")
    
    def _save_profiles(self) -> None:
        """Sauvegarde les profils dans le stockage."""
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            profiles_data = [profile.to_dict() for profile in self.profiles.values()]
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Sauvegarde de {len(self.profiles)} profils utilisateur")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des profils: {str(e)}")
    
    def get_audience_segment(self, 
                           criteria: Dict[str, Any]) -> List[UserProfile]:
        """
        Récupère un segment d'audience selon des critères spécifiques.
        
        Args:
            criteria: Critères de segmentation
            
        Returns:
            Liste des profils correspondant aux critères
        """
        matching_profiles = []
        
        for profile in self.profiles.values():
            match = True
            
            # Vérifier chaque critère
            for key, value in criteria.items():
                if key == "audience_level" and hasattr(profile, key):
                    if getattr(profile, key) != value:
                        match = False
                        break
                elif key == "preferences" and hasattr(profile, key):
                    if not all(pref in profile.preferences for pref in value):
                        match = False
                        break
                elif key == "domain_expertise" and hasattr(profile, key):
                    for domain, level in value.items():
                        if domain not in profile.domain_expertise or profile.domain_expertise[domain] < level:
                            match = False
                            break
                elif key == "language" and hasattr(profile, key):
                    if getattr(profile, key) != value:
                        match = False
                        break
            
            if match:
                matching_profiles.append(profile)
        
        return matching_profiles
    
    def analyze_audience_preferences(self) -> Dict[str, Any]:
        """
        Analyse les préférences globales de l'audience.
        
        Returns:
            Statistiques sur les préférences de l'audience
        """
        if not self.profiles:
            return {}
        
        # Compteurs pour les différentes préférences et niveaux
        preference_counts = {pref: 0 for pref in ExplanationPreference}
        audience_level_counts = {level: 0 for level in AudienceLevel}
        domain_expertise = {}
        languages = {}
        
        # Analyser tous les profils
        for profile in self.profiles.values():
            # Compter les préférences
            for pref in profile.preferences:
                preference_counts[pref] += 1
            
            # Compter les niveaux d'audience
            audience_level_counts[profile.audience_level] += 1
            
            # Agréger l'expertise par domaine
            for domain, level in profile.domain_expertise.items():
                if domain not in domain_expertise:
                    domain_expertise[domain] = {}
                
                if level not in domain_expertise[domain]:
                    domain_expertise[domain][level] = 0
                
                domain_expertise[domain][level] += 1
            
            # Compter les langues
            if profile.language not in languages:
                languages[profile.language] = 0
            
            languages[profile.language] += 1
        
        # Calculer les pourcentages
        total_profiles = len(self.profiles)
        
        preference_stats = {
            pref.name: {
                "count": count,
                "percentage": (count / total_profiles) * 100
            }
            for pref, count in preference_counts.items() if count > 0
        }
        
        audience_level_stats = {
            level.name: {
                "count": count,
                "percentage": (count / total_profiles) * 100
            }
            for level, count in audience_level_counts.items() if count > 0
        }
        
        language_stats = {
            lang: {
                "count": count,
                "percentage": (count / total_profiles) * 100
            }
            for lang, count in languages.items()
        }
        
        return {
            "total_profiles": total_profiles,
            "preferences": preference_stats,
            "audience_levels": audience_level_stats,
            "domain_expertise": domain_expertise,
            "languages": language_stats
        }
