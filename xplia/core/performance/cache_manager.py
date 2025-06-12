"""
Gestionnaire de cache pour optimiser les performances.

Ce module fournit des mécanismes de mise en cache intelligents pour éviter
de recalculer des résultats intermédiaires coûteux dans la bibliothèque XPLIA.
"""

import os
import time
import hashlib
import pickle
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gestionnaire de cache pour les résultats intermédiaires.
    
    Cette classe fournit des mécanismes pour stocker et récupérer efficacement
    des résultats intermédiaires dans les traitements coûteux.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 max_size_mb: int = 512,
                 expiration_seconds: int = 3600 * 24 * 7):  # 1 semaine par défaut
        """
        Initialise le gestionnaire de cache.
        
        Args:
            cache_dir: Répertoire de stockage du cache.
                Si None, utilise un répertoire temporaire.
            max_size_mb: Taille maximale du cache en Mo
            expiration_seconds: Durée de validité des entrées en secondes
        """
        # Configuration du cache
        if cache_dir is None:
            import tempfile
            self.cache_dir = os.path.join(tempfile.gettempdir(), 'xplia_cache')
        else:
            self.cache_dir = cache_dir
        
        self.max_size_mb = max_size_mb
        self.expiration_seconds = expiration_seconds
        
        # S'assurer que le répertoire de cache existe
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Méta-informations du cache
        self._meta_file = os.path.join(self.cache_dir, 'meta.pkl')
        self._meta = self._load_meta()
    
    def _load_meta(self) -> Dict:
        """
        Charge les méta-informations du cache.
        
        Returns:
            Dictionnaire des méta-informations ou structure par défaut si inexistant
        """
        if os.path.exists(self._meta_file):
            try:
                with open(self._meta_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                logger.warning("Failed to load cache metadata, creating new one")
        
        # Créer une nouvelle structure de méta-données
        return {
            'entries': {},  # clé -> {path, size, timestamp}
            'total_size': 0  # taille totale en octets
        }
    
    def _save_meta(self):
        """Sauvegarde les méta-informations du cache."""
        try:
            with open(self._meta_file, 'wb') as f:
                pickle.dump(self._meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Génère une clé unique à partir des arguments.
        
        Args:
            *args, **kwargs: Arguments à hacher
            
        Returns:
            Clé de cache unique sous forme de chaîne hexadécimale
        """
        # Concaténer tous les arguments dans une représentation string
        combined = str(args) + str(sorted(kwargs.items()))
        
        # Calculer un hash SHA-256
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """
        Obtient le chemin de fichier pour une clé de cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Chemin complet du fichier de cache
        """
        # Utiliser les 2 premiers caractères comme sous-répertoire pour
        # répartir les fichiers et éviter d'avoir trop de fichiers dans un dossier
        subdir = key[:2]
        cache_subdir = os.path.join(self.cache_dir, subdir)
        os.makedirs(cache_subdir, exist_ok=True)
        
        return os.path.join(cache_subdir, f"{key}.pkl")
    
    def put(self, key: str, value: Any) -> bool:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé unique pour identifier la valeur
            value: Valeur à stocker (doit être sérialisable)
            
        Returns:
            True si le stockage a réussi, False sinon
        """
        # Vérifier l'espace disponible et nettoyer si nécessaire
        self._cleanup_if_needed()
        
        try:
            # Sérialiser et stocker la valeur
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Mettre à jour les méta-informations
            size = os.path.getsize(cache_path)
            self._meta['entries'][key] = {
                'path': cache_path,
                'size': size,
                'timestamp': time.time()
            }
            self._meta['total_size'] += size
            self._save_meta()
            
            return True
        except Exception as e:
            logger.error(f"Failed to cache value for key {key}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de la valeur à récupérer
            default: Valeur par défaut si non trouvée ou expirée
            
        Returns:
            Valeur mise en cache ou default si non trouvée/expirée
        """
        # Vérifier si la clé existe dans les méta-données
        if key not in self._meta['entries']:
            return default
        
        entry = self._meta['entries'][key]
        cache_path = entry['path']
        
        # Vérifier si l'entrée a expiré
        if time.time() - entry['timestamp'] > self.expiration_seconds:
            self._remove_entry(key)
            return default
        
        # Tenter de charger la valeur
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                # Le fichier a été supprimé manuellement
                self._remove_entry(key)
        except Exception as e:
            logger.error(f"Failed to load cached value for key {key}: {e}")
            self._remove_entry(key)
        
        return default
    
    def _remove_entry(self, key: str):
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
        """
        if key in self._meta['entries']:
            entry = self._meta['entries'][key]
            if os.path.exists(entry['path']):
                try:
                    os.remove(entry['path'])
                except OSError:
                    pass  # Ignorer les erreurs de suppression
            
            # Mettre à jour les méta-informations
            self._meta['total_size'] -= entry['size']
            del self._meta['entries'][key]
            self._save_meta()
    
    def _cleanup_if_needed(self):
        """
        Nettoie le cache si nécessaire pour respecter les limites de taille.
        Supprime d'abord les entrées expirées, puis les moins récemment utilisées.
        """
        # Convertir MB en octets pour la comparaison
        max_bytes = self.max_size_mb * 1024 * 1024
        
        if self._meta['total_size'] < max_bytes * 0.9:  # 90% de la capacité
            return  # Pas besoin de nettoyer
        
        current_time = time.time()
        
        # 1. Supprimer les entrées expirées
        expired_keys = [
            key for key, entry in self._meta['entries'].items()
            if current_time - entry['timestamp'] > self.expiration_seconds
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        # 2. Si toujours trop grand, supprimer les entrées les plus anciennes
        if self._meta['total_size'] >= max_bytes * 0.9:
            # Trier les entrées par timestamp
            sorted_entries = sorted(
                self._meta['entries'].items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Supprimer jusqu'à atteindre 70% de la capacité
            while self._meta['total_size'] > max_bytes * 0.7 and sorted_entries:
                key, _ = sorted_entries.pop(0)  # Le plus ancien
                self._remove_entry(key)
    
    def clear(self):
        """Vide complètement le cache."""
        for key in list(self._meta['entries'].keys()):
            self._remove_entry(key)
        
        # Recréer les méta-données
        self._meta = {
            'entries': {},
            'total_size': 0
        }
        self._save_meta()


# Décorateur pour mettre en cache les résultats de fonction
def cached_result(cache_manager: Optional[CacheManager] = None, 
                 key_prefix: str = '',
                 ignore_args: Optional[List[int]] = None,
                 ignore_kwargs: Optional[List[str]] = None,
                 expiration_seconds: Optional[int] = None):
    """
    Décorateur pour mettre en cache les résultats de fonction.
    
    Args:
        cache_manager: Gestionnaire de cache à utiliser.
            Si None, crée un gestionnaire par défaut.
        key_prefix: Préfixe pour les clés de cache
        ignore_args: Indices des arguments positionnels à ignorer pour le hachage
        ignore_kwargs: Noms des arguments nommés à ignorer pour le hachage
        expiration_seconds: Durée de validité en secondes (écrase celle du gestionnaire)
        
    Returns:
        Décorateur qui met en cache les résultats
    """
    # Créer un gestionnaire par défaut si nécessaire
    if cache_manager is None:
        cache_manager = CacheManager()
    
    # Initialiser les listes d'arguments à ignorer
    ignore_args = ignore_args or []
    ignore_kwargs = ignore_kwargs or []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Filtrer les arguments à ignorer pour le hachage
            cache_args = [
                arg for i, arg in enumerate(args)
                if i not in ignore_args
            ]
            
            cache_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ignore_kwargs
            }
            
            # Générer la clé de cache
            func_name = f"{func.__module__}.{func.__name__}"
            key = f"{key_prefix}{func_name}_{cache_manager._generate_key(*cache_args, **cache_kwargs)}"
            
            # Tenter de récupérer du cache
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value
            
            # Calculer la valeur
            result = func(*args, **kwargs)
            
            # Stocker dans le cache
            cache_manager.put(key, result)
            
            return result
        return wrapper
    return decorator


# Classes spécialisées pour différents types de caching

class MemoryCache:
    """
    Cache en mémoire pour les résultats intermédiaires fréquemment utilisés.
    Utilise une stratégie LRU (Least Recently Used) pour gérer la capacité.
    """
    
    def __init__(self, max_entries: int = 1000):
        """
        Initialise le cache mémoire.
        
        Args:
            max_entries: Nombre maximum d'entrées dans le cache
        """
        self.max_entries = max_entries
        self.cache = {}  # clé -> (valeur, timestamp)
        self.access_order = []  # Liste des clés par ordre d'accès (plus récent à la fin)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de la valeur
            default: Valeur par défaut si non trouvée
            
        Returns:
            Valeur mise en cache ou default si non trouvée
        """
        if key in self.cache:
            # Mettre à jour l'ordre d'accès
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return self.cache[key]
        return default
    
    def put(self, key: str, value: Any):
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé unique
            value: Valeur à stocker
        """
        # Faire de la place si nécessaire
        if key not in self.cache and len(self.cache) >= self.max_entries:
            # Supprimer l'entrée la moins récemment utilisée
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        # Stocker la valeur
        self.cache[key] = value
        
        # Mettre à jour l'ordre d'accès
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """Vide le cache."""
        self.cache.clear()
        self.access_order.clear()


class ResultMemoizer:
    """
    Mémoizeur optimisé pour les résultats coûteux à calculer.
    Combine cache mémoire et cache disque pour des performances optimales.
    """
    
    def __init__(self, name: str = 'default',
                disk_cache_size_mb: int = 512,
                memory_cache_entries: int = 500):
        """
        Initialise le mémoizeur.
        
        Args:
            name: Nom unique pour identifier ce mémoizeur
            disk_cache_size_mb: Taille maximale du cache disque en Mo
            memory_cache_entries: Nombre maximal d'entrées en cache mémoire
        """
        self.name = name
        self.disk_cache = CacheManager(
            cache_dir=os.path.join(os.path.expanduser('~'), '.xplia', 'cache', name),
            max_size_mb=disk_cache_size_mb
        )
        self.memory_cache = MemoryCache(max_entries=memory_cache_entries)
    
    def memoize(self, key_prefix: str = '',
               ignore_args: Optional[List[int]] = None,
               ignore_kwargs: Optional[List[str]] = None):
        """
        Crée un décorateur pour mémoizer une fonction.
        
        Args:
            key_prefix: Préfixe pour les clés de cache
            ignore_args: Indices des arguments positionnels à ignorer
            ignore_kwargs: Noms des arguments nommés à ignorer
            
        Returns:
            Décorateur de mémoization
        """
        ignore_args = ignore_args or []
        ignore_kwargs = ignore_kwargs or []
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Filtrer les arguments à ignorer
                cache_args = [
                    arg for i, arg in enumerate(args)
                    if i not in ignore_args
                ]
                
                cache_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ignore_kwargs
                }
                
                # Générer la clé
                func_name = f"{func.__module__}.{func.__name__}"
                key = f"{key_prefix}{func_name}_{self.disk_cache._generate_key(*cache_args, **cache_kwargs)}"
                
                # 1. Vérifier d'abord le cache mémoire (plus rapide)
                result = self.memory_cache.get(key)
                if result is not None:
                    return result
                
                # 2. Sinon, vérifier le cache disque
                result = self.disk_cache.get(key)
                if result is not None:
                    # Ajouter au cache mémoire pour les prochains accès
                    self.memory_cache.put(key, result)
                    return result
                
                # 3. Calculer la valeur si non trouvée
                result = func(*args, **kwargs)
                
                # 4. Mettre en cache le résultat
                self.memory_cache.put(key, result)
                self.disk_cache.put(key, result)
                
                return result
            return wrapper
        return decorator


# Singleton global pour faciliter l'accès aux mémoizeurs
class Memoizers:
    """Gestionnaire global des mémoizeurs."""
    
    _instances = {}
    
    @classmethod
    def get(cls, name: str = 'default') -> ResultMemoizer:
        """
        Obtient un mémoizeur par son nom.
        
        Args:
            name: Nom du mémoizeur
            
        Returns:
            Instance de ResultMemoizer
        """
        if name not in cls._instances:
            cls._instances[name] = ResultMemoizer(name=name)
        return cls._instances[name]
    
    @classmethod
    def clear_all(cls):
        """Vide tous les mémoizeurs."""
        for memoizer in cls._instances.values():
            memoizer.memory_cache.clear()
            memoizer.disk_cache.clear()
        cls._instances.clear()
