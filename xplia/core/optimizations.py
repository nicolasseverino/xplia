"""
Module d'optimisation unifié pour XPLIA.

Ce module centralise les fonctionnalités d'optimisation de performances pour la 
bibliothèque XPLIA, offrant une interface simplifiée pour les optimisations de
parallélisation, mise en cache, et gestion mémoire.
"""

import logging
import numpy as np
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import des modules spécialisés
from xplia.core.performance.parallel_executor import (
    ParallelExecutor, parallelize, auto_parallelize
)
from xplia.core.performance.cache_manager import (
    CacheManager, cached_result, ResultMemoizer, Memoizers
)
from xplia.core.performance.memory_optimizer import (
    MemoryOptimizer, process_in_chunks, memory_efficient, optimize_explanations
)


logger = logging.getLogger(__name__)


class XPLIAOptimizer:
    """
    Classe principale d'optimisation qui combine toutes les stratégies
    pour améliorer les performances des explainers et modules de conformité.
    """
    
    def __init__(self, 
                 enable_parallelization: bool = True, 
                 enable_caching: bool = True,
                 enable_memory_optimization: bool = True,
                 max_workers: Optional[int] = None,
                 cache_name: str = 'xplia',
                 memory_threshold_mb: int = 1024):
        """
        Initialise l'optimiseur XPLIA.
        
        Args:
            enable_parallelization: Activer la parallélisation automatique
            enable_caching: Activer la mise en cache des résultats
            enable_memory_optimization: Activer l'optimisation mémoire
            max_workers: Nombre maximum de workers pour parallélisation
            cache_name: Nom du cache à utiliser
            memory_threshold_mb: Seuil en Mo pour les optimisations mémoire
        """
        self.enable_parallelization = enable_parallelization
        self.enable_caching = enable_caching
        self.enable_memory_optimization = enable_memory_optimization
        
        # Initialiser les composants selon les flags
        self.parallel_executor = None
        if enable_parallelization:
            self.parallel_executor = ParallelExecutor(max_workers=max_workers)
        
        self.memoizer = None
        if enable_caching:
            self.memoizer = Memoizers.get(cache_name)
        
        self.memory_optimizer = None
        if enable_memory_optimization:
            self.memory_optimizer = MemoryOptimizer(threshold_mb=memory_threshold_mb)
    
    def optimize(self, func: Callable) -> Callable:
        """
        Décorateur principal qui applique toutes les optimisations activées.
        
        Args:
            func: Fonction à optimiser
            
        Returns:
            Fonction optimisée avec parallélisation, cache et gestion mémoire
        """
        # Appliquer les optimisations dans l'ordre: mémoire, cache, parallélisation
        optimized_func = func
        
        if self.enable_memory_optimization:
            optimized_func = memory_efficient()(optimized_func)
        
        if self.enable_caching:
            optimized_func = self.memoizer.memoize()(optimized_func)
        
        if self.enable_parallelization:
            optimized_func = parallelize()(optimized_func)
        
        return optimized_func
    
    def parallel_map(self, func: Callable, items: List, *args, **kwargs) -> List:
        """
        Applique une fonction à une liste d'éléments en parallèle.
        
        Args:
            func: Fonction à appliquer
            items: Liste d'éléments
            *args, **kwargs: Arguments additionnels pour func
            
        Returns:
            Liste des résultats
        """
        if not self.enable_parallelization or len(items) <= 1:
            return [func(item, *args, **kwargs) for item in items]
        
        if self.parallel_executor is None:
            self.parallel_executor = ParallelExecutor()
        
        with self.parallel_executor as executor:
            return executor.execute(func, items, *args, **kwargs)
    
    def chunked_processing(self, func: Callable, data: Any, 
                         chunk_size: int = 1000, *args, **kwargs) -> Any:
        """
        Traite de grandes données par chunks pour optimiser la mémoire.
        
        Args:
            func: Fonction à appliquer
            data: Données à traiter
            chunk_size: Taille des chunks
            *args, **kwargs: Arguments additionnels pour func
            
        Returns:
            Résultat combiné
        """
        if not self.enable_memory_optimization or len(data) <= chunk_size:
            return func(data, *args, **kwargs)
        
        # Helper function pour traiter un chunk
        def process_chunk(chunk):
            return func(chunk, *args, **kwargs)
        
        return process_in_chunks(
            data=data,
            process_func=process_chunk,
            chunk_size=chunk_size,
            combine_results=True
        )
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Appelle une fonction avec mise en cache du résultat.
        
        Args:
            func: Fonction à appeler
            *args, **kwargs: Arguments pour la fonction
            
        Returns:
            Résultat de la fonction (depuis le cache si disponible)
        """
        if not self.enable_caching:
            return func(*args, **kwargs)
        
        if self.memoizer is None:
            self.memoizer = Memoizers.get('xplia')
        
        # Générer une clé basée sur la fonction et ses arguments
        cache_key = f"{func.__module__}.{func.__name__}"
        
        # Vérifier le cache
        result = self.memoizer.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Vérifier le cache disque
        result = self.memoizer.disk_cache.get(cache_key)
        if result is not None:
            self.memoizer.memory_cache.put(cache_key, result)
            return result
        
        # Calculer et mettre en cache
        result = func(*args, **kwargs)
        self.memoizer.memory_cache.put(cache_key, result)
        self.memoizer.disk_cache.put(cache_key, result)
        
        return result
    
    def optimize_memory(self):
        """
        Déclenche une optimisation mémoire immédiate.
        """
        if self.enable_memory_optimization and self.memory_optimizer:
            self.memory_optimizer.optimize()


# Instance globale de l'optimiseur avec paramètres par défaut
optimizer = XPLIAOptimizer()


# Fonctions d'accès global pour simplifier l'utilisation

def optimize(func: Optional[Callable] = None, 
            enable_parallelization: bool = True,
            enable_caching: bool = True,
            enable_memory_optimization: bool = True) -> Callable:
    """
    Décorateur pour appliquer toutes les optimisations à une fonction.
    
    Usage:
        @optimize
        def my_func(data):
            # Traitement
    
    Args:
        func: Fonction à optimiser
        enable_parallelization: Activer la parallélisation
        enable_caching: Activer le cache
        enable_memory_optimization: Activer l'optimisation mémoire
        
    Returns:
        Fonction optimisée
    """
    # Support pour @optimize et @optimize(...)
    if func is None:
        # Appel avec paramètres: @optimize(params)
        def decorator(f):
            # Créer un optimiseur temporaire avec les paramètres spécifiés
            temp_optimizer = XPLIAOptimizer(
                enable_parallelization=enable_parallelization,
                enable_caching=enable_caching,
                enable_memory_optimization=enable_memory_optimization
            )
            return temp_optimizer.optimize(f)
        return decorator
    else:
        # Appel sans paramètres: @optimize
        return optimizer.optimize(func)


def parallel_map(func: Callable, items: List, *args, **kwargs) -> List:
    """
    Applique une fonction à une liste d'éléments en parallèle.
    
    Args:
        func: Fonction à appliquer
        items: Liste d'éléments
        *args, **kwargs: Arguments additionnels pour func
        
    Returns:
        Liste des résultats
    """
    return optimizer.parallel_map(func, items, *args, **kwargs)


def chunked_processing(func: Callable, data: Any, 
                     chunk_size: int = 1000, *args, **kwargs) -> Any:
    """
    Traite de grandes données par chunks pour optimiser la mémoire.
    
    Args:
        func: Fonction à appliquer
        data: Données à traiter
        chunk_size: Taille des chunks
        *args, **kwargs: Arguments additionnels pour func
        
    Returns:
        Résultat combiné
    """
    return optimizer.chunked_processing(func, data, chunk_size, *args, **kwargs)


def cached_call(func: Callable, *args, **kwargs) -> Any:
    """
    Appelle une fonction avec mise en cache du résultat.
    
    Args:
        func: Fonction à appeler
        *args, **kwargs: Arguments pour la fonction
        
    Returns:
        Résultat de la fonction (depuis le cache si disponible)
    """
    return optimizer.cached_call(func, *args, **kwargs)


def optimize_memory():
    """
    Déclenche une optimisation mémoire immédiate.
    """
    optimizer.optimize_memory()


# Configuration globale
def configure(enable_parallelization: bool = True, 
             enable_caching: bool = True,
             enable_memory_optimization: bool = True,
             max_workers: Optional[int] = None,
             cache_name: str = 'xplia',
             memory_threshold_mb: int = 1024):
    """
    Configure les paramètres d'optimisation globaux.
    
    Args:
        enable_parallelization: Activer la parallélisation
        enable_caching: Activer le cache
        enable_memory_optimization: Activer l'optimisation mémoire
        max_workers: Nombre maximum de workers
        cache_name: Nom du cache
        memory_threshold_mb: Seuil mémoire
    """
    global optimizer
    optimizer = XPLIAOptimizer(
        enable_parallelization=enable_parallelization,
        enable_caching=enable_caching,
        enable_memory_optimization=enable_memory_optimization,
        max_workers=max_workers,
        cache_name=cache_name,
        memory_threshold_mb=memory_threshold_mb
    )


def get_optimizer() -> XPLIAOptimizer:
    """
    Obtient l'instance globale de l'optimiseur.
    
    Returns:
        Instance de XPLIAOptimizer
    """
    return optimizer
