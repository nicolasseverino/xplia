"""
Module d'exécution parallèle pour optimiser les performances.

Ce module fournit des utilitaires pour paralléliser les calculs intensifs 
dans la bibliothèque XPLIA, adaptés aux différents contextes d'utilisation.
"""

import os
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
import numpy as np
from typing import Callable, List, Dict, Any, Union, Optional, Tuple


logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Gestionnaire d'exécution parallèle adaptative.
    
    Cette classe fournit des méthodes pour exécuter des tâches en parallèle,
    en s'adaptant automatiquement à la charge de travail et aux ressources disponibles.
    """
    
    def __init__(self, max_workers: Optional[int] = None, 
                 use_processes: bool = True, 
                 adaptive: bool = True):
        """
        Initialise le gestionnaire d'exécution parallèle.
        
        Args:
            max_workers: Nombre maximum de workers (threads/processus).
                Si None, utilisera le nombre de CPU disponibles.
            use_processes: Si True, utilise des processus (multiprocessing).
                Si False, utilise des threads (multithreading).
            adaptive: Si True, adapte automatiquement le nombre de workers
                en fonction de la charge de travail.
        """
        # Paramètres de configuration
        self.use_processes = use_processes
        self.adaptive = adaptive
        
        # Déterminer le nombre optimal de workers
        if max_workers is None:
            # Utiliser le nombre de CPU disponibles par défaut
            self.max_workers = os.cpu_count() or 4
        else:
            self.max_workers = max_workers
        
        # Création de l'executor approprié
        self._executor = None
        self._active = False
    
    def start(self):
        """Démarre l'executor s'il n'est pas déjà actif."""
        if not self._active:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._active = True
    
    def stop(self):
        """Arrête l'executor s'il est actif."""
        if self._active and self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._active = False
    
    def __enter__(self):
        """Support pour le context manager (with statement)."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Gestion de la sortie du context manager."""
        self.stop()
    
    def map(self, func: Callable, *iterables: List, 
            chunk_size: Optional[int] = None, 
            timeout: Optional[float] = None) -> List:
        """
        Applique une fonction à chaque élément des iterables en parallèle.
        
        Args:
            func: La fonction à appliquer
            *iterables: Les séquences d'arguments à passer à la fonction
            chunk_size: Taille des chunks pour traitement batch (optimisation)
            timeout: Délai maximum d'exécution en secondes
            
        Returns:
            Liste des résultats dans l'ordre des entrées
        """
        if not self._active:
            self.start()
        
        # Déterminer la taille de chunk optimale si non spécifiée
        if chunk_size is None and self.adaptive:
            # Estimer une taille de chunk basée sur le nombre d'items et de workers
            total_items = min(len(iterable) for iterable in iterables)
            chunk_size = max(1, total_items // (self.max_workers * 4))
        
        # Exécuter le mapping parallèle
        return list(self._executor.map(func, *iterables, 
                                      chunksize=chunk_size or 1,
                                      timeout=timeout))
    
    def submit_all(self, tasks: List[Tuple[Callable, List, Dict]]) -> List:
        """
        Soumet un lot de tâches pour exécution parallèle.
        
        Args:
            tasks: Liste de tuples (fonction, args, kwargs)
            
        Returns:
            Liste des résultats dans l'ordre de soumission
        """
        if not self._active:
            self.start()
        
        # Soumettre les tâches
        futures = []
        for func, args, kwargs in tasks:
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            futures.append(self._executor.submit(func, *args, **kwargs))
        
        # Récupérer les résultats dans l'ordre
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")
                results.append(None)
        
        return results
    
    def execute(self, func: Callable, items: List, 
                *args, **kwargs) -> List:
        """
        Exécute une fonction sur une liste d'éléments en parallèle.
        
        Args:
            func: Fonction à exécuter pour chaque élément
            items: Liste des éléments à traiter
            *args, **kwargs: Arguments additionnels à passer à la fonction
            
        Returns:
            Liste des résultats dans l'ordre des éléments
        """
        if not self._active:
            self.start()
        
        # Helper function to apply additional args/kwargs
        def _wrapper(item):
            return func(item, *args, **kwargs)
        
        # Exécuter en parallèle
        return self.map(_wrapper, items)


# Décorateur pour paralléliser une fonction
def parallelize(max_workers: Optional[int] = None, 
                use_processes: bool = True,
                chunk_size: Optional[int] = None):
    """
    Décorateur pour paralléliser automatiquement une fonction sur des collections.
    
    Args:
        max_workers: Nombre maximum de workers parallèles
        use_processes: Si True, utilise des processus (multiprocessing)
        chunk_size: Taille des chunks pour le traitement par lots
        
    Returns:
        Fonction décorée avec parallélisation automatique
    """
    def decorator(func):
        @wraps(func)
        def wrapper(collection, *args, **kwargs):
            # Ne paralléliser que si la collection est suffisamment grande
            if hasattr(collection, '__len__') and len(collection) > 1:
                with ParallelExecutor(max_workers=max_workers, 
                                     use_processes=use_processes) as executor:
                    
                    # Pour les objets de type array/list, appliquer la fonction à chaque élément
                    if isinstance(collection, (list, tuple, np.ndarray)):
                        results = executor.map(
                            lambda x: func([x], *args, **kwargs), 
                            collection,
                            chunk_size=chunk_size
                        )
                        # Combiner les résultats en fonction du type de retour
                        if results and isinstance(results[0], (list, tuple)):
                            # Aplatir les résultats
                            flat_results = []
                            for r in results:
                                flat_results.extend(r)
                            return flat_results
                        return results
                    
                    # Pour d'autres types de collections, appeler directement
                    return func(collection, *args, **kwargs)
            else:
                # Collection trop petite, exécuter normalement
                return func(collection, *args, **kwargs)
        return wrapper
    return decorator


# Utilitaire pour déterminer automatiquement le meilleur mode d'exécution
def auto_parallelize(collection, func: Callable, 
                     threshold: int = 100,
                     *args, **kwargs) -> List:
    """
    Parallélise automatiquement une fonction en fonction de la taille de la collection.
    
    Args:
        collection: Collection d'éléments à traiter
        func: Fonction à appliquer à chaque élément
        threshold: Seuil à partir duquel paralléliser
        *args, **kwargs: Arguments additionnels pour la fonction
        
    Returns:
        Résultats de l'exécution
    """
    # Vérifier si la parallélisation est pertinente
    if hasattr(collection, '__len__') and len(collection) >= threshold:
        # Déterminer le mode optimal (processes vs threads)
        # Les calculs intensifs bénéficient des processus
        # Les opérations I/O bénéficient des threads
        cpu_bound = kwargs.pop('cpu_bound', True)
        
        with ParallelExecutor(use_processes=cpu_bound) as executor:
            return executor.execute(func, collection, *args, **kwargs)
    else:
        # Exécution séquentielle pour les petites collections
        return [func(item, *args, **kwargs) for item in collection]


# Classe utilitaire pour l'estimation des performances
class PerformanceTracker:
    """
    Utilitaire pour suivre et estimer les performances d'exécution.
    
    Permet d'adapter dynamiquement les stratégies de parallélisation
    en fonction des performances mesurées.
    """
    
    def __init__(self):
        """Initialise le tracker de performances."""
        self.execution_times = {}
        self.count = {}
    
    def time_execution(self, func_name: str) -> Callable:
        """
        Décorateur pour mesurer le temps d'exécution d'une fonction.
        
        Args:
            func_name: Nom de la fonction pour l'enregistrement
            
        Returns:
            Décorateur qui mesure le temps d'exécution
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Enregistrer le temps d'exécution
                if func_name not in self.execution_times:
                    self.execution_times[func_name] = elapsed
                    self.count[func_name] = 1
                else:
                    # Moyenne mobile
                    self.execution_times[func_name] = (
                        self.execution_times[func_name] * 0.7 + elapsed * 0.3
                    )
                    self.count[func_name] += 1
                
                return result
            return wrapper
        return decorator
    
    def get_average_time(self, func_name: str) -> float:
        """
        Obtient le temps d'exécution moyen pour une fonction.
        
        Args:
            func_name: Nom de la fonction
            
        Returns:
            Temps d'exécution moyen en secondes
        """
        if func_name in self.execution_times:
            return self.execution_times[func_name]
        return 0.0
    
    def recommend_parallelization(self, func_name: str, 
                                 collection_size: int) -> bool:
        """
        Recommande si une fonction doit être parallélisée.
        
        Args:
            func_name: Nom de la fonction
            collection_size: Taille de la collection à traiter
            
        Returns:
            True si la parallélisation est recommandée
        """
        if func_name not in self.execution_times:
            # Par défaut, paralléliser les grandes collections
            return collection_size > 100
        
        avg_time = self.execution_times[func_name]
        
        # Paralléliser si:
        # - La fonction prend plus de 0.1s par appel en moyenne
        # - ET la collection est suffisamment grande
        return avg_time > 0.1 and collection_size > 50
