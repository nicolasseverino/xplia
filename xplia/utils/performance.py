"""
Outils de mesure de performance
================================

Utilitaires pour mesurer le temps d'exécution et l'utilisation mémoire.
"""

import time
import psutil
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Gestionnaire de contexte pour mesurer le temps d'exécution."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialise le timer.
        
        Args:
            name: Nom de l'opération à mesurer
            verbose: Si True, affiche le temps écoulé
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Démarre le timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Arrête le timer et affiche le résultat."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            logger.info(f"{self.name} took {self.elapsed:.4f} seconds")
    
    def get_elapsed(self) -> Optional[float]:
        """Retourne le temps écoulé en secondes."""
        return self.elapsed


class MemoryTracker:
    """Gestionnaire de contexte pour suivre l'utilisation mémoire."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialise le tracker mémoire.
        
        Args:
            name: Nom de l'opération à suivre
            verbose: Si True, affiche l'utilisation mémoire
        """
        self.name = name
        self.verbose = verbose
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.end_memory = None
        self.memory_used = None
    
    def __enter__(self):
        """Démarre le suivi mémoire."""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, *args):
        """Arrête le suivi et affiche le résultat."""
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_used = self.end_memory - self.start_memory
        if self.verbose:
            logger.info(f"{self.name} used {self.memory_used:.2f} MB of memory")
    
    def get_memory_used(self) -> Optional[float]:
        """Retourne la mémoire utilisée en MB."""
        return self.memory_used


@contextmanager
def measure_performance(name: str = "Operation", track_memory: bool = True):
    """
    Gestionnaire de contexte combiné pour mesurer temps et mémoire.
    
    Args:
        name: Nom de l'opération
        track_memory: Si True, suit aussi l'utilisation mémoire
        
    Yields:
        Dict contenant les métriques de performance
    """
    metrics = {}
    
    timer = Timer(name, verbose=False)
    timer.__enter__()
    
    if track_memory:
        mem_tracker = MemoryTracker(name, verbose=False)
        mem_tracker.__enter__()
    
    try:
        yield metrics
    finally:
        timer.__exit__()
        metrics['elapsed_time'] = timer.elapsed
        
        if track_memory:
            mem_tracker.__exit__()
            metrics['memory_used'] = mem_tracker.memory_used
        
        logger.info(f"{name} - Time: {metrics.get('elapsed_time', 0):.4f}s, "
                   f"Memory: {metrics.get('memory_used', 0):.2f}MB")
