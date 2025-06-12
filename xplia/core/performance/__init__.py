"""
Module d'optimisation des performances pour XPLIA.

Ce module fournit des outils pour optimiser les performances de la bibliothèque XPLIA
en termes de temps d'exécution, d'utilisation de la mémoire et de mise en cache
pour les explications et analyses de conformité.
"""

from .parallel_executor import (
    ParallelExecutor, 
    parallelize,
    auto_parallelize,
    PerformanceTracker
)

from .cache_manager import (
    CacheManager,
    cached_result,
    MemoryCache,
    ResultMemoizer,
    Memoizers
)

from .memory_optimizer import (
    MemoryOptimizer,
    ChunkedDataIterator,
    process_in_chunks,
    LargeArrayHandler,
    memory_efficient,
    chunked_dataframe_iterator,
    optimize_explanations
)

# Exporter les principales interfaces d'optimisation
__all__ = [
    # Parallélisation
    'ParallelExecutor',
    'parallelize',
    'auto_parallelize',
    'PerformanceTracker',
    
    # Cache
    'CacheManager',
    'cached_result',
    'MemoryCache',
    'ResultMemoizer',
    'Memoizers',
    
    # Optimisation mémoire
    'MemoryOptimizer',
    'ChunkedDataIterator',
    'process_in_chunks',
    'LargeArrayHandler',
    'memory_efficient',
    'chunked_dataframe_iterator',
    'optimize_explanations'
]
