"""
Module d'optimisation mémoire pour grands ensembles de données.

Ce module fournit des utilitaires pour optimiser l'utilisation de la mémoire
lors du traitement de grands ensembles de données dans la bibliothèque XPLIA.
"""

import os
import gc
import logging
import numpy as np
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator, Generator
import psutil


logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Gestionnaire d'optimisation mémoire.
    
    Cette classe fournit des méthodes pour optimiser l'utilisation de la mémoire
    lors du traitement de grands ensembles de données.
    """
    
    def __init__(self, threshold_mb: int = 1024, aggressive: bool = False):
        """
        Initialise le gestionnaire d'optimisation mémoire.
        
        Args:
            threshold_mb: Seuil en Mo à partir duquel déclencher des optimisations
            aggressive: Si True, utilise des optimisations plus agressives
        """
        self.threshold_mb = threshold_mb
        self.aggressive = aggressive
        self._process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """
        Obtient l'utilisation mémoire actuelle du processus.
        
        Returns:
            Utilisation mémoire en Mo
        """
        try:
            # En Mo
            return self._process.memory_info().rss / (1024 * 1024)
        except:
            # Fallback si psutil échoue
            return 0.0
    
    def check_memory_threshold(self) -> bool:
        """
        Vérifie si l'utilisation mémoire dépasse le seuil.
        
        Returns:
            True si le seuil est dépassé
        """
        return self.get_memory_usage() > self.threshold_mb
    
    def optimize_if_needed(self):
        """
        Déclenche des optimisations mémoire si nécessaire.
        """
        if self.check_memory_threshold():
            self.optimize()
    
    def optimize(self):
        """
        Déclenche une optimisation mémoire immédiate.
        """
        gc.collect()
        
        if self.aggressive:
            # Optimisations plus agressives en mode agressif
            if hasattr(gc, 'set_threshold'):  # Python 3+
                # Rendre la collecte plus agressive
                old_threshold = gc.get_threshold()
                gc.set_threshold(old_threshold[0] // 2,
                               old_threshold[1] // 2,
                               old_threshold[2] // 2)
                
                # Exécuter deux cycles de collecte
                gc.collect()
                gc.collect()
                
                # Restaurer les seuils précédents
                gc.set_threshold(*old_threshold)


class ChunkedDataIterator:
    """
    Itérateur sur de grands ensembles de données par chunks.
    
    Permet de traiter de grands ensembles de données sans les charger
    entièrement en mémoire, en les divisant en chunks gérables.
    """
    
    def __init__(self, data: Any, chunk_size: int = 1000):
        """
        Initialise l'itérateur par chunks.
        
        Args:
            data: Données à itérer (doit supporter l'indexation et len)
            chunk_size: Taille de chaque chunk
        """
        self.data = data
        self.chunk_size = chunk_size
        self.total_size = len(data)
        self.current_pos = 0
    
    def __iter__(self) -> 'ChunkedDataIterator':
        """
        Permet l'utilisation dans une boucle for.
        
        Returns:
            L'instance de l'itérateur
        """
        self.current_pos = 0
        return self
    
    def __next__(self) -> Any:
        """
        Retourne le prochain chunk.
        
        Returns:
            Prochain chunk de données
            
        Raises:
            StopIteration: Quand toutes les données ont été parcourues
        """
        if self.current_pos >= self.total_size:
            raise StopIteration
        
        end_pos = min(self.current_pos + self.chunk_size, self.total_size)
        chunk = self.data[self.current_pos:end_pos]
        self.current_pos = end_pos
        
        return chunk


def process_in_chunks(data: Any, process_func: Callable,
                     chunk_size: int = 1000, 
                     combine_results: bool = True) -> Any:
    """
    Traite de grands ensembles de données en chunks pour optimiser la mémoire.
    
    Args:
        data: Données à traiter (doit supporter l'indexation et len)
        process_func: Fonction à appliquer à chaque chunk
        chunk_size: Taille de chaque chunk
        combine_results: Si True, combine les résultats des chunks
        
    Returns:
        Résultats combinés ou liste de résultats par chunk
    """
    iterator = ChunkedDataIterator(data, chunk_size=chunk_size)
    results = []
    
    for chunk in iterator:
        # Appliquer la fonction de traitement au chunk
        chunk_result = process_func(chunk)
        results.append(chunk_result)
        
        # Optimiser la mémoire entre les chunks
        optimizer = MemoryOptimizer()
        optimizer.optimize_if_needed()
    
    if combine_results:
        # Tenter de combiner les résultats
        if results and isinstance(results[0], (list, tuple)):
            # Pour les listes/tuples, concaténer
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        elif results and isinstance(results[0], np.ndarray):
            # Pour les arrays numpy, utiliser vstack/hstack selon la dimension
            try:
                return np.vstack(results)
            except:
                try:
                    return np.hstack(results)
                except:
                    pass  # Fallback to returning list of results
        elif results and isinstance(results[0], dict):
            # Pour les dictionnaires, fusionner les clés
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        
    # Par défaut, retourner la liste des résultats par chunk
    return results


class LargeArrayHandler:
    """
    Gestionnaire pour manipuler efficacement de grands tableaux.
    
    Fournit des méthodes pour traiter de grands tableaux numpy
    de manière mémoire-efficiente.
    """
    
    @staticmethod
    def create_memory_mapped(shape: Tuple[int, ...], 
                            dtype: np.dtype = np.float32,
                            filename: Optional[str] = None) -> np.ndarray:
        """
        Crée un tableau memory-mapped pour économiser de la RAM.
        
        Args:
            shape: Forme du tableau
            dtype: Type de données
            filename: Fichier pour le mapping (temporaire si None)
            
        Returns:
            Tableau memory-mapped
        """
        import tempfile
        
        if filename is None:
            # Créer un fichier temporaire
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = temp_file.name
            temp_file.close()
        
        # Créer le tableau memory-mapped
        return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    
    @staticmethod
    def compute_in_blocks(array: np.ndarray, 
                         func: Callable,
                         block_size: int = 1000,
                         axis: int = 0) -> np.ndarray:
        """
        Applique une fonction sur un grand tableau par blocs.
        
        Args:
            array: Tableau à traiter
            func: Fonction à appliquer
            block_size: Taille de chaque bloc
            axis: Axe selon lequel diviser le tableau
            
        Returns:
            Tableau des résultats
        """
        size = array.shape[axis]
        results = []
        
        for i in range(0, size, block_size):
            # Créer l'index pour sélectionner le bloc
            end = min(i + block_size, size)
            indices = [slice(None)] * array.ndim
            indices[axis] = slice(i, end)
            
            # Extraire et traiter le bloc
            block = array[tuple(indices)]
            block_result = func(block)
            results.append(block_result)
            
            # Libérer la mémoire
            gc.collect()
        
        # Combine results based on their type
        if isinstance(results[0], np.ndarray):
            try:
                if axis == 0:
                    return np.vstack(results)
                else:
                    return np.concatenate(results, axis=axis)
            except ValueError:
                # Fallback if shapes are incompatible
                return np.array(results, dtype=object)
        else:
            return results
    
    @staticmethod
    def optimize_dtype(array: np.ndarray) -> np.ndarray:
        """
        Optimise le type de données d'un tableau pour économiser de la mémoire.
        
        Args:
            array: Tableau à optimiser
            
        Returns:
            Tableau avec type de données optimisé
        """
        # Déterminer si c'est un tableau d'entiers ou de flottants
        if np.issubdtype(array.dtype, np.integer):
            # Pour les entiers, trouver la précision minimale nécessaire
            info = np.iinfo(array.dtype)
            min_val = array.min()
            max_val = array.max()
            
            # Choisir le type le plus compact qui peut représenter les données
            if min_val >= 0:
                if max_val <= 255:
                    return array.astype(np.uint8)
                elif max_val <= 65535:
                    return array.astype(np.uint16)
                elif max_val <= 4294967295:
                    return array.astype(np.uint32)
            else:
                if min_val >= -128 and max_val <= 127:
                    return array.astype(np.int8)
                elif min_val >= -32768 and max_val <= 32767:
                    return array.astype(np.int16)
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return array.astype(np.int32)
        
        elif np.issubdtype(array.dtype, np.floating):
            # Pour les flottants, voir si float16 ou float32 suffit
            # Attention: float16 peut causer une perte de précision significative
            if array.dtype == np.float64:
                # Tester si float32 suffit (pour la plupart des cas)
                float32_array = array.astype(np.float32)
                if np.allclose(array, float32_array, rtol=1e-5, atol=1e-8):
                    return float32_array
        
        # Retourner l'array original si aucune optimisation n'est possible
        return array
    
    @staticmethod
    def generator_from_array(array: np.ndarray, 
                           batch_size: int = 32,
                           shuffle: bool = False) -> Generator:
        """
        Crée un générateur qui produit des batches à partir d'un grand tableau.
        Utile pour l'entraînement de modèles ML sans charger tout le dataset.
        
        Args:
            array: Tableau source
            batch_size: Taille de chaque batch
            shuffle: Si True, mélange les indices
            
        Yields:
            Batches successifs
        """
        indices = np.arange(len(array))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(array), batch_size):
            end_idx = min(start_idx + batch_size, len(array))
            batch_indices = indices[start_idx:end_idx]
            yield array[batch_indices]


# Décorateur pour des fonctions mémoire-efficientes
def memory_efficient(chunk_size: int = 1000, 
                    optimize_result: bool = True,
                    monitor_threshold_mb: Optional[int] = None):
    """
    Décorateur pour rendre une fonction mémoire-efficiente.
    
    Si la fonction prend une liste/tableau en premier argument,
    divise l'entrée en chunks et applique la fonction à chaque chunk.
    
    Args:
        chunk_size: Taille des chunks pour le traitement
        optimize_result: Si True, optimise le type des résultats numpy
        monitor_threshold_mb: Seuil en Mo pour surveillance mémoire
        
    Returns:
        Fonction mémoire-efficiente
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Vérifier si le premier argument est une collection divisible
            if args and hasattr(args[0], '__len__') and len(args) > 0:
                data = args[0]
                # Vérifier si la taille justifie le traitement en chunks
                if len(data) > chunk_size:
                    # Extraire les autres arguments
                    other_args = args[1:]
                    
                    # Créer la fonction de traitement de chunk
                    def process_chunk(chunk):
                        return func(chunk, *other_args, **kwargs)
                    
                    # Traiter en chunks
                    result = process_in_chunks(
                        data=data,
                        process_func=process_chunk,
                        chunk_size=chunk_size,
                        combine_results=True
                    )
                    
                    # Optimiser le résultat si demandé et si c'est un array numpy
                    if optimize_result and isinstance(result, np.ndarray):
                        result = LargeArrayHandler.optimize_dtype(result)
                    
                    return result
            
            # Si les conditions ne sont pas remplies, exécuter normalement
            result = func(*args, **kwargs)
            
            # Surveiller l'utilisation mémoire si un seuil est défini
            if monitor_threshold_mb is not None:
                optimizer = MemoryOptimizer(threshold_mb=monitor_threshold_mb)
                optimizer.optimize_if_needed()
            
            return result
        return wrapper
    return decorator


# Fonction utilitaire pour diviser un dataframe pandas en chunks
def chunked_dataframe_iterator(df, chunk_size: int = 1000) -> Iterator:
    """
    Crée un itérateur qui découpe un DataFrame pandas en chunks.
    
    Args:
        df: DataFrame pandas à découper
        chunk_size: Nombre de lignes par chunk
        
    Yields:
        Chunks successifs du DataFrame
    """
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        yield df.iloc[start_idx:end_idx]


# Fonction utilitaire pour réduire l'empreinte mémoire des explications
def optimize_explanations(explanations: List[Dict]) -> List[Dict]:
    """
    Optimise l'empreinte mémoire d'une liste d'explications.
    
    Args:
        explanations: Liste de dictionnaires d'explication
        
    Returns:
        Liste d'explications optimisée
    """
    # Optimisations possibles:
    # 1. Utiliser des types de données compacts
    # 2. Éliminer les informations redondantes
    # 3. Compresser les valeurs numériques
    
    optimized = []
    
    for exp in explanations:
        # Créer une copie optimisée de chaque explication
        opt_exp = {}
        
        for key, value in exp.items():
            # Convertir les numpy float64/int64 en types Python natifs
            if isinstance(value, np.floating):
                opt_exp[key] = float(value)
            elif isinstance(value, np.integer):
                opt_exp[key] = int(value)
            elif isinstance(value, np.ndarray):
                # Pour les tableaux numpy, optimiser le type
                opt_exp[key] = LargeArrayHandler.optimize_dtype(value).tolist()
            else:
                opt_exp[key] = value
        
        optimized.append(opt_exp)
    
    return optimized
