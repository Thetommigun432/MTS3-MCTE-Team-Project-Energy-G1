"""
Cache 24h Package - Gestione cache Redis per dati energetici.

Componenti:
    - CacheWorker: Lavoratore unico che processa e mantiene i dati in Redis
    - CacheManager: Interfaccia di lettura per l'API Flask
"""

from .cache_manager import CacheManager, get_cache_manager
from .cache_worker import CacheWorker

__all__ = ['CacheManager', 'CacheWorker', 'get_cache_manager']
