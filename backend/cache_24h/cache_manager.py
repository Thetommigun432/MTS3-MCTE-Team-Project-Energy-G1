"""
Cache Manager - Gestisce l'accesso in lettura alla cache Redis.

Fornisce un'interfaccia pulita per l'API Flask per accedere ai dati
processati dal Cache Worker senza duplicare la logica.
"""

import redis
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any

# Configurazione Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Redis Keys (condivise con cache_worker.py)
REDIS_KEYS = {
    "stream": "energy:stream",           # Lista dei dati real-time (ultimi 24h)
    "cleaned": "energy:cleaned",          # Dati puliti pronti per il modello
    "predictions": "energy:predictions",  # Ultime predizioni
    "metadata": "energy:metadata",        # Metadati (ultimo aggiornamento, stato, ecc.)
    "lock": "energy:worker_lock"          # Lock del worker
}


class CacheManager:
    """
    Manager per l'accesso ai dati in cache Redis.
    
    Uso tipico:
        cache = CacheManager()
        predictions = cache.get_predictions()
        stream = cache.get_stream(limit=100)
    """
    
    def __init__(self, redis_host: str = None, redis_port: int = None, cache_dir=None):
        """
        Inizializza il Cache Manager.
        
        Args:
            redis_host: Host Redis (default da env REDIS_HOST)
            redis_port: Porta Redis (default da env REDIS_PORT)
            cache_dir: Directory cache locale (deprecato, mantenuto per compatibilità)
        """
        self.host = redis_host or REDIS_HOST
        self.port = redis_port or REDIS_PORT
        self.cache_dir = cache_dir
        self.client = redis.Redis(
            host=self.host, 
            port=self.port, 
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
    
    def is_connected(self) -> bool:
        """Verifica se Redis è raggiungibile."""
        try:
            self.client.ping()
            return True
        except redis.ConnectionError:
            return False
    
    def get_cache(self, key: str) -> Optional[str]:
        """
        Recupera un valore generico dalla cache.
        
        Args:
            key: Chiave Redis
            
        Returns:
            Valore come stringa o None
        """
        try:
            return self.client.get(key)
        except redis.ConnectionError:
            return None
    
    def set_cache(self, key: str, data: Any, expire: int = 86400):
        """
        Salva un valore nella cache.
        
        Args:
            key: Chiave Redis
            data: Dati da salvare (verranno convertiti in JSON se necessario)
            expire: TTL in secondi (default 24 ore)
        """
        try:
            if not isinstance(data, str):
                data = json.dumps(data)
            self.client.set(key, data, ex=expire)
            return True
        except redis.ConnectionError:
            return False
    
    def get_predictions(self) -> Optional[Dict]:
        """
        Recupera le ultime predizioni dal cache.
        
        Returns:
            Dict con le predizioni o None se non disponibili
        """
        try:
            data = self.client.get(REDIS_KEYS["predictions"])
            if data:
                return json.loads(data)
            return None
        except (redis.ConnectionError, json.JSONDecodeError):
            return None
    
    def get_cleaned_data(self) -> Optional[List[Dict]]:
        """
        Recupera i dati puliti/preprocessati.
        
        Returns:
            Lista di record o None
        """
        try:
            data = self.client.get(REDIS_KEYS["cleaned"])
            if data:
                return json.loads(data)
            return None
        except (redis.ConnectionError, json.JSONDecodeError):
            return None
    
    def get_stream(self, limit: int = None, offset: int = 0) -> List[Dict]:
        """
        Recupera i dati dalla stream real-time.
        
        Args:
            limit: Numero massimo di elementi (None = tutti)
            offset: Offset dall'inizio (0 = più vecchio, -1 = più recente)
            
        Returns:
            Lista di data points
        """
        try:
            stream_key = REDIS_KEYS["stream"]
            
            if limit is None:
                # Recupera tutti gli elementi
                data = self.client.lrange(stream_key, 0, -1)
            else:
                # Recupera gli ultimi 'limit' elementi
                data = self.client.lrange(stream_key, -limit, -1)
            
            return [json.loads(item) for item in data]
            
        except (redis.ConnectionError, json.JSONDecodeError):
            return []
    
    def get_stream_length(self) -> int:
        """
        Restituisce il numero di elementi nella stream.
        
        Returns:
            Numero di elementi
        """
        try:
            return self.client.llen(REDIS_KEYS["stream"])
        except redis.ConnectionError:
            return 0
    
    def get_latest(self, count: int = 1) -> List[Dict]:
        """
        Recupera gli ultimi N elementi dalla stream.
        
        Args:
            count: Numero di elementi da recuperare
            
        Returns:
            Lista degli ultimi data points
        """
        return self.get_stream(limit=count)
    
    def get_metadata(self) -> Optional[Dict]:
        """
        Recupera i metadati del worker.
        
        Returns:
            Dict con metadati o None
        """
        try:
            data = self.client.get(REDIS_KEYS["metadata"])
            if data:
                return json.loads(data)
            return None
        except (redis.ConnectionError, json.JSONDecodeError):
            return None
    
    def is_worker_active(self) -> bool:
        """
        Verifica se il cache worker è attivo.
        
        Returns:
            True se il worker è attivo e aggiornato negli ultimi 60 secondi
        """
        metadata = self.get_metadata()
        if not metadata:
            return False
        
        try:
            last_update = datetime.fromisoformat(metadata.get("last_update", ""))
            age_seconds = (datetime.now() - last_update).total_seconds()
            return metadata.get("status") == "running" and age_seconds < 60
        except (ValueError, TypeError):
            return False
    
    def get_status(self) -> Dict:
        """
        Restituisce lo stato completo della cache.
        
        Returns:
            Dict con informazioni di stato
        """
        return {
            "connected": self.is_connected(),
            "worker_active": self.is_worker_active(),
            "stream_length": self.get_stream_length(),
            "has_predictions": self.get_predictions() is not None,
            "has_cleaned_data": self.get_cleaned_data() is not None,
            "metadata": self.get_metadata(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_time_range(self, start_seconds_ago: int = 3600, end_seconds_ago: int = 0) -> List[Dict]:
        """
        Recupera i dati in un range temporale relativo.
        
        Args:
            start_seconds_ago: Inizio del range (secondi fa)
            end_seconds_ago: Fine del range (secondi fa, 0 = ora)
            
        Returns:
            Lista di data points nel range
        """
        stream = self.get_stream()
        now = datetime.now()
        
        filtered = []
        for item in stream:
            try:
                ts = datetime.fromisoformat(item.get("timestamp", ""))
                age = (now - ts).total_seconds()
                if end_seconds_ago <= age <= start_seconds_ago:
                    filtered.append(item)
            except (ValueError, TypeError):
                continue
        
        return filtered


# Singleton instance per uso globale
_cache_manager_instance = None


def get_cache_manager() -> CacheManager:
    """
    Restituisce un'istanza singleton del CacheManager.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance