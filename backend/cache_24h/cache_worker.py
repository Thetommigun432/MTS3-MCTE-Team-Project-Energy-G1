"""
Cache Worker - Lavoratore unico per la gestione della cache Redis.

Responsabilità:
    - Processare i dati grezzi ogni secondo
    - Pulire e preprocessare i dati
    - Salvare in Redis le ultime 24 ore di dati processati
    - Mantenere una finestra scorrevole di 86400 secondi (24h)
"""

import time
import os
import redis
import json
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys

# Aggiungi il path per importare i moduli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.preprocessing.preprocessing_engine import PreprocessingEngine
from ai.model.model_predictor import ModelPredictor

# Configurazione
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Path configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_DATA_PATH = os.path.join(BASE_DIR, 'Client-data', 'nilm_ready_dataset.parquet')
MODEL_PATH = os.path.join(BASE_DIR, 'ai', 'model', 'transformer_heatpump_best.pth')

# Redis Keys
REDIS_KEYS = {
    "stream": "energy:stream",           # Lista dei dati real-time (ultimi 24h)
    "cleaned": "energy:cleaned",          # Dati puliti pronti per il modello
    "predictions": "energy:predictions",  # Ultime predizioni
    "metadata": "energy:metadata",        # Metadati (ultimo aggiornamento, stato, ecc.)
    "lock": "energy:worker_lock"          # Lock per garantire un solo worker attivo
}

# Costanti
SECONDS_IN_24H = 86400
WORKER_INTERVAL = 1  # secondi tra ogni ciclo


class CacheWorker:
    """
    Worker singleton che gestisce la cache Redis con i dati delle ultime 24 ore.
    
    Utilizza un lock distribuito Redis per garantire che solo un'istanza
    sia attiva alla volta (importante in ambiente Docker con più repliche).
    """
    
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        self.preprocessor = None
        self.model_predictor = None
        self.running = False
        self.worker_id = f"worker_{os.getpid()}_{datetime.now().timestamp()}"
        
    def _init_components(self):
        """Inizializza preprocessor e model predictor (lazy loading)."""
        if self.preprocessor is None:
            print("📦 Inizializzazione PreprocessingEngine...")
            self.preprocessor = PreprocessingEngine()
            
        if self.model_predictor is None:
            print("🤖 Caricamento modello AI...")
            self.model_predictor = ModelPredictor(MODEL_PATH)
            
    def _acquire_lock(self, timeout=30):
        """
        Acquisisce un lock distribuito per garantire un solo worker.
        
        Args:
            timeout: Tempo massimo di validità del lock in secondi
            
        Returns:
            bool: True se il lock è stato acquisito, False altrimenti
        """
        lock_key = REDIS_KEYS["lock"]
        
        # Prova ad acquisire il lock con NX (solo se non esiste)
        acquired = self.redis.set(lock_key, self.worker_id, nx=True, ex=timeout)
        
        if acquired:
            return True
            
        # Controlla se il lock è nostro
        current_holder = self.redis.get(lock_key)
        return current_holder == self.worker_id
    
    def _release_lock(self):
        """Rilascia il lock se è nostro."""
        lock_key = REDIS_KEYS["lock"]
        current_holder = self.redis.get(lock_key)
        
        if current_holder == self.worker_id:
            self.redis.delete(lock_key)
    
    def _refresh_lock(self):
        """Rinnova il TTL del lock."""
        lock_key = REDIS_KEYS["lock"]
        current_holder = self.redis.get(lock_key)
        
        if current_holder == self.worker_id:
            self.redis.expire(lock_key, 30)
            return True
        return False
    
    def _wait_for_redis(self, max_retries=30, retry_interval=2):
        """Attende che Redis sia disponibile."""
        for attempt in range(max_retries):
            try:
                self.redis.ping()
                print(f"✅ Connessione a Redis stabilita ({REDIS_HOST}:{REDIS_PORT})")
                return True
            except redis.ConnectionError:
                print(f"⏳ Attesa Redis... tentativo {attempt + 1}/{max_retries}")
                time.sleep(retry_interval)
        
        print("❌ Impossibile connettersi a Redis")
        return False
    
    def push_data_point(self, data_point: dict):
        """
        Aggiunge un singolo dato alla stream e mantiene solo le ultime 24 ore.
        
        Args:
            data_point: Dizionario con timestamp e valore processato
        """
        stream_key = REDIS_KEYS["stream"]
        
        # Aggiungi il dato alla lista
        self.redis.rpush(stream_key, json.dumps(data_point))
        
        # Mantieni solo gli ultimi SECONDS_IN_24H elementi (24 ore a 1 dato/sec)
        self.redis.ltrim(stream_key, -SECONDS_IN_24H, -1)
    
    def save_cleaned_data(self, cleaned_data):
        """
        Salva i dati puliti in Redis.
        
        Args:
            cleaned_data: DataFrame o dict con i dati già preprocessati
        """
        cleaned_key = REDIS_KEYS["cleaned"]
        
        if isinstance(cleaned_data, pd.DataFrame):
            data_json = cleaned_data.to_json(orient='records', date_format='iso')
        else:
            data_json = json.dumps(cleaned_data)
        
        # Salva con TTL di 24 ore
        self.redis.set(cleaned_key, data_json, ex=SECONDS_IN_24H)
    
    def save_predictions(self, predictions):
        """
        Salva le predizioni in Redis.
        
        Args:
            predictions: DataFrame o dict con le predizioni
        """
        predictions_key = REDIS_KEYS["predictions"]
        
        if isinstance(predictions, pd.DataFrame):
            data = {
                "predictions": predictions.to_dict(orient='records'),
                "generated_at": datetime.now().isoformat(),
                "model": "transformer_heatpump"
            }
        else:
            data = predictions
        
        self.redis.set(predictions_key, json.dumps(data), ex=SECONDS_IN_24H)
    
    def update_metadata(self, status="running", **extra):
        """Aggiorna i metadati del worker."""
        metadata = {
            "worker_id": self.worker_id,
            "status": status,
            "last_update": datetime.now().isoformat(),
            "redis_host": REDIS_HOST,
            **extra
        }
        self.redis.set(REDIS_KEYS["metadata"], json.dumps(metadata), ex=SECONDS_IN_24H)
    
    def process_and_cache(self):
        """
        Esegue il ciclo completo di processamento:
            1. Legge i dati grezzi
            2. Preprocessa
            3. Esegue predizioni
            4. Salva tutto in Redis
        """
        try:
            # 1. Leggi dati grezzi
            if not os.path.exists(CLIENT_DATA_PATH):
                print(f"⚠️ Dataset non trovato: {CLIENT_DATA_PATH}")
                return False
            
            raw_df = pd.read_parquet(CLIENT_DATA_PATH)
            
            # 2. Preprocessing
            cleaned = self.preprocessor.clean_data(raw_df)
            sequences = self.preprocessor.prepare_sequences(cleaned)
            
            if len(sequences) == 0:
                print("⚠️ Nessuna sequenza generata dal preprocessing")
                return False
            
            # 3. Salva dati puliti
            self.save_cleaned_data(cleaned)
            
            # 4. Esegui predizioni
            sequence = sequences[0]
            predictions = self.model_predictor.predict_batch(sequence)
            
            # 5. Salva predizioni
            self.save_predictions(predictions)
            
            # 6. Aggiungi punti dati alla stream (simula real-time)
            if isinstance(predictions, pd.DataFrame):
                for _, row in predictions.iterrows():
                    data_point = {
                        "timestamp": row.get('timestamp', datetime.now().isoformat()),
                        "heatpump_power": row.get('heatpump_power', 0),
                        "total_power": row.get('total_power', 0)
                    }
                    self.push_data_point(data_point)
            
            return True
            
        except Exception as e:
            print(f"❌ Errore nel processamento: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """
        Loop principale del worker.
        
        Mantiene il lock e processa i dati continuamente.
        """
        print("=" * 60)
        print("🔧 CACHE WORKER - Lavoratore Unico")
        print("=" * 60)
        print(f"Worker ID: {self.worker_id}")
        print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
        print(f"Intervallo: {WORKER_INTERVAL}s")
        print(f"Finestra dati: {SECONDS_IN_24H}s (24 ore)")
        print("=" * 60)
        
        # Attendi che Redis sia disponibile
        if not self._wait_for_redis():
            return
        
        # Inizializza componenti
        self._init_components()
        
        # Primo processamento completo
        print("\n🔄 Esecuzione processamento iniziale...")
        if self.process_and_cache():
            print("✅ Processamento iniziale completato")
        
        self.running = True
        last_full_process = datetime.now()
        
        print("\n🚀 Worker avviato. Monitoraggio continuo...")
        
        while self.running:
            try:
                # Acquisisci/Rinnova lock
                if not self._acquire_lock():
                    print("⚠️ Un altro worker è attivo. In attesa...")
                    time.sleep(5)
                    continue
                
                self._refresh_lock()
                
                # Aggiorna metadata
                stream_length = self.redis.llen(REDIS_KEYS["stream"])
                self.update_metadata(
                    status="running",
                    stream_length=stream_length,
                    last_full_process=last_full_process.isoformat()
                )
                
                # Ogni ora esegui un processamento completo
                if (datetime.now() - last_full_process).total_seconds() > 3600:
                    print("\n🔄 Rielaborazione periodica...")
                    if self.process_and_cache():
                        last_full_process = datetime.now()
                        print("✅ Rielaborazione completata")
                
                # Log periodico
                if stream_length % 60 == 0:  # Ogni minuto
                    print(f"📊 Stream: {stream_length} elementi | {datetime.now().strftime('%H:%M:%S')}")
                
                time.sleep(WORKER_INTERVAL)
                
            except redis.ConnectionError as e:
                print(f"❌ Connessione Redis persa: {e}")
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n🛑 Interruzione richiesta...")
                self.running = False
                
            except Exception as e:
                print(f"❌ Errore: {e}")
                time.sleep(5)
        
        # Cleanup
        self._release_lock()
        self.update_metadata(status="stopped")
        print("👋 Worker terminato")
    
    def stop(self):
        """Ferma il worker gracefully."""
        self.running = False


def main():
    """Entry point del worker."""
    worker = CacheWorker()
    
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
