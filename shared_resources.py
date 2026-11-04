"""
Recursos compartidos por todas las sesiones de la aplicación:
cliente de OpenAI, límite de llamadas simultáneas y caché en memoria.
"""
import os
import json
import hashlib
import threading
import openai

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_CONCURRENT_CALLS = int(os.environ.get("OPENAI_MAX_CONCURRENT_CALLS", "8"))
ANALYSIS_CACHE_MAX_ITEMS = 5000

# Límite global de llamadas simultáneas a OpenAI (compartido por todos los usuarios)
openai_semaphore = threading.BoundedSemaphore(OPENAI_MAX_CONCURRENT_CALLS)

_client = None
_client_lock = threading.Lock()

_cache = {}
_cache_lock = threading.Lock()


def get_openai_client():
    """Devuelve un único cliente de OpenAI, creado la primera vez que se necesita."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY en el servidor.")
            _client = openai.OpenAI(api_key=api_key, timeout=90.0, max_retries=3)
    return _client


def make_cache_key(*parts) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_get(key: str):
    with _cache_lock:
        return _cache.get(key)


def cache_set(key: str, value) -> None:
    with _cache_lock:
        if len(_cache) >= ANALYSIS_CACHE_MAX_ITEMS:
            _cache.pop(next(iter(_cache)))  # elimina el más antiguo
        _cache[key] = value
