"""Logging estruturado (JSON Lines) para tracking de desempenho.

Cada evento vira uma linha JSON em outputs/logs/<nome>.jsonl — fácil de consultar com
pandas (`pd.read_json(path, lines=True)`) e de ingerir em ferramentas como Loki,
Cloud Logging ou ELK. No console, um formato curto e legível.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "logs"

# Atributos padrão de LogRecord — tudo que não estiver aqui veio de `extra=` e
# entra no JSON como campo estruturado.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(name: str = "app", level: str | None = None, to_file: bool = True) -> Path | None:
    """Configura o logger raiz: JSON em arquivo + texto no console (stderr).

    Em container (LOG_FORMAT=json), o console também sai em JSON — é o que plataformas
    como Cloud Run / Kubernetes esperam para indexar campos.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # acentos corretos no console do Windows
    console = logging.StreamHandler(sys.stderr)
    if os.getenv("LOG_FORMAT", "text") == "json":
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    log_path = None
    if to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"{name}.jsonl"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Bibliotecas ruidosas
    for noisy in ("urllib3", "httpx", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path
