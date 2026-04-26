import logging
import sys
import json
from typing import Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # include record.__dict__ extras (avoid core attributes)
        extras = {k: v for k, v in record.__dict__.items() if k not in ('name','msg','args','levelname','levelno','pathname','filename','module','exc_info','exc_text','stack_info','lineno','funcName','created','msecs','relativeCreated','thread','threadName','processName','process')}
        if extras:
            msg.update(extras)
        return json.dumps(msg, default=str)


def setup_logging(level: Optional[str] = 'INFO') -> None:
    lvl = getattr(logging, (level or 'INFO').upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(lvl)
