"""Write structured JSONL transcripts with timestamps and reasoning traces."""
import json
import os
import platform
import sys
import uuid
from datetime import datetime

class TranscriptLogger:
    def __init__(self, out_dir: str = "outputs/transcripts"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        self.path = os.path.join(self.out_dir, "log.txt")
        self.base_context = {
            "session_id": self.session_id,
            "app": "triage-agent",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        }

    def log(self, record: dict):
        record = {**self.base_context, **record}
        record['ts'] = datetime.utcnow().isoformat()
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
