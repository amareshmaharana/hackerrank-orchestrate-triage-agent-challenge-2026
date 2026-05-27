"""Export evaluation results and transcripts to CSV."""
import csv
import os
from typing import List, Dict

class CSVExporter:
    def __init__(self, out_path: str = "outputs/csv/output.csv"):
        self.out_path = out_path
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def export(self, rows: List[Dict], fieldnames: List[str] = None):
        if not rows:
            return
        if fieldnames is None:
            keys = []
            seen = set()
            for row in rows:
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
        else:
            keys = list(fieldnames)
        with open(self.out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            for r in rows:
                writer.writerow({key: r.get(key, "") for key in keys})
