"""Tools for failure analysis: aggregate logs and surface common failure modes."""
from collections import Counter
from typing import List, Dict

def common_failure_modes(logs: List[Dict]) -> Dict[str, int]:
    reasons = Counter()
    for l in logs:
        dec = l.get('decision')
        if dec:
            reasons[dec.get('reason')] += 1
    return dict(reasons)
