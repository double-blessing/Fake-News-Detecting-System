import re
from collections import Counter

def analyze_logs(log_file):
    with open(log_file) as f:
        logs = f.readlines()
    
    verdicts = Counter()
    common_flags = Counter()
    
    for line in logs:
        if "FAKE" in line:
            verdicts['fake'] += 1
            if "Sensational" in line:
                common_flags['sensational'] += 1
        elif "REAL" in line:
            verdicts['real'] += 1
    
    return {
        "total_checks": len(logs),
        "verdict_distribution": dict(verdicts),
        "common_flags": dict(common_flags)
    }