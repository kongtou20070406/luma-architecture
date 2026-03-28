from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
import time


@dataclass
class Stage0MetricRecord:
    event: str
    ok: bool
    value: float
    note: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_metric_jsonl(path: str, rec: Stage0MetricRecord) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def now() -> float:
    return time.time()

