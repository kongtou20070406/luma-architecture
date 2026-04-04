#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests' / 'datamix_v1.yaml'


def main() -> None:
    text = MANIFEST.read_text(encoding='utf-8')
    print(text)
    summary = {
        'workspace': str(ROOT),
        'manifest': str(MANIFEST),
        'persona_seed_files': sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'persona_seed').glob('*.jsonl')),
        'bucket_docs': sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'buckets').rglob('README.md')),
    }
    out = ROOT / 'manifests' / 'datamix_stats.template.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote template stats to: {out}')


if __name__ == '__main__':
    main()
