#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description='Check dynamics autoresearch guard summary.')
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f'missing summary: {args.summary}')
        return 2
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    guard = summary.get('guard', {})
    if not guard.get('all_ok', False):
        print(json.dumps({'status': 'guard_failed', 'guard': guard}, ensure_ascii=False))
        return 1
    print(json.dumps({'status': 'ok', 'score': summary.get('score'), 'guard': guard}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
