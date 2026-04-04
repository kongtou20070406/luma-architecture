#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'raw'
PROCESSED = ROOT / 'processed'

DATASET_ENDPOINTS = {
    'openai/gsm8k': 'https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=train&offset=0&length={length}',
    'EleutherAI/hendrycks_math': 'https://datasets-server.huggingface.co/first-rows?dataset=EleutherAI%2Fhendrycks_math&config=algebra&split=train',
    'ricdomolm/MATH-500': 'https://datasets-server.huggingface.co/first-rows?dataset=ricdomolm%2FMATH-500&config=default&split=train',
    'Maxwell-Jia/AIME_2024': 'https://datasets-server.huggingface.co/first-rows?dataset=Maxwell-Jia%2FAIME_2024&config=default&split=train',
    'facebook/empathetic_dialogues': 'https://datasets-server.huggingface.co/first-rows?dataset=facebook%2Fempathetic_dialogues&config=default&split=train',
    'thu-coai/esconv': 'https://datasets-server.huggingface.co/first-rows?dataset=thu-coai%2Fesconv&config=default&split=train',
    'OpenAssistant/oasst1': 'https://datasets-server.huggingface.co/first-rows?dataset=OpenAssistant%2Foasst1&config=default&split=train',
    'HuggingFaceH4/ultrafeedback_binarized': 'https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2Fultrafeedback_binarized&config=default&split=train_prefs',
}

ENABLED = [
    'openai/gsm8k',
    'EleutherAI/hendrycks_math',
    'ricdomolm/MATH-500',
    'Maxwell-Jia/AIME_2024',
    'facebook/empathetic_dialogues',
    'thu-coai/esconv',
    'OpenAssistant/oasst1',
    'HuggingFaceH4/ultrafeedback_binarized',
]


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode('utf-8'))


def normalize_row(dataset_id: str, row: dict) -> dict:
    return {
        'dataset_id': dataset_id,
        'text': json.dumps(row, ensure_ascii=False),
        'raw': row,
    }


def safe_name(dataset_id: str) -> str:
    return dataset_id.replace('/', '__')


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch small normalized samples for enabled public DataMix sources.')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--only', nargs='*', default=[])
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    targets = args.only or ENABLED
    index = []
    for dataset_id in targets:
        if dataset_id not in DATASET_ENDPOINTS:
            print(f'skip unsupported source: {dataset_id}')
            continue
        url = DATASET_ENDPOINTS[dataset_id].format(length=args.limit)
        try:
            payload = fetch_json(url)
        except Exception as exc:
            index.append({'dataset_id': dataset_id, 'status': 'fetch_failed', 'error': str(exc)})
            print(f'failed {dataset_id}: {exc}')
            continue
        raw_path = RAW / f'{safe_name(dataset_id)}.json'
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        rows = payload.get('rows', [])
        if 'first_rows' in payload:
            rows = payload['rows']
        out_path = PROCESSED / f'{safe_name(dataset_id)}.sample.jsonl'
        count = 0
        with out_path.open('w', encoding='utf-8') as handle:
            for item in rows[: args.limit]:
                row = item.get('row', item)
                norm = normalize_row(dataset_id, row)
                handle.write(json.dumps(norm, ensure_ascii=False) + '\n')
                count += 1
        index.append({'dataset_id': dataset_id, 'raw_path': str(raw_path), 'sample_path': str(out_path), 'rows': count})
        print(f'fetched {dataset_id}: {count} rows -> {out_path}')

    index_path = PROCESSED / 'enabled_sources_index.json'
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote index: {index_path}')


if __name__ == '__main__':
    main()
