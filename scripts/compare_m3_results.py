from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _row(label: str, payload: dict) -> dict:
    m = payload['metrics']
    t = payload['timing']
    r = payload['retriever']
    return {
        'encoder': label,
        'backend': r.get('backend', 'open_clip'),
        'dimension': r['dimension'],
        'recall_at_1': m['recall_at_1'],
        'recall_at_5': m['recall_at_5'],
        'recall_at_10': m['recall_at_10'],
        'recall_at_50': m['recall_at_50'],
        'recall_at_100': m['recall_at_100'],
        'mrr_at_10': m['mrr_at_10'],
        'ndcg_at_10': m['ndcg_at_10'],
        'query_embed_ms': t['query_embedding']['amortized_ms_per_query'],
        'faiss_ms': t['amortized_search_ms_per_query'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-reference', default='docs/m2_reference_run.json')
    parser.add_argument('--fixres', default='artifacts/evaluation/screen2words-test-primary-siglip2-fixres/metrics.json')
    parser.add_argument('--naflex', default='artifacts/evaluation/screen2words-test-primary-siglip2-naflex/metrics.json')
    parser.add_argument('--output-dir', default='artifacts/evaluation/m3-comparison')
    args = parser.parse_args()

    ref = _load(Path(args.baseline_reference))
    primary = ref['primary']
    baseline = {
        'retriever': {
            'backend': 'open_clip',
            'dimension': ref['retriever']['dimension'],
        },
        'metrics': {k: primary[k] for k in [
            'recall_at_1','recall_at_5','recall_at_10','recall_at_50','recall_at_100','mrr_at_10','ndcg_at_10'
        ]},
        'timing': {
            'amortized_search_ms_per_query': primary['faiss_ms_per_query'],
            'query_embedding': {'amortized_ms_per_query': primary['query_embedding_ms_per_query']},
        },
    }

    rows = [
        _row('CLIP ViT-B/32 legacy', baseline),
        _row('SigLIP2 Base FixRes 224', _load(Path(args.fixres))),
        _row('SigLIP2 Base NaFlex p256', _load(Path(args.naflex))),
    ]
    base_r10 = rows[0]['recall_at_10']
    base_r100 = rows[0]['recall_at_100']
    for row in rows:
        row['delta_r10_vs_clip'] = row['recall_at_10'] - base_r10
        row['delta_r100_vs_clip'] = row['recall_at_100'] - base_r100

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'comparison.json'
    csv_path = out / 'comparison.csv'
    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print('M3 controlled encoder comparison')
    print('encoder                         R@1     R@5    R@10   R@100   MRR@10   nDCG@10  embed_ms')
    for row in rows:
        print(
            f"{row['encoder']:<30} "
            f"{row['recall_at_1']*100:6.2f}% "
            f"{row['recall_at_5']*100:6.2f}% "
            f"{row['recall_at_10']*100:6.2f}% "
            f"{row['recall_at_100']*100:7.2f}% "
            f"{row['mrr_at_10']:8.4f} "
            f"{row['ndcg_at_10']:9.4f} "
            f"{row['query_embed_ms']:8.3f}"
        )
    print(f'json={json_path}')
    print(f'csv={csv_path}')


if __name__ == '__main__':
    main()
