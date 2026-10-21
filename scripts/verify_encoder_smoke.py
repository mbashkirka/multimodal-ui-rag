from __future__ import annotations

import argparse
from pathlib import Path

from ui_rag.indexing.manifest import read_json, sha256_file
from ui_rag.retrieval.orchestrator import search_isolated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--self-image', default='data_storage/RICO_dataset/71018.jpg')
    parser.add_argument('--text', default='mobile application screen')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    index_path = Path(manifest['index']['path'])
    if not index_path.is_absolute():
        index_path = Path.cwd() / index_path
    mapping_path = Path(manifest['mapping']['artifact_path'])
    if not mapping_path.is_absolute():
        mapping_path = Path.cwd() / mapping_path

    if sha256_file(index_path) != manifest['index']['sha256']:
        raise RuntimeError('FAISS index hash mismatch')
    if sha256_file(mapping_path) != manifest['mapping']['artifact_sha256']:
        raise RuntimeError('mapping hash mismatch')

    image_result = search_isolated(
        manifest_path=manifest_path,
        query=None,
        image_path=Path(args.self_image),
        top_k=args.top_k,
        text_weight=0.5,
        device=args.device,
        verify_hashes=True,
    )
    top1 = image_result['results'][0]
    expected_name = Path(args.self_image).name
    if top1['filename'] != expected_name:
        raise RuntimeError(
            f'self retrieval failed: expected {expected_name}, got {top1["filename"]}'
        )
    if top1['score'] < 0.999:
        raise RuntimeError(f'self retrieval cosine too low: {top1["score"]}')

    text_result = search_isolated(
        manifest_path=manifest_path,
        query=args.text,
        image_path=None,
        top_k=args.top_k,
        text_weight=0.5,
        device=args.device,
        verify_hashes=True,
    )

    emb = manifest['embedding']
    print('Encoder smoke verification PASSED')
    print(f"backend={emb.get('backend', 'open_clip')}")
    print(f"model={emb['model_name']}")
    print(f"vectors={manifest['index']['num_vectors']}")
    print(f"dimension={emb['dimension']}")
    print(f"index_type={manifest['index']['type']}")
    print(f"query_encoder_device={image_result['encoder']['device']}")
    print('\nImage self-retrieval:')
    for item in image_result['results']:
        print(
            f"  {item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )
    print(f'\nText retrieval for "{args.text}":')
    for item in text_result['results']:
        print(
            f"  {item['rank']:>2}. score={item['score']:.6f} "
            f"faiss_id={item['faiss_id']} file={item['filename']}"
        )


if __name__ == '__main__':
    main()
