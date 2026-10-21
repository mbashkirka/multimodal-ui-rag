from pathlib import Path


def test_m3_makefile_has_controlled_candidates():
    text = Path('Makefile').read_text(encoding='utf-8')
    assert 'google/siglip2-base-patch16-224' in text
    assert 'google/siglip2-base-patch16-naflex' in text
    assert '--max-num-patches 256' in text
    assert 'm3-benchmark-fixres' in text
    assert 'm3-benchmark-naflex' in text


def test_siglip2_transformers_import_is_lazy():
    text = Path('src/ui_rag/embeddings/siglip2.py').read_text(encoding='utf-8')
    before_init = text.split('def __init__', 1)[0]
    assert 'from transformers import' not in before_init
