from ui_rag.embeddings.factory import encoder_kwargs_from_manifest, infer_backend


def test_backend_inference_keeps_legacy_clip():
    assert infer_backend('ViT-B-32-quickgelu') == 'open_clip'


def test_backend_inference_recognizes_siglip2():
    assert infer_backend('google/siglip2-base-patch16-224') == 'transformers_siglip2'


def test_manifest_factory_args_are_backward_compatible():
    cfg = {'model_name': 'ViT-B-32-quickgelu', 'pretrained': 'openai', 'dimension': 512}
    kwargs = encoder_kwargs_from_manifest(cfg)
    assert kwargs == {
        'model_name': 'ViT-B-32-quickgelu',
        'pretrained': 'openai',
        'backend': None,
        'max_num_patches': None,
    }
