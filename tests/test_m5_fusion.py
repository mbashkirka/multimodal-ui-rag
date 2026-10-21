import numpy as np

from ui_rag.fusion.exemplar import choose_nonself_reference_ids, fuse_text_and_reference


def test_fusion_is_l2_normalized_and_respects_endpoints():
    text = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    image = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    fused = fuse_text_and_reference(text, image, text_weight=0.5)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0)
    assert np.allclose(fuse_text_and_reference(text, image, text_weight=1.0), text)
    assert np.allclose(fuse_text_and_reference(text, image, text_weight=0.0), image)


def test_choose_nonself_reference_skips_target():
    neighbors = np.array([[5, 7, 8], [4, 9, 3]], dtype=np.int64)
    targets = np.array([5, 9], dtype=np.int64)
    refs = choose_nonself_reference_ids(neighbors, targets)
    assert refs.tolist() == [7, 4]
