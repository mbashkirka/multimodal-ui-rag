from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest


def test_flat_ip_equals_cosine_for_normalized_vectors_in_isolated_process() -> None:
    """Run FAISS contract in a process that never imports torch.

    On macOS/Apple Silicon, importing PyTorch and faiss-cpu in the same Python
    process can load conflicting OpenMP runtimes and abort the interpreter.
    The production pipeline already isolates these native stacks; the test
    harness must preserve the same contract.
    """

    if importlib.util.find_spec("faiss") is None:
        pytest.skip("faiss is not installed in this environment")

    code = textwrap.dedent(
        """
        import faiss
        import numpy as np

        vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        index = faiss.IndexFlatIP(2)
        index.add(vectors)

        query = np.array([[1.0, 0.0]], dtype=np.float32)
        scores, indices = index.search(query, 3)

        assert indices[0, 0] == 0
        assert np.isclose(scores[0, 0], 1.0)
        assert scores[0, 1] >= scores[0, 2]
        print("FAISS_ISOLATED_CONTRACT_OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        "isolated FAISS contract failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "FAISS_ISOLATED_CONTRACT_OK" in result.stdout
