from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from ui_rag.api.app import create_app


class FakeService:
    def readiness(self):
        return {"status": "ready", "api_version": "v1", "artifacts": {"base_manifest": {"exists": True}}}

    def model_info(self):
        return {
            "service_version": "0.8.0", "api_version": "v1",
            "retrieval_encoder": "siglip", "retrieval_index": "flatip", "reranker": "rrf",
            "vlm_model": "qwen", "artifact_fingerprint": "abc", "native_isolation": True,
            "evidence_policy": "rank1", "max_concurrent_inference": 1,
        }

    def search(self, *, query: str, top_k: int):
        return {
            "query": query,
            "results": [{"rank": 1, "faiss_id": 7, "filename": "7.jpg", "score": 0.1,
                         "base_rank": 1, "base_score": 0.2, "fine_score": 0.3}],
            "latency": {"total_ms": 4.0, "encode_ms": 2.0, "search_ms": 1.0,
                        "rerank_ms": 1.0, "generation_ms": None},
        }

    def rag(self, *, query: str, max_tokens: int):
        return {
            "query": query,
            "evidence": [{"evidence_id": "E1", "rank": 1, "faiss_id": 7,
                          "filename": "7.jpg", "retrieval_score": 0.1}],
            "answer": {"answer": "login screen", "selected_evidence_id": "E1",
                       "evidence_ids": ["E1"], "observations": [{"evidence_id": "E1", "claim": "login"}],
                       "confidence": None, "abstain": False},
            "citations_valid": True,
            "latency": {"total_ms": 9.0, "encode_ms": 2.0, "search_ms": 1.0,
                        "rerank_ms": 1.0, "generation_ms": 5.0},
        }


def test_control_plane_and_request_id():
    client = TestClient(create_app(service=FakeService()))
    r = client.get('/health', headers={'X-Request-ID': 'req-123'})
    assert r.status_code == 200
    assert r.headers['X-Request-ID'] == 'req-123'
    assert client.get('/ready').status_code == 200
    assert client.get('/v1/model-info').json()['artifact_fingerprint'] == 'abc'


def test_search_contract_has_no_local_path():
    client = TestClient(create_app(service=FakeService()))
    data = client.post('/v1/search', json={'query': 'dark login', 'top_k': 1}).json()
    assert data['results'][0]['filename'] == '7.jpg'
    assert 'path' not in data['results'][0]
    assert data['latency']['encode_ms'] == 2.0


def test_rag_contract_has_no_raw_text_or_local_path():
    client = TestClient(create_app(service=FakeService()))
    data = client.post('/v1/rag', json={'query': 'dark login', 'max_tokens': 64}).json()
    assert data['answer']['selected_evidence_id'] == 'E1'
    assert 'raw_text' not in data['answer']
    assert 'path' not in data['evidence'][0]
    assert data['citations_valid'] is True


def test_blank_query_rejected():
    client = TestClient(create_app(service=FakeService()))
    r = client.post('/v1/search', json={'query': '   '})
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'validation_error'


def test_api_import_does_not_load_native_ml_stacks():
    code = "import sys; import ui_rag.api.app; print(','.join(x for x in ['torch','faiss','mlx','mlx_vlm'] if x in sys.modules))"
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    out = subprocess.check_output([sys.executable, '-c', code], text=True, env=env).strip()
    assert out == ''
