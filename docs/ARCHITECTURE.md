# Architecture

## 1. Online data plane

```mermaid
flowchart LR
    Q[Query] --> A[FastAPI /v1/search or /v1/rag]
    A --> E[Encoder worker\nSigLIP2 NaFlex p256]
    E --> F[FAISS worker\nIndexFlatIP]
    F --> T[Top-20]
    T --> R[p512 rescoring + weighted RRF]
    R --> K[Final ranked evidence]
    K -->|/v1/search| S[Search response]
    K -->|/v1/rag: E1| V[VLM worker\nQwen2.5-VL-3B 4-bit]
    V --> G[Grounded answer]
    K --> P[Deterministic provenance]
    P --> G
```

## 2. Native-process isolation

The host intentionally does not co-load PyTorch, FAISS, and MLX.

```mermaid
flowchart TB
    H[FastAPI host\nno torch / faiss / mlx imports]
    H <-->|JSON IPC| W1[Encoder worker\nPyTorch + Transformers\nApple MPS]
    H <-->|JSON IPC| W2[FAISS worker\nCPU IndexFlatIP]
    H <-->|JSON IPC| W3[VLM worker\nMLX + Qwen2.5-VL]
```

This architecture was introduced because loading PyTorch/OpenCLIP and FAISS in the same macOS process caused a native OpenMP crash. M8.1 changed the workers from per-request processes to persistent processes so that isolation remained intact without repeated model startup.

## 3. Retrieval design

```mermaid
flowchart LR
    C[66,261 RICO screenshots] --> P256[SigLIP2 NaFlex p256\n768-d normalized embeddings]
    P256 --> FLAT[Exact FAISS IndexFlatIP]
    Q[Text query] --> QE[SigLIP2 text encoder]
    QE --> FLAT
    FLAT --> C20[Top-20]
    C20 --> P512[p512 corpus candidate scores]
    P512 --> RRF[Weighted RRF\nbase 0.5 / fine 0.5 / c=60]
    RRF --> TOP[Final Top-K]
```

The p512 stage does not build another global FAISS index. It only rescales the candidate set produced by the frozen p256 retriever.

## 4. Grounded generation contract

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant RET as Retrieval + Rerank
    participant VLM as Qwen2.5-VL worker

    U->>API: query
    API->>RET: retrieve Top-K
    RET-->>API: E1, E2, E3, E4
    Note over API: E1 selected deterministically
    API->>VLM: query + E1 screenshot only
    VLM-->>API: grounded natural-language answer
    Note over API: evidence_ids=[E1] and observation provenance built by application code
    API-->>U: answer + E1 citation + latency
```

Rejected alternatives are documented in M6:

- VLM multi-image JSON grounding;
- VLM evidence selector;
- VLM verifier/abstention gate.

They were removed because measured quality was worse than simpler deterministic behavior.

## 5. Operational control plane

```mermaid
flowchart TB
    HEALTH[/health/] --> L[Liveness]
    READY[/ready/] --> A[Artifact presence]
    READY --> W[Worker liveness / restarts / timeouts]
    INFO[/v1/model-info/] --> M[Model / index / fingerprint / serving identity]
    METRICS[/metrics/] --> PM[Prometheus-compatible counters]
    REQ[Inference request] --> SEM{Capacity available?}
    SEM -->|yes| PIPE[Data plane]
    SEM -->|no| BUSY[HTTP 429 + busy counter]
    DEAD{Worker dead / transport failure?} -->|yes| RESTART[Restart worker + one retry]
```

## 6. Artifact model

Large generated artifacts are not committed. Manifests tie the runtime to:

- source mapping hash;
- canonical mapping hash;
- encoder identity/configuration;
- embedding dimensions and dtype;
- corpus vector count;
- FAISS index hash/type;
- p512 reranker artifact hash;
- library/runtime metadata.

This makes the project reproducible without storing the full corpus, model cache, or hundreds of megabytes of generated arrays/indexes in Git.
