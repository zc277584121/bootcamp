<a href="https://colab.research.google.com/github/milvus-io/bootcamp/blob/master/integration/mempalace_with_milvus.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>   <a href="https://github.com/milvus-io/bootcamp/blob/master/integration/mempalace_with_milvus.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on%20GitHub-555555?style=flat&logo=github&logoColor=white" alt="GitHub Repository"/>
</a>

# MemPalace with Milvus

[MemPalace](https://github.com/MemPalace/mempalace) is a memory layer for coding agents and long-running development workflows. It can mine and search project memories, conversation notes, and other "drawers" that help an agent keep useful context across sessions.

In this tutorial, we will configure MemPalace to use [Milvus](https://milvus.io/) as its storage backend. Native Milvus support is officially available starting with [MemPalace 3.6.0](https://github.com/MemPalace/mempalace/releases/tag/v3.6.0). The notebook uses Milvus Lite by default, so it can run locally or in Google Colab without a separate Milvus server. The same MemPalace configuration can also point to Milvus server or Zilliz Cloud for shared or production deployments.

## Prerequisites

Install the official MemPalace 3.6.0 release from PyPI with its optional Milvus dependencies. Version 3.6.0 is the first published release that includes the Milvus backend.

```python
%%capture
! pip install --upgrade "mempalace[milvus]==3.6.0"

```

> If you are using Google Colab, to enable dependencies just installed, you may need to **restart the runtime** (click on the "Runtime" menu at the top of the screen, and select "Restart session" from the dropdown menu).

This tutorial uses MemPalace's local embedding model, so you do not need an external model API key. The first run may download a small ONNX embedding model.

Verify that the notebook is running with the published MemPalace release and its Milvus dependencies.

```python
from importlib.metadata import version

print("MemPalace version:", version("mempalace"))
print("PyMilvus version:", version("pymilvus"))
print("Milvus Lite version:", version("milvus-lite"))
```

## Configure MemPalace to use Milvus

MemPalace can work with a real project directory through its CLI, but a notebook is easier to run when it creates a temporary palace directory and inserts a small set of example memories. We set the backend to `milvus`, force CPU embeddings, and keep the demo inside a temporary folder.

```python
import os
import tempfile
from pathlib import Path

os.environ["MEMPALACE_BACKEND"] = "milvus"
os.environ["MEMPALACE_EMBEDDING_MODEL"] = "minilm"
os.environ["MEMPALACE_EMBEDDING_DEVICE"] = "cpu"
os.environ["MEMPALACE_EMBEDDING_THREADS"] = "2"

work_dir = Path(tempfile.mkdtemp(prefix="mempalace_milvus_demo_"))
palace_dir = work_dir / "palace"
palace_dir.mkdir(parents=True, exist_ok=True)

print(f"Palace directory: {palace_dir}")
```

When no remote Milvus URI is configured, the MemPalace Milvus backend creates a local Milvus Lite database at `<palace>/milvus.db`.

> As for the argument of `MilvusClient` used by the backend:
> - Setting the `uri` as a local file, e.g. `./milvus.db`, is the most convenient method, as it automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store all data in this file.
> - If you have large scale of data, you can set up a more performant Milvus server on [Docker or Kubernetes](https://milvus.io/docs/quickstart.md). In this setup, please use the server uri, e.g. `http://localhost:19530`, as your `uri`.
> - If you want to use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.

## Create a MemPalace collection

MemPalace exposes `get_collection` as the main Python API for opening a memory drawer collection. Passing `backend="milvus"` makes the collection use Milvus while still letting us call MemPalace methods such as `upsert`, `query`, and `lexical_search`.

```python
from mempalace.palace import get_collection

collection_name = "mempalace_drawers"
collection = get_collection(
    str(palace_dir),
    collection_name,
    create=True,
    backend="milvus",
)

print(type(collection).__name__)
```

## Add example project memories

MemPalace organizes memory as `wings`, `rooms`, and `drawers`: a wing usually represents a project or person, a room represents a topic area, and a drawer stores the original memory text. Higher-level MemPalace flows such as `mempalace mine` can create these drawers from a project directory or conversation export. In this notebook, we write a few drawers directly so the Milvus integration is easy to see.

The records below mimic a team using MemPalace as long-term project memory for a coding agent. The dataset intentionally mixes deployment, architecture, evaluation, retrieval, UI, and content-planning notes so the search examples have both relevant memories and unrelated distractors. MemPalace computes embeddings locally and stores the text, metadata, dense vectors, and BM25 lexical index in Milvus.

```python
memories = [
    {
        "id": "mem-001",
        "text": (
            "The team uses MemPalace as a long-term memory layer for coding agents, "
            "with Milvus storing searchable drawers for architecture decisions and runbooks."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "architecture",
            "source_file": "architecture-notes.md",
        },
    },
    {
        "id": "mem-002",
        "text": (
            "Local development uses Milvus Lite so engineers can run the memory stack "
            "without a separate vector database service."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "deployment",
            "source_file": "local-dev-runbook.md",
        },
    },
    {
        "id": "mem-003",
        "text": (
            "For shared environments, the same MemPalace backend can point to Milvus "
            "server or Zilliz Cloud by setting MEMPALACE_MILVUS_URI and MEMPALACE_MILVUS_TOKEN."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "deployment",
            "source_file": "cloud-deployment.md",
        },
    },
    {
        "id": "mem-004",
        "text": (
            "The backend keeps verbatim memory text and metadata, while Milvus stores "
            "vectors and supports semantic search over drawers."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "architecture",
            "source_file": "storage-contract.md",
        },
    },
    {
        "id": "mem-005",
        "text": (
            "The evaluation checklist asks the agent to verify semantic search, "
            "metadata filters, and lexical search before publishing an integration guide."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "evaluation",
            "source_file": "evaluation-checklist.md",
        },
    },
    {
        "id": "mem-006",
        "text": (
            "When a query includes exact terms such as Zilliz Cloud token, the agent "
            "should use lexical search as a complement to vector similarity."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "retrieval",
            "source_file": "retrieval-playbook.md",
        },
    },
    {
        "id": "mem-007",
        "text": (
            "The dashboard UI should keep operational controls compact and avoid "
            "large hero sections in repeated engineering workflows."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "ui",
            "source_file": "ui-style-guide.md",
        },
    },
    {
        "id": "mem-008",
        "text": (
            "The community content calendar tracks blog drafts, screenshots, and "
            "review dates for launch posts."
        ),
        "metadata": {
            "wing": "milvus_memory_assistant",
            "room": "content",
            "source_file": "content-calendar.md",
        },
    },
]

collection.upsert(
    ids=[memory["id"] for memory in memories],
    documents=[memory["text"] for memory in memories],
    metadatas=[memory["metadata"] for memory in memories],
)

print(f"Inserted {collection.count()} memories")
print()
print("Memory catalog:")
for memory in memories:
    metadata = memory["metadata"]
    print(
        f"- {memory['id']} room={metadata['room']:<12} "
        f"source={metadata['source_file']}"
    )
```

## Semantic search

Use `query_texts` to search memories by meaning. The query below asks about running the memory stack locally without a database server. In the mixed memory catalog above, the most relevant results should come from the local deployment and Milvus backend notes, not from the UI or content-planning distractors.

```python
semantic_results = collection.query(
    query_texts=["How can I run MemPalace memory locally without a database server?"],
    n_results=3,
    include=["documents", "metadatas"],
)

for rank, (item_id, document, metadata) in enumerate(
    zip(
        semantic_results.ids[0],
        semantic_results.documents[0],
        semantic_results.metadatas[0],
    ),
    start=1,
):
    print(
        f"{rank}. {item_id} room={metadata.get('room')} "
        f"source={metadata.get('source_file')}"
    )
    print(f"   {document}")
```

## Search within a metadata scope

Metadata filters are useful when a project has many memory rooms. This query searches for deployment guidance, but the `where={"room": "deployment"}` filter limits the search to deployment drawers even though the collection also contains architecture, evaluation, retrieval, UI, and content notes.

```python
filtered_results = collection.query(
    query_texts=["How do we deploy this memory backend?"],
    n_results=3,
    where={"room": "deployment"},
    include=["documents", "metadatas"],
)

for rank, (item_id, document, metadata) in enumerate(
    zip(
        filtered_results.ids[0],
        filtered_results.documents[0],
        filtered_results.metadatas[0],
    ),
    start=1,
):
    print(
        f"{rank}. {item_id} room={metadata.get('room')} "
        f"source={metadata.get('source_file')}"
    )
    print(f"   {document}")
```

## Lexical search

The Milvus backend also creates a BM25 sparse index over the memory text. `lexical_search` is useful when you need exact terms such as environment variable names, product names, or file names. Here, the query contains `Zilliz Cloud token`, so the lexical results favor drawers that mention those exact words.

```python
lexical_hits = collection.lexical_search(query="Zilliz Cloud token", n_results=3).hits

for rank, hit in enumerate(lexical_hits, start=1):
    print(
        f"{rank}. {hit.id} room={hit.metadata.get('room')} "
        f"source={hit.metadata.get('source_file')}"
    )
    print(f"   {hit.document}")
```

## Inspect the Milvus collection

MemPalace manages the Milvus schema for us. To confirm what was created, we can open the same Milvus Lite file with `MilvusClient` and inspect the collection directly.

```python
from pymilvus import MilvusClient

collection.close()

milvus_uri = str(palace_dir / "milvus.db")
client = MilvusClient(uri=milvus_uri)
client.load_collection(collection_name)

print("Collections:", client.list_collections())
print("Collection stats:", client.get_collection_stats(collection_name))

rows = client.query(
    collection_name=collection_name,
    filter="",
    limit=5,
    output_fields=["id", "document", "metadata"],
)

print()
print("Sample rows from Milvus:")
for row in rows:
    metadata = row.get("metadata", {})
    print(
        f"- {row['id']} room={metadata.get('room')} "
        f"source={metadata.get('source_file')}"
    )
```

## Optional: use Milvus server or Zilliz Cloud

For a shared deployment, set the Milvus connection environment variables before opening the MemPalace collection. Leave them unset to use the local Milvus Lite file shown above.

```python
# os.environ["MEMPALACE_MILVUS_URI"] = "https://your-cluster.api.region.zillizcloud.com"
# os.environ["MEMPALACE_MILVUS_TOKEN"] = "***********"
# os.environ["MEMPALACE_MILVUS_DB_NAME"] = "default"
# os.environ["MEMPALACE_MILVUS_NAMESPACE"] = "team-memory"
```

## Use the high-level MemPalace workflow

For a real project, you usually do not need to hand-write the drawers as we did in the small Python example above. `mempalace init` inspects the project, proposes rooms from folder structure or filename patterns, and saves that taxonomy in `mempalace.yaml`. `mempalace mine` then scans readable project files, routes each chunk by path, filename, and content keywords, and stores the resulting drawers in the configured backend. For conversation exports, `--mode convos` normalizes chat history and routes it into topic rooms such as `technical`, `architecture`, `planning`, `decisions`, and `problems`.

Use the same Milvus backend flag with those high-level commands:

```shell
mempalace init /path/to/project --backend milvus --yes
mempalace mine /path/to/project --backend milvus
mempalace mine /path/to/conversations --mode convos --backend milvus --wing my_project
mempalace search "How do we deploy memory?" --backend milvus
```

## Conclusion

MemPalace gives agents a structured way to preserve project context: wings keep domains separate, rooms make memory scoping explicit, and drawers keep the original text available for retrieval. Starting with the official MemPalace 3.6.0 release, Milvus provides the storage and search layer behind that structure, combining vector search, metadata filtering, and lexical retrieval in one backend. Milvus Lite keeps the notebook simple, while the same integration path can scale to Milvus server or Zilliz Cloud when the memory layer needs to serve a team or a production agent workflow.
