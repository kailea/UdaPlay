"""
UdaPlay shared building blocks.

This module holds the pieces that BOTH notebooks need, so that Part 1
(offline RAG) and Part 2 (agent) are guaranteed to talk to the *same*
database with the *same* embedding function.

Contents
--------
Configuration
    load_config()              -> read config.env / .env and validate keys
    get_embedding_function()   -> OpenAI embeddings (with safe fallbacks)

Data handling
    load_games()               -> read the games/*.json files
    format_game_document()     -> turn a game record into text to embed

Storage
    GameVectorStore            -> reusable manager around the games collection
    LongTermMemoryStore        -> persistent memory of things learned on the web

LLM plumbing
    SafeLLM                    -> lib.llm.LLM with strict message serialisation

Schemas (pydantic)
    RetrievedGame, EvaluationReport, Citation, GameFact, GameReport
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lib.llm import LLM
from lib.messages import BaseMessage

# --------------------------------------------------------------------------
# Constants - single source of truth for both notebooks
# --------------------------------------------------------------------------

GAMES_DIR = "games"
CHROMA_PATH = "chromadb"
GAMES_COLLECTION = "udaplay"
MEMORY_COLLECTION = "udaplay_long_term_memory"

# Tried in order; the first one the endpoint accepts is used.
EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-ada-002")
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def load_config(env_files: tuple = ("config.env", ".env"), verbose: bool = True) -> Dict[str, str]:
    """
    Load API credentials from the first env file that exists and validate them.

    The OpenAI SDK picks up ``OPENAI_BASE_URL`` from the environment on its own,
    which is what makes the Vocareum proxy work without touching ``lib/llm.py``.
    ChromaDB does *not* read that variable, so ``get_embedding_function`` passes
    the base URL through explicitly.

    Returns a dict with the resolved values (keys are never printed in full).
    """
    loaded_from = None
    for name in env_files:
        if Path(name).exists():
            load_dotenv(name, override=True)
            loaded_from = name
            break

    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    # Make sure both the SDK and ChromaDB see the same values.
    os.environ["OPENAI_BASE_URL"] = base_url
    if openai_key and not os.getenv("CHROMA_OPENAI_API_KEY"):
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    missing = [
        name for name, value in
        (("OPENAI_API_KEY", openai_key), ("TAVILY_API_KEY", tavily_key))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            f"Create a `config.env` file next to this notebook containing "
            f"OPENAI_API_KEY, TAVILY_API_KEY and OPENAI_BASE_URL."
        )

    if verbose:
        source = loaded_from or "the existing environment"
        print(f"Credentials loaded from {source}")
        print(f"  OPENAI_API_KEY  : {_mask(openai_key)}")
        print(f"  TAVILY_API_KEY  : {_mask(tavily_key)}")
        print(f"  OPENAI_BASE_URL : {base_url}")

    return {
        "OPENAI_API_KEY": openai_key,
        "TAVILY_API_KEY": tavily_key,
        "OPENAI_BASE_URL": base_url,
    }


def _mask(value: Optional[str]) -> str:
    if not value:
        return "<missing>"
    return f"{value[:6]}...{value[-4:]} ({len(value)} chars)"


def get_embedding_function(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    models: tuple = EMBEDDING_MODELS,
    verbose: bool = True,
):
    """
    Build the embedding function used by every collection in this project.

    Both notebooks call this so the games collection and the long-term memory
    collection are always embedded the same way. Re-opening a collection with a
    different embedding model silently ruins retrieval (or hard-fails on vector
    dimensions), so the model choice lives here and nowhere else.

    Each candidate model is smoke-tested with a one-word embedding request, and
    the first one the endpoint actually accepts wins. If none work, we fall back
    to ChromaDB's bundled local model so the pipeline still runs offline.
    """
    from chromadb.utils import embedding_functions

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    api_base = api_base or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    for model_name in models:
        try:
            fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=api_base,
                model_name=model_name,
            )
            probe = fn(["udaplay embedding smoke test"])
            if verbose:
                print(f"Embedding function: OpenAI `{model_name}` "
                      f"({len(probe[0])} dimensions)")
            return fn
        except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
            if verbose:
                print(f"  `{model_name}` unavailable -> {type(exc).__name__}: {exc}")

    if verbose:
        print("Falling back to ChromaDB's built-in local embedding model.")
    return embedding_functions.DefaultEmbeddingFunction()


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------

REQUIRED_FIELDS = ("Name", "Platform", "Genre", "Publisher", "Description", "YearOfRelease")


def load_games(data_dir: str = GAMES_DIR) -> List[Dict[str, Any]]:
    """
    Read every ``*.json`` file in ``data_dir`` into a list of game records.

    The file stem (``001``, ``002``, ...) becomes the document id, which keeps
    ingestion idempotent: re-running the notebook upserts over the same ids
    instead of creating duplicates.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"No `{data_dir}` directory found (cwd: {Path.cwd()})")

    games: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            game = json.load(handle)

        missing = [field for field in REQUIRED_FIELDS if field not in game]
        if missing:
            raise ValueError(f"{path.name} is missing field(s): {missing}")

        game["id"] = path.stem
        games.append(game)

    return games


def format_game_document(game: Dict[str, Any]) -> str:
    """
    Build the text that gets embedded for a game.

    Every field goes into the embedded string, not just the description. A
    question like "which games did Nintendo publish on the Game Boy Color?"
    only matches semantically if the publisher and platform are part of the
    vector, and the same fields are also stored as metadata for exact filters.
    """
    return (
        f"{game['Name']} ({game['YearOfRelease']}) - "
        f"Platform: {game['Platform']}. "
        f"Genre: {game['Genre']}. "
        f"Publisher: {game['Publisher']}. "
        f"{game['Description']}"
    )


def game_metadata(game: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar-only metadata dict (ChromaDB rejects nested values)."""
    return {
        "Name": game["Name"],
        "Platform": game["Platform"],
        "Genre": game["Genre"],
        "Publisher": game["Publisher"],
        "Description": game["Description"],
        "YearOfRelease": int(game["YearOfRelease"]),
    }


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class RetrievedGame(BaseModel):
    """One hit from the games collection."""
    source_id: str
    Name: str
    Platform: str
    YearOfRelease: int
    Genre: str = ""
    Publisher: str = ""
    Description: str = ""
    similarity: float = Field(description="1.0 = identical, 0.0 = unrelated")

    def as_context(self) -> str:
        return (
            f"[{self.source_id}] {self.Name} ({self.YearOfRelease}) - "
            f"Platform: {self.Platform}. Genre: {self.Genre}. "
            f"Publisher: {self.Publisher}. {self.Description}"
        )


class EvaluationReport(BaseModel):
    """Verdict produced by the LLM-as-judge in `evaluate_retrieval`."""
    useful: bool = Field(
        description="True only if the documents contain enough information to answer the question"
    )
    confidence: float = Field(
        description="How confident the judge is that the documents answer the question, from 0.0 to 1.0"
    )
    description: str = Field(
        description="Detailed explanation of the verdict, so a caller can act on it"
    )
    missing_information: str = Field(
        default="",
        description="What is missing from the documents; empty string when nothing is missing",
    )


class Citation(BaseModel):
    """Where a piece of the final answer came from."""
    source_type: Literal["internal_db", "web", "long_term_memory"]
    reference: str = Field(description="Document id, URL, or memory id")
    detail: str = Field(default="", description="Short note on what this source contributed")


class GameFact(BaseModel):
    """A structured game fact extracted from the answer."""
    name: str
    platform: str = ""
    year_of_release: str = ""
    publisher: str = ""


class GameReport(BaseModel):
    """
    Machine-readable twin of the agent's natural-language answer.

    Returning both lets a downstream service consume the JSON while a human
    reads the prose.
    """
    question: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    used_web_search: bool
    sources: List[Citation] = Field(default_factory=list)
    games: List[GameFact] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class GameVectorStore:
    """
    Reusable manager for the persistent games collection.

    Wraps the ChromaDB client so callers never deal with raw
    ``{'documents': [[...]], 'distances': [[...]]}`` payloads - queries come
    back as typed :class:`RetrievedGame` objects with a similarity score that
    is directly comparable across queries (cosine distance -> similarity).
    """

    def __init__(
        self,
        path: str = CHROMA_PATH,
        collection_name: str = GAMES_COLLECTION,
        embedding_function=None,
        reset: bool = False,
        verbose: bool = True,
    ):
        import chromadb

        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = embedding_function or get_embedding_function(verbose=verbose)

        if reset:
            try:
                self.client.delete_collection(collection_name)
                if verbose:
                    print(f"Dropped existing collection `{collection_name}`")
            except Exception:
                pass  # nothing to drop on a first run

        self.collection = self._get_or_create(collection_name)
        if verbose:
            print(f"Collection `{collection_name}` ready with {self.count()} documents")

    def _get_or_create(self, name: str):
        # Cosine distance makes similarity scores comparable across queries,
        # which the evaluation tool relies on. Older/newer ChromaDB releases
        # disagree on where that setting lives, so try both spellings.
        try:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
            )

    def count(self) -> int:
        return self.collection.count()

    def add_games(self, games: List[Dict[str, Any]]) -> int:
        """Upsert game records. Safe to call repeatedly - ids are stable."""
        if not games:
            return 0
        self.collection.upsert(
            ids=[game["id"] for game in games],
            documents=[format_game_document(game) for game in games],
            metadatas=[game_metadata(game) for game in games],
        )
        return len(games)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedGame]:
        """Semantic search returning typed results ordered by similarity."""
        raw = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, max(self.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[RetrievedGame] = []
        ids = raw.get("ids", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for source_id, metadata, distance in zip(ids, metadatas, distances):
            hits.append(
                RetrievedGame(
                    source_id=source_id,
                    similarity=round(max(0.0, 1.0 - float(distance)), 4),
                    **{
                        "Name": metadata.get("Name", ""),
                        "Platform": metadata.get("Platform", ""),
                        "YearOfRelease": int(metadata.get("YearOfRelease", 0)),
                        "Genre": metadata.get("Genre", ""),
                        "Publisher": metadata.get("Publisher", ""),
                        "Description": metadata.get("Description", ""),
                    },
                )
            )
        return hits


class LongTermMemoryStore:
    """
    Persistent long-term memory, stored in the same ChromaDB directory.

    Anything the agent learns from the web is written here, so a later session
    can answer the same question without spending another web search. Ids are
    content hashes, which deduplicates repeated findings for free.

    (``lib.memory.LongTermMemory`` exists but is backed by an in-memory
    ``chromadb.Client()`` recreated with ``force=True`` on every run, so it
    cannot survive a kernel restart - hence this persistent variant.)
    """

    def __init__(
        self,
        path: str = CHROMA_PATH,
        collection_name: str = MEMORY_COLLECTION,
        embedding_function=None,
        verbose: bool = True,
    ):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = embedding_function or get_embedding_function(verbose=False)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        if verbose:
            print(f"Long-term memory ready with {self.count()} fragments")

    def count(self) -> int:
        return self.collection.count()

    def remember(self, content: str, source: str, question: str = "", owner: str = "udaplay") -> str:
        """Store one fact. Returns the fragment id."""
        import time

        fragment_id = "mem-" + hashlib.sha1(content.strip().encode("utf-8")).hexdigest()[:12]
        self.collection.upsert(
            ids=[fragment_id],
            documents=[content.strip()],
            metadatas=[{
                "source": source,
                "question": question,
                "owner": owner,
                "timestamp": int(time.time()),
            }],
        )
        return fragment_id

    def recall(self, query: str, n_results: int = 3, min_similarity: float = 0.35) -> List[Dict[str, Any]]:
        """Semantic search over stored fragments, filtered by a similarity floor."""
        if self.count() == 0:
            return []

        raw = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for fragment_id, document, metadata, distance in zip(
            raw.get("ids", [[]])[0],
            raw.get("documents", [[]])[0],
            raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
        ):
            similarity = round(max(0.0, 1.0 - float(distance)), 4)
            if similarity < min_similarity:
                continue
            results.append({
                "id": fragment_id,
                "content": document,
                "source": metadata.get("source", ""),
                "question": metadata.get("question", ""),
                "similarity": similarity,
            })
        return results

    def all_fragments(self) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []
        raw = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": i, "content": d, **(m or {})}
            for i, d, m in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]


# --------------------------------------------------------------------------
# LLM plumbing
# --------------------------------------------------------------------------

class SafeLLM(LLM):
    """
    ``lib.llm.LLM`` with stricter message serialisation.

    ``BaseMessage.dict()`` emits every pydantic field, including
    ``token_usage`` and ``tool_calls: None``, and leaves tool-call objects as
    pydantic models. The Chat Completions API only wants the fields it defines,
    so this subclass converts messages to plain dicts containing exactly the
    keys each role supports. Everything else about ``LLM`` is unchanged.
    """

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        payload = super()._build_payload(messages)
        payload["messages"] = [self._to_api_message(message) for message in messages]
        return payload

    @staticmethod
    def _to_api_message(message: BaseMessage) -> Dict[str, Any]:
        role = message.role

        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content or "",
            }

        if role == "assistant":
            api_message: Dict[str, Any] = {"role": "assistant"}
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                api_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
                if message.content:
                    api_message["content"] = message.content
            else:
                api_message["content"] = message.content or ""
            return api_message

        return {"role": role, "content": message.content or ""}


def truncate(text: str, limit: int = 400) -> str:
    """Shorten long tool output for readable traces."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."
