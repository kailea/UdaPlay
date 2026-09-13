# UdaPlay - solution notes

## What to run

```
project/solution/
├── Udaplay_01_solution_project.ipynb   # Part 1 - offline RAG / vector database
├── Udaplay_02_solution_project.ipynb   # Part 2 - the agent
├── lib/udaplay.py                      # shared config, schemas, stores (new)
├── games/                              # 25 game records (15 provided + 10 added)
└── config.env.example                  # copy to config.env and add your keys
```

1. `cp config.env.example config.env` and paste in your `OPENAI_API_KEY` and
   `TAVILY_API_KEY`.
2. Run `Udaplay_01_solution_project.ipynb` top to bottom. It creates the persistent
   ChromaDB collection `udaplay` in `./chromadb`.
3. Run `Udaplay_02_solution_project.ipynb` top to bottom. It opens that collection and
   runs the agent on the example queries.

Notebook 2 depends on notebook 1 having been run - it raises immediately if the
collection is empty.

## How the rubric is covered

**RAG**
- `games/*.json` loaded and validated by `load_games()`, each record formatted into a
  document (all six fields embedded) plus scalar metadata for filtering.
- Persistent ChromaDB collection with OpenAI embeddings, cosine distance.
- Semantic search, metadata-filtered search, a persistence check against a fresh client,
  and `GameVectorStore` as the reusable manager both notebooks share.

**Agent tools**
- `retrieve_game` - semantic search over the vector DB, returns typed results with a
  similarity score and a citable `source_id`.
- `evaluate_retrieval` - LLM-as-judge, parsed into the `EvaluationReport` schema
  (`useful`, `confidence`, `description`, `missing_information`), with a JSON fallback
  and a fail-to-web default.
- `game_web_search` - Tavily, returns a synthesised answer plus cited URLs and writes
  each finding to long-term memory.
- `recall_learned_facts` - extra tool; searches what was learned from earlier web
  searches so the same question is not paid for twice.

**Stateful agent**
- `UdaPlayAgent` is a `StateMachine` with seven nodes:
  `entry -> memory_recall -> message_prep -> llm_processor <-> tool_executor ->
  structured_report -> memory_update -> termination`, with an iteration cap on the loop.
- Per-session conversation history via `ShortTermMemory` (one `Run` per turn); the
  notebook demonstrates a follow-up question answered purely from context, and a second
  session that cannot see it.
- Persistent long-term memory in ChromaDB, so knowledge survives a kernel restart.
- Every answer comes with a validated `GameReport`: prose answer, confidence level,
  `used_web_search` flag and typed citations (`internal_db` / `web` / `long_term_memory`).

**Demonstration**
- Three example queries covering the three paths (internal hit, internal hit requiring
  reasoning, web fallback), each printed with its state-machine path, every tool call and
  result, the final answer, citations and the JSON report.
- An automated evaluation section scores those queries with `lib/evaluation.py`'s LLM
  judge and checks the actual tool trace against the expected tools.

## Things worth knowing

- **`SafeLLM`** (in `lib/udaplay.py`) subclasses the course's `LLM` only to fix message
  serialisation: `BaseMessage.dict()` emits `token_usage` and leaves tool-call objects as
  pydantic models, which is not what the Chat Completions API accepts. Everything else is
  unchanged.
- **`LongTermMemoryStore`** exists instead of `lib.memory.LongTermMemory` because that
  class is backed by an in-memory `chromadb.Client()` recreated with `force=True` on every
  run, so it cannot persist between sessions.
- **The embedding model is chosen in one place** (`get_embedding_function()`), because
  opening a collection with a different model silently breaks retrieval. It smoke-tests
  `text-embedding-3-small`, then `text-embedding-ada-002`, then falls back to ChromaDB's
  local model.
- **Dataset additions** are `games/016.json` - `games/025.json` (Breath of the Wild,
  Pokémon Red and Blue, God of War Ragnarök, FIFA 21, The Witcher 3, Red Dead Redemption
  2, Elden Ring, Stardew Valley, Hades, Forza Horizon 5). Mortal Kombat X was deliberately
  left out so the web-fallback demo still has something to fall back on.
