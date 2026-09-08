# Setup

## Prerequisites

- Python 3.10 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running commands
- Access to the project's DVC remote (`s3://zanista`, Cloudflare R2-backed) for
  tracked data and benchmark artifacts

## 1. Install dependencies

```bash
uv sync
```

The model-training path needs `torch`/`torchvision`, which are gated behind
hardware-specific optional extras (see `[project.optional-dependencies]` in
`pyproject.toml`):

```bash
uv sync --extra cpu     # CPU-only, any platform
uv sync --extra mps     # Apple Silicon
uv sync --extra cu118   # CUDA 11.8 (Pascal/Volta/Turing)
uv sync --extra cu126   # CUDA 12.6
uv sync --extra cu128   # CUDA 12.8
uv sync --extra cu130   # CUDA 13.0 (Blackwell)
```

Only one hardware extra may be active at a time (they're declared as mutually
exclusive in `[tool.uv.conflicts]`).

## 2. Configure environment variables

```bash
cp .env.example .env
```

Every setting has a working default except the Azure OpenAI credential used by
the default news-sentiment scorer backend (`gpt4o_mini`):

- `TYCHE_SENTIMENT_AZURE_API_KEY` — required to run the news pipeline's scoring
  stage against the default `gpt4o_mini` backend (not needed for the local HF
  backends — `finbert`, `mistral_7b_instruct`,
  `llama2_13b_chat` — selected via `TYCHE_SENTIMENT_BACKENDS`)
- `HF_TOKEN` — optional; only needed for gated/private HuggingFace models
  (e.g. `llama2_13b_chat`) or a higher rate limit on model revision lookups
  (the summarizer and embedder load public weights locally, no token needed
  by default)

See [Configuration Reference](configuration.md) for the full variable list and
how each domain reads it.

## 3. Pull tracked data and artifacts

Data (`data/`) and benchmark results (`benchmark/`) are DVC-tracked, not
committed to git.

```bash
uv run dvc pull
```

This retrieves the daily OHLCV parquet, the news parquet sources, and (optionally) the tracked benchmark CSV/`.npz` outputs used for
report generation.

## 4. Serve the sentiment models you want to test

The Scorer can run any of four sentiment backends, and only some of them need a
model server. Which backends run is chosen with `TYCHE_SENTIMENT_BACKENDS` (see
[News Sentiment Pipeline](news-pipeline.md)).

| Backend | How it runs | What you must set up |
| --- | --- | --- |
| `gpt4o_mini` | Hosted Azure OpenAI call | `TYCHE_SENTIMENT_AZURE_API_KEY` |
| `finbert` | HF weights loaded in-process onto CPU/CUDA/MPS | nothing — weights download on first use |
| `mistral_7b_instruct` | Local OpenAI-compatible HTTP server | start `llm-serve/mistral_7b_instruct.yaml` |
| `llama2_13b_chat` | Local OpenAI-compatible HTTP server | start `llm-serve/llama2_13b_chat.yaml` |

The two generative backends are *not* loaded in-process: they talk to a local
OpenAI-compatible endpoint over HTTP, exactly the way `gpt4o_mini` talks to
Azure. The compose files under `llm-serve/` bring up those endpoints as vLLM
servers, each serving an AWQ-quantized checkpoint with bf16 compute.

### Start the servers

Requires an NVIDIA GPU and the NVIDIA Container Toolkit on the host. Run from
the repo root so the root `.env` is picked up.

```bash
# Start one model
docker compose -f llm-serve/mistral_7b_instruct.yaml up -d
docker compose -f llm-serve/llama2_13b_chat.yaml up -d

# Or start both model files in one compose project
docker compose \
  -f llm-serve/mistral_7b_instruct.yaml \
  -f llm-serve/llama2_13b_chat.yaml \
  up -d

# First start downloads several GB of weights — watch until the health check passes
docker compose -f llm-serve/mistral_7b_instruct.yaml logs -f
docker compose -f llm-serve/llama2_13b_chat.yaml ps

docker compose -f llm-serve/mistral_7b_instruct.yaml down
docker compose -f llm-serve/llama2_13b_chat.yaml down
```

| Compose file | Service | Serves | Published at | `--served-model-name` |
| --- | --- | --- | --- | --- |
| `llm-serve/mistral_7b_instruct.yaml` | `mistral_7b_instruct` | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | `http://localhost:8001/v1` | `mistral-7b-instruct` |
| `llm-serve/llama2_13b_chat.yaml` | `llama2_13b_chat` | `TheBloke/Llama-2-13B-chat-AWQ` | `http://localhost:8002/v1` | `llama2-13b-chat` |

These match `TYCHE_SENTIMENT_MISTRAL_BASE_URL` / `_MODEL` and
`TYCHE_SENTIMENT_LLAMA2_BASE_URL` / `_MODEL` out of the box — if you change a
port, served-model name, or checkpoint in the compose file, change the matching
`.env` value too, since Tyche requests the model by that name.

Notes before you start them:

- The Llama 2 checkpoint is derived from a gated base model. Accept the license
  on the Hub and set `HF_TOKEN` in `.env` first.
- Both services default to GPU 0. Put them on separate GPUs (or start only one
  at a time) with `MISTRAL_GPU_DEVICE` / `LLAMA2_GPU_DEVICE`.
- Override the checkpoint with `MISTRAL_MODEL` / `LLAMA2_MODEL` if you prefer a
  different quantization. If you do, check whether that repo's tokenizer defines
  a `chat_template` — see below.
- Both services are started with an explicit `--chat-template` from
  `docker/chat-templates/`. The AWQ repos ship tokenizers with no chat template
  of their own, and transformers ≥ 4.44 refuses to fall back to a default, so
  without it every request fails with:

  ```
  BadRequestError: 400 — As of transformers v4.44, default chat template is no
  longer allowed, so you must provide a chat template if the tokenizer does not
  define one.
  ```

  The templates encode each model's fine-tuned prompt format (Llama-2's
  `[INST] <<SYS>>…<</SYS>>` block; Mistral's `[INST] …` with the system turn
  folded into the first user turn, since it has no dedicated system role).
  Getting this wrong does not error — it silently degrades output quality, so
  keep the template matched to the checkpoint you serve.

Once a server is healthy, confirm Tyche can reach it before a real run:

```bash
curl http://localhost:8001/v1/models
uv run tyche audit-a   # with the backend selected, per below
```

### Select which backend(s) to score with

`TYCHE_SENTIMENT_BACKENDS` is an ordered, comma-separated list. Each listed
backend writes its own `<backend>_`-prefixed columns; the **first** is the
primary one and additionally fills the canonical unprefixed columns
(`agg_p_pos`, `raw_score`, …) that the Neutralizer, audits, and the output
contract consume.

```bash
# One model at a time — the usual way to produce a run to benchmark
TYCHE_SENTIMENT_BACKENDS=mistral_7b_instruct uv run tyche run

# Score every model in a single pass over the corpus, to compare them row for
# row in one output file (the first entry stays primary)
TYCHE_SENTIMENT_BACKENDS=gpt4o_mini,mistral_7b_instruct,llama2_13b_chat,finbert \
  uv run tyche run
```

Scoring several backends at once summarizes the corpus once and reuses it, so
it is much cheaper than repeating full runs — but the downstream portfolio
pipeline only reads the primary backend's `sentiment_final`. To *benchmark* the
models against each other, do a separate run per backend, writing to its own
output file:

```bash
for backend in gpt4o_mini mistral_7b_instruct llama2_13b_chat finbert; do
  TYCHE_SENTIMENT_BACKENDS=$backend \
    uv run tyche run --output data/output/news_sentiment_$backend.parquet
done
```

Then point the portfolio pipeline at one of them per run. Benchmark artifacts
are keyed by the primary sentiment backend
(`benchmark/<news_sentiment_model>/<target_distribution>/`), so per-backend
results never overwrite each other:

```bash
TYCHE_SENTIMENT_BACKENDS=mistral_7b_instruct \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_mistral_7b_instruct.parquet \
  uv run python -m tyche.portfolio.run --holding 5
# → benchmark/mistral_7b_instruct/student_t/
```

Keep `TYCHE_SENTIMENT_BACKENDS` consistent between the two commands: the
portfolio run reads it only to name the artifact directory, so a mismatch files
results under the wrong model.

## 5. Verify the install

```bash
uv run ruff check .
uv --version
uv run tyche --help
uv run python -m tyche.portfolio.run --help
```

Then exercise each pipeline on a bounded slice before a full run:

```bash
uv run tyche audit-a                    # sanity-check the selected sentiment backend
uv run tyche run --limit 50             # bounded news-pipeline run
uv run python -m tyche.portfolio.run --holding 5
```
