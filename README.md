# Policy Assistant

Ask a question about company policy and get an answer with quotes from the exact policy section. Every quote is checked against the source text before it is shown.

This is a local demo with sample policies for a fictional retailer. It has no login, so it only listens on your own machine.

## How it works

1. `ingest.py` splits each policy by numbered section, turns each section into an embedding with `nomic-embed-text`, and stores it in Postgres with pgvector.
2. `POST /api/ask` embeds the question and finds the 5 closest sections.
3. An answer model reads those sections and writes an answer with quotes.
4. The app removes any quote that does not appear in the section it cites.

## Prerequisites

- Docker
- [Ollama](https://ollama.com), running
- Python 3.12 or later

## Setup

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d --wait db
python ingest.py
```

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first answer takes longer while the model loads.

To ask from the command line:

```bash
curl -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" -d '{"question": "How many vacation days do I get?"}'
```

## Use Claude for answers

The Claude API is billed separately from a Claude subscription. Create a key in the [Claude Console](https://platform.claude.com), then set these in `.env` and restart the server:

```
ANSWER_PROVIDER=claude
ANTHROPIC_API_KEY=your-key
```

Embeddings and search stay local. The retrieved sections are sent to Anthropic.

**Not yet verified:** the Claude option is tested with fake responses only. It has not been run against the live API.

## Tests

```bash
pytest
python evaluate.py
```

`pytest` needs Postgres running. `evaluate.py` needs Postgres, Ollama and loaded policies. Add `--think` to turn on model thinking, or `--retrieval-only` to skip the answer model.

## Results

`evaluate.py` ran 18 questions with `qwen3:8b`: 13 the policies answer, and 5 they do not.

| Measure | Target | Thinking off (default) | Thinking on |
|---|---|---|---|
| Expected section found by search | 90% | 92% | 92% |
| Answer cites the expected section | 80% | 85% | 77% |
| Unanswerable questions declined | 100% | 80% | 100% |
| Quotes that passed verification | none set | 93% | 71% |
| Average time per question | none set | 7.7s | 23.9s |

**No setting meets every target.** Thinking is off by default because it is about three times faster and cites more accurately.

The 100% with thinking on overstates the result. `evaluate.py` counts a question as declined when it ends with no verified citation. With thinking on, the model still answered the warehouse dress code question, and it only counted as declined because its quotes failed verification. On the page, that answer appears with a warning.

Known weaknesses:

- "How far ahead do I have to book time off?" fails because search does not find section 2.2, so the model says the policies do not cover it.
- "What is the dress code for warehouse staff?" is not declined. The model answers from the safety footwear section instead.
- Similarity scores for answerable and unanswerable questions overlap, so `MIN_SIMILARITY` (0.55) only filters clearly unrelated questions. The model has to decline the rest.
- Verification proves a quote is real. It does not prove the answer text is correct, so the page shows quotes beside the answer.

These results come from a small question set and will vary between runs.
