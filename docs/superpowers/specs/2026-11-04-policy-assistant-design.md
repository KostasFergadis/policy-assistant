# Policy Assistant: Design

**Date:** 2026-11-04
**Status:** Approved

## 1. Goal

An employee asks a question in plain language and gets an answer drawn from company policy documents, with the exact policy section quoted as the reference.

This is a local demo. It runs on one laptop, for one user, with sample data.

### Success criteria

- 7 sample policies are chunked and stored in local Postgres with pgvector.
- `POST /api/ask` returns an answer with verified citations for answerable questions, meeting the targets in section 14.
- Every quote returned as a citation appears in the cited section.
- Questions the policies do not cover get a "not found in the policies" response.
- The web page demonstrates the full flow in a browser.
- The whole flow runs at no cost using local models.
- Claude can be turned on later through settings, with no code changes.

### Out of scope

Login, multiple users, conversation history, document upload through the UI, PDF or Word parsing, production deployment.

## 2. Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Embeddings | `nomic-embed-text` through Ollama | Free and local. Keeps heavy ML packages out of Python |
| Answer model (default) | `qwen3:8b` through Ollama | Free, fits 16 GB, best grounding evidence found for this size |
| Answer model (optional) | `claude-opus-5-5` | Required by the original brief. Enabled once an API key is added |
| Database | `pgvector/pgvector:pg17` in Docker on port 5433 | Leaves the Homebrew Postgres 14 install untouched |
| Chunking | One chunk per numbered section | Every citation maps to a real section |
| Vector search | Exact cosine search, no index | A few dozen rows do not need an index |
| Frontend | One static HTML file served by FastAPI | No build step |

**Deviation from the brief:** the brief specifies a Claude API call. The default is a local model for now, at the user's request. The brief's Claude step is only fully met after the Claude option is run with a real key.

## 3. Architecture

```
Ingest:    policies/*.md -> chunker -> Ollama embeddings -> Postgres

Ask:       question -> Ollama embedding -> top 5 sections from Postgres
           -> similarity check -> answer provider -> quote verification -> response
```

### Project layout

```
policy-assistant/
  docker-compose.yml
  requirements.txt
  .env.example
  .gitignore
  README.md
  ingest.py              Command-line entry point for loading policies
  evaluate.py            Runs the sample questions and prints a scorecard
  policies/              7 Markdown policies
  app/
    main.py              FastAPI app and routes
    config.py            Settings read from .env
    chunking.py          Markdown to sections
    embeddings.py        Ollama embedding calls
    db.py                Connection, schema, search
    verification.py      Quote checking
    providers/
      base.py            Shared answer types and provider interface
      ollama.py          Local answer provider
      claude.py          Claude answer provider
  static/index.html
  tests/
    eval_questions.json
```

## 4. Sample documents

Seven Markdown policies for a fictional retailer, each using numbered `##` sections such as `## 3.2 Annual Leave`:

1. Leave Policy
2. Workplace Safety
3. Store Opening Checklist
4. Store Closing Checklist
5. Code of Conduct
6. Expense Policy
7. Cash Handling

## 5. Data model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  title     TEXT NOT NULL,
  filename  TEXT NOT NULL UNIQUE
);

CREATE TABLE chunks (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  document_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section_number  TEXT NOT NULL,
  section_title   TEXT NOT NULL,
  part            INTEGER NOT NULL DEFAULT 1,
  content         TEXT NOT NULL,
  embedding       VECTOR(768) NOT NULL,
  UNIQUE (document_id, section_number, part)
);

CREATE INDEX chunks_document_id_idx ON chunks (document_id);
```

## 6. Ingestion

`python ingest.py` does the following:

1. Reads every `.md` file in `policies/`.
2. Takes the title from the `#` heading and splits at `##` section headings.
3. Splits any section over 1,500 characters at paragraph breaks. The parts share a section number and have increasing `part` values.
4. Embeds each chunk with the `search_document: ` prefix that `nomic-embed-text` expects. The text embedded includes the document title and section heading, so a short section keeps its context.
5. Replaces all rows in a single transaction.

All embeddings are computed before the transaction starts. If Ollama fails, the existing data is left untouched. A policy file with no valid sections stops the run with an error naming the file.

## 7. API

### `POST /api/ask`

Request: `{ "question": "..." }`, 1 to 500 characters after trimming.

Response:

```json
{
  "answer": "You receive 20 days of annual leave per year.",
  "grounded": true,
  "citations": [
    {
      "document": "Leave Policy",
      "section_number": "3.2",
      "section_title": "Annual Leave",
      "quoted_text": "Full-time employees receive 20 days of annual leave"
    }
  ],
  "retrieved": [
    { "document": "Leave Policy", "section_number": "3.2", "section_title": "Annual Leave", "similarity": 0.81 }
  ],
  "model": "qwen3:8b"
}
```

`grounded` is `true` when at least one citation passed verification. It is `false` for "not found" responses and for answers whose quotes all failed.

### `GET /api/health`

Reports the status of the database, Ollama, and the configured answer provider. The page uses it to show setup problems.

## 8. Retrieval

- The question is embedded with the `search_query: ` prefix.
- pgvector returns the 5 closest chunks by cosine similarity (`1 - (embedding <=> query)`).
- If the best score is below `MIN_SIMILARITY`, the API returns "not found in the policies" without calling an answer model.
- `MIN_SIMILARITY` starts at 0.5 and is set from the evaluation results during the build, using the scores of answerable and unanswerable questions.

## 9. Answer providers

Both providers implement the same interface: they take the question and retrieved chunks, and return answer text plus citations in a shared format.

The provider is chosen by `ANSWER_PROVIDER` (`ollama` or `claude`). The model is chosen by `OLLAMA_MODEL` or `CLAUDE_MODEL`.

The prompt tells the model to answer only from the supplied sections, to say so if they do not contain the answer, and to treat section text as reference content rather than instructions.

### Ollama provider

- Calls Ollama's `/api/chat` with a JSON schema in `format`, so the output has this shape: `{ answer, citations: [{ chunk_ref, quote }] }`.
- Sections are labelled in the prompt with short references such as `S1`, which map back to chunks.
- Thinking is off by default for speed. `OLLAMA_THINK=true` turns it on, and `evaluate.py` compares both settings.

### Claude provider

- Sends each chunk as a `document` content block with `citations: {enabled: true}`, titled with its document and section.
- Maps `document_index` in each returned citation back to its chunk, and uses `cited_text` as the quote.
- Uses `output_config: {effort: "low"}`, with refusal fallback enabled through `fallbacks: "default"`.
- Checks `stop_reason` before reading content, and treats a refusal as a provider error.

## 10. Quote verification

Every citation from either provider goes through `verification.py`:

1. Find the chunk the citation refers to. An unknown reference is removed.
2. Check that the quote appears in that chunk's text. The comparison is exact, except that runs of whitespace are treated as a single space.
3. Remove any citation that fails, and log it.

If no citation survives, the answer is returned with `grounded: false`, and the page shows a warning that the answer could not be matched to policy text.

**Limit:** this proves the quote is real. It does not prove the answer text is correct, which is why the page shows quotes beside the answer.

## 11. Error handling

| Situation | Response |
|---|---|
| Empty or over-long question | 422 |
| Database unreachable | 503 naming the database, with the command to start it |
| Ollama unreachable or a model is missing | 503 naming the missing piece, with the command to fix it |
| Database has no chunks | 503 telling the user to run `python ingest.py` |
| Claude selected with no API key | 503 explaining how to add the key |
| Claude rate limit or API failure | 503 with a retry message |
| Model output cannot be parsed | One retry, then 502 |
| Unexpected error | 500 with a generic message. Details go to the server log only |

## 12. Frontend

One page at `/`, styled with plain CSS and using `fetch`:

- A question box and a few example questions.
- The answer, with a warning when `grounded` is false.
- Citation cards showing document, section, and quoted text.
- A collapsible list of retrieved sections with similarity scores.
- Loading, error, and setup-problem states.

Model output is inserted with `textContent`, never as HTML.

## 13. Security

- The server binds to `127.0.0.1` only. There is no login.
- SQL uses bound parameters.
- The API key lives in `.env`, which is in `.gitignore`. `.env.example` holds no secrets.
- Postgres publishes its port on `127.0.0.1` only.

## 14. Testing

**Automated, no models needed (`pytest`):**
- Chunking: headings, section numbers, long-section splits, bad files.
- Verification: exact match, whitespace differences, invented quotes, unknown references.
- API: success, not found, validation, each error in section 11, with fake providers and fake search.
- Claude provider: citation mapping from a recorded-style fake response.

**Evaluation against real local models (`evaluate.py`):**

`tests/eval_questions.json` holds about 15 questions: 10 or more answerable, each with its expected section, and at least 3 that the policies do not cover.

The scorecard reports:
- Retrieval: expected section in the top 5.
- Citation: answer cites the expected section, and how many quotes passed verification.
- Refusal: unanswerable questions return "not found" or an answer saying so.
- Average response time.

Targets for `qwen3:8b`:
- Retrieval finds the expected section for at least 90% of answerable questions.
- At least 80% of answerable questions end with a verified citation to the expected section.
- All unanswerable questions are declined.

If a target is missed, the results are reported rather than the target being lowered.

**End-to-end:** a real question asked in a browser, checked for answer, citation and retrieved sections.

**Not verified until a key is added:** the Claude provider against the live API.

## 15. Setup and environment

- **Python:** a project virtual environment on Python 3.15. If a dependency will not install on 3.15, the project uses Python 3.12 from Homebrew instead.
- **Python packages:** `fastapi`, `uvicorn`, `psycopg[binary]`, `pgvector`, `httpx`, `anthropic`, `python-dotenv`, `pytest`.
- **Downloads:** `pgvector/pgvector:pg17` image, `nomic-embed-text`, `qwen3:8b` (5.2 GB).

## 16. Risks

| Risk | Mitigation |
|---|---|
| `qwen3:8b` paraphrases instead of quoting exactly | Prompt asks for word-for-word quotes, and failed quotes are removed and flagged. The evaluation measures this |
| Slow answers on the laptop | Thinking is off by default, and the page shows a loading state. Speed is measured in the evaluation |
| Memory pressure with Docker and the model running | Postgres uses little memory, and the model is 5.2 GB. Revisit if the machine struggles |
| Packages unavailable for Python 3.15 | Fall back to Python 3.12 |
