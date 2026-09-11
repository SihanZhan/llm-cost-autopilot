# LLM Cost Autopilot — Project Brief

## What you're building

An intelligent routing layer that sits in front of multiple LLM providers,
analyzes each incoming request's complexity, routes it to the **cheapest model
capable of handling it at acceptable quality**, and continuously validates that
routing decisions are correct.

## Why this project lands interviews

Every company running LLMs at scale is bleeding money on over-provisioned model
calls. Building a cost optimizer signals that you understand AI engineering as a
business problem, not just a technical one — and that's the gap between a junior
hire and a senior one.

## Tech stack

| Component | Tool / library | Why this choice |
|---|---|---|
| Language | Python 3.11+ | Ecosystem compatibility |
| LLM providers | OpenAI, Anthropic, Ollama (local) | Mix of cloud and local models |
| Router | FastAPI | Async-native, production-grade |
| Classifier | scikit-learn or small fine-tuned model | Lightweight complexity scoring |
| Eval | Custom scoring + LLM-as-judge | Quality verification loop |
| Logging | SQLite + structured JSON logs | Full audit trail per request |
| Dashboard | Streamlit or Grafana | Cost and quality visualization |
| Containerization | Docker + docker-compose | Multi-service orchestration |

## Step-by-step build guide

### Phase 1 — Unified model interface (Day 1–3)

1. **Create a model registry.** Define a `ModelConfig` dataclass with: provider
   name, model ID, cost per input token, cost per output token, average latency,
   and a quality tier (high/medium/low). Populate it with real pricing for
   GPT-4o, GPT-4o-mini, Claude Sonnet, Claude Haiku, and a local Llama model via
   Ollama.
2. **Build the abstraction layer.** Write a single `send_request(prompt,
   model_config)` function that handles the provider-specific API calls behind a
   unified interface. Every call returns a standardized `Response` object with:
   output text, tokens used (input + output), latency, cost, and the model ID.
3. **Test every provider.** Send the same 10 prompts to every model in the
   registry. Log the outputs, costs, and latencies. This gives baseline data for
   the routing logic and validates the abstraction layer works.

### Phase 2 — Complexity classifier (Day 3–6)

1. **Define complexity tiers.** Three tiers:
   - **Tier 1 (simple):** reformatting, extraction, basic Q&A from provided context.
   - **Tier 2 (moderate):** summarization, classification, structured analysis.
   - **Tier 3 (complex):** multi-step reasoning, creative generation, nuanced judgment calls.
2. **Build a labeled dataset.** Write 200+ example prompts across all three
   tiers. Label each by hand. Include features to extract: token count, presence
   of instructions like "analyze" or "compare," number of constraints, whether
   context is provided, output-format complexity.
3. **Train the classifier.** Start with a simple scikit-learn model (logistic
   regression or random forest) on the extracted features. Not optimizing for
   classifier perfection — building the routing skeleton. Track accuracy and
   confusion matrix. Anything above 80% accuracy on a held-out set is fine for V1.
4. **Create the routing map.** Map each tier to a model. Tier 1 → cheapest
   (Haiku or local Llama). Tier 2 → mid-tier (GPT-4o-mini or Sonnet). Tier 3 →
   highest quality (GPT-4o or Opus). Store as a configurable YAML so models swap
   without code changes.

### Phase 3 — Async quality verification loop (Day 6–9)

1. **Define quality thresholds per use case.** For each request type, define
   what "good enough" means. Extraction: did it get all key fields?
   Summarization: LLM-as-judge score above 4/5. Classification: does the label
   match what GPT-4o would have said?
2. **Build the async verifier.** After the response is returned to the user,
   queue an async job that sends the same prompt to the highest-tier model and
   compares outputs. Score the agreement. If the cheap model's output diverges
   significantly, log it as a routing failure.
3. **Implement auto-escalation.** If the verifier catches a failure,
   automatically re-run the request with the higher-tier model and return the
   better result (if latency permits). Log the escalation event: original model,
   escalated model, cost delta, and the quality gap that triggered it.
4. **Feed failures back to the classifier.** Every routing failure becomes a new
   training example. Build a simple feedback loop that retrains the classifier
   weekly using accumulated failure data. This is the flywheel that makes the
   system get smarter over time.

### Phase 4 — Logging and cost dashboard (Day 9–11)

1. **Log everything.** Every request gets a row with: timestamp, prompt hash,
   complexity tier, routed model, cost, latency, verifier quality score, and
   whether it was escalated. This is the audit trail.
2. **Build the cost dashboard.** Show: total cost per day/week vs. what it would
   have cost using GPT-4o for everything ("you saved $X"), routing distribution
   (pie chart of which models handle what percentage), quality-score
   distribution, and escalation rate over time.
3. **Add the money-shot metric.** Calculate and prominently display the cost
   reduction percentage. If routing to cheaper models saved 60% vs. sending
   everything to the most expensive model, that number is the headline of the
   portfolio piece.

### Phase 5 — Expose as an API (Day 11–13)

1. **Build the FastAPI service.** A single `POST /v1/completions` endpoint that
   accepts a standard chat-completion request. The user doesn't choose the
   model — the router does. Return the response with metadata showing which
   model was selected and why.
2. **Add configuration endpoints.** `GET /v1/models` (list available models and
   costs), `GET /v1/stats` (cost-savings summary), `PUT /v1/routing-config`
   (update tier-to-model mappings without redeploying).
3. **Containerize and document.** docker-compose with the API service, a
   background worker for async verification, and the SQLite database. Write a
   README with architecture diagram, setup instructions, and the cost-savings
   number front and center.

### Phase 6 — Polish for portfolio (Day 13–14)

1. **Run a realistic load test.** Send 500–1,000 diverse prompts through the
   system. Generate the final cost-savings report. Screenshot the dashboard.
   These artifacts go in the portfolio.
2. **Write the case study.** Frame it as: "I built a system that reduced LLM API
   costs by X% while maintaining Y% quality parity." Lead with the number.
   Explain the routing logic. Show the feedback loop. This is a story a VP of
   Engineering immediately understands.
