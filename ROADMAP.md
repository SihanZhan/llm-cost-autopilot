# Roadmap

Six phases, ~2 weeks. Each phase produces a concrete artifact.

## Phase 1 — Unified model interface

- [x] `ModelConfig` dataclass: provider, model ID, cost per input/output token, avg latency, quality tier
- [x] `MODEL_REGISTRY` populated with GPT-4o, GPT-4o-mini, Claude Sonnet, Claude Haiku, local Llama (Ollama)
- [x] Real pricing verified against provider docs (OpenAI + Anthropic first-party rates, Sep 2026)
- [x] `send_request(prompt, model_config)` returns a standardized `Response` (output text, input/output tokens, latency, cost, model ID) for all three providers
- [x] Credentials read from environment, not literals
- [x] Error handling + retries per provider (`LLMRequestError`, exponential backoff, fast-fail on config errors)
- [x] Baseline harness: `baseline.py` runs the 10-prompt set through every model -> `baseline_results.csv` + summary
- [ ] Run the full baseline against live providers and commit the numbers / replace registry `avg_latency` placeholders

## Phase 2 — Complexity classifier

- [ ] Define tiers: T1 simple (reformat, extract, basic Q&A) / T2 moderate (summarize, classify, structured analysis) / T3 complex (multi-step reasoning, creative, judgment)
- [ ] Hand-label 200+ example prompts across tiers
- [ ] Feature extraction: token count, instruction verbs, constraint count, context present, output-format complexity
- [ ] Train scikit-learn model (logistic regression / random forest); track accuracy + confusion matrix; target >80% held-out
- [ ] `routing.yaml`: tier → model mapping, swappable without code changes

## Phase 3 — Async quality verification loop

- [ ] Per-use-case quality thresholds (extraction field coverage, summary LLM-judge ≥ 4/5, classification label match vs. top tier)
- [ ] Async verifier: re-run prompt against top-tier model, score agreement, log divergence as routing failure
- [ ] Auto-escalation: on failure, re-run with higher tier and return the better result; log original model, escalated model, cost delta, quality gap
- [ ] Feedback: each routing failure becomes a classifier training example; weekly retrain on accumulated failures

## Phase 4 — Logging and cost dashboard

- [ ] Per-request row: timestamp, prompt hash, complexity tier, routed model, cost, latency, verifier quality score, escalated?
- [ ] Streamlit dashboard: daily/weekly cost vs. "all GPT-4o" baseline, routing distribution, quality-score distribution, escalation rate over time
- [ ] Headline metric: cost reduction % prominently displayed

## Phase 5 — Expose as an API

- [ ] `POST /v1/completions` — standard chat-completion request; router picks the model; response includes which model and why
- [ ] `GET /v1/models` — available models and costs
- [ ] `GET /v1/stats` — cost-savings summary
- [ ] `PUT /v1/routing-config` — update tier→model mappings without redeploy
- [ ] docker-compose: API service + async verification worker + SQLite

## Phase 6 — Portfolio polish

- [ ] Load test: 500–1,000 diverse prompts through the system
- [ ] Final cost-savings report + dashboard screenshots
- [ ] Case study: "reduced LLM API costs by X% while maintaining Y% quality parity" — lead with the number, explain routing logic, show the feedback loop
- [ ] README architecture diagram + setup instructions with the savings number front and center
