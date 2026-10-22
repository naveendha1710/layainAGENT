# LAYA Agent Research Project — Requirements Specification (The Future)

## 1. Project Vision & Desired End-State
The objective of this project is to build, benchmark, and publish **LAYA**, a sub-40 ms confidence-gated modular control layer wrapped around a ReAct-style tool-using LLM agent backed by a local vector policy store. 

The system follows a **"build once, toggle via config"** philosophy: a single unified codebase where five non-autoregressive decision gates (**Difficulty Scorer**, **Applicability Filter**, **Action Router**, **Context Firewall**, and **Supervisor**) are toggled via `configs/arms.yaml`. The desired end-state is a fully reproducible empirical evaluation across 600 core test runs on **AgentDojo**, proving that local confidence-gated control reduces hosted API cost per task, bounds latency overhead, and defends against indirect prompt injections without degrading benign task utility—culminating in an open-science release (code, immutable JSONL traces, and preprint manuscript).

---

## 2. Core Research Hypotheses
* **H1 — Safety (Primary Pre-Registered Claim):** The Context Firewall (`[B]`) significantly reduces Attack Success Rate (ASR) under indirect prompt injection while maintaining a low false-block rate on benign tool outputs.
* **H2 — Efficiency:** The Difficulty Scorer (`[C]`), Applicability Filter (`[D]`), and Action Router (`[D]`) reduce total input/output tokens, hosted LLM calls, and dollar cost per task while maintaining task utility comparable to the unaugmented baseline.
* **H3 — Reliability:** The Supervisor gate (`[B]`) detects and recovers from repetitive tool loops and trajectory drift, bounding wasted execution steps.
* **H4 — Bounded Overhead:** Running local non-autoregressive LAYA gates on GPU introduces negligible step latency ($p_{50} \approx 35\text{ ms}$, $p_{99} < 75\text{ ms}$) that is amortized by avoided LLM calls and pruned context tokens.

---

## 3. Researcher & System Stories
* **US-1 (Single Unified Codebase):** As a systems researcher, I want all five LAYA gates and baseline comparison modes built as modular hooks inside one ReAct loop and controlled via `configs/arms.yaml`, so that I never write throwaway code or maintain separate branches per experiment arm.
* **US-2 (Model-Agnostic Hosted Tiering):** As an experimenter running on NVIDIA Build APIs, I want the Big Tier (`BIG_TIER_MODEL`) and Small Tier (`SMALL_TIER_MODEL`) injected via environment/config variables with automated rate-limit backoff and token metering, so I can swap underlying hosted models without touching agent code.
* **US-3 (Non-Destructive Security Redaction):** As an evaluation designer operating without a human in the loop, I want suspected prompt injections in tool outputs to be automatically replaced with a deterministic redaction warning string rather than crashing the run, so the agent can attempt recovery and we can measure both Benign Utility and Utility Under Attack.
* **US-4 (Leakage-Free Calibration):** As an empirical author, I want threshold tuning ($\tau$) and criteria prompt refinement strictly restricted to a 25-user-task Dev split, keeping the 50-user-task Test split and the 40-rule FAISS policy store completely frozen during final evaluation.

---

## 4. Functional Requirements (Feature List)

### FR-1: Config-Driven Ablation Matrix (`configs/arms.yaml`)
The harness must support toggling gates and baseline modes purely through configuration:
* **Arm A (Baseline):** All gates `off`. Standard ReAct agent + FAISS Top-$K$ policy retrieval using `BIG_TIER_MODEL`.
* **Arm B\_regex (Heuristic Security Baseline):** Firewall set to `regex`, all other gates `off`.
* **Arm B (LAYA Safety & Control):** Context Firewall (`laya`) + Supervisor (`laya`) enabled.
* **Arm C\_always\_small (Static Cost Baseline):** Arm B + Scorer set to `always_small`.
* **Arm C\_random (Random Routing Baseline):** Arm B + Scorer set to `random`.
* **Arm C (LAYA Cost Routing):** Arm B + Scorer (`laya`) dynamically routing between `SMALL_TIER_MODEL` and `BIG_TIER_MODEL`.
* **Arm D (Full LAYA Architecture):** Arm C + Applicability Filter (`laya`) + Action Router (`laya`).

### FR-2: Dual-Tier Hosted LLM Client (`agent/llm/`)
* OpenAI-compatible client targeting NVIDIA Build (`https://integrate.api.nvidia.com/v1`).
* Default Big Tier: `nemotron-3.5-super-vl-120b-a12b` (overridable via config/env).
* Configurable Small Tier: 7B/8B instruction model on NVIDIA Build (overridable via config/env).
* Built-in concurrency semaphore, exponential backoff on HTTP 429/5xx, exact input/output token counting, and frozen price-sheet cost calculation.

### FR-3: Read-Only Vector Policy Store (`agent/retrieval/` & `policies/`)
* Local in-memory `FAISS` index populated at startup using `BAAI/bge-small-en-v1.5` embeddings via `sentence-transformers`.
* Contains 40 domain and security rules tailored to AgentDojo's four suites (`workspace`, `banking`, `slack`, `travel`).
* Strictly read-only after initialization to prevent test-set leakage.

### FR-4: Five LAYA Decision Gates (`agent/gates/`)
All gates execute locally on GPU via `laya.Router` (`ModernBERT-large` English checkpoint) using concise criteria descriptions ($< 15$ words per option):
1. **Scorer Gate (`scorer.py`):** Evaluates initial task prompt (`choice` primitive) to select model tier (`small` vs. `big`) and step budget. Fails open to `big` + full budget when $\text{confidence} < \tau_{\text{scorer}}$.
2. **Applicability Filter Gate (`filter.py`):** Evaluates retrieved FAISS policy candidates (`choice` primitive: `keep` vs. `drop`). Fails open to `keep` when $\text{confidence} < \tau_{\text{filter}}$.
3. **Action Router Gate (`router.py`):** Evaluates state before LLM invocation (`choice` primitive: `bypass_llm` vs. `call_llm`) for deterministic follow-ups. Fails open to `call_llm` when $\text{confidence} < \tau_{\text{router}}$.
4. **Context Firewall Gate (`firewall.py`):** Splits raw tool outputs using a **300-token sliding chunker** and evaluates each chunk in a single forward pass:
   * *Injection check (`noul` primitive):* If $P(\text{injection}) > \tau_{\text{inj}}$ on any chunk, redacts the chunk with `[FIREWALL: Tool output redacted due to suspected injection]`.
   * *Relevance check (`choice` primitive):* Drops chunks classified as `low_relevance` when $\text{confidence} \ge \tau_{\text{rel}}$; fails open (`keep`) on low confidence.
5. **Supervisor Gate (`supervisor.py`):** Evaluates post-step trajectory (`choice` primitive: `continue`, `nudge`, `stop`). Fails open to `continue` when $\text{confidence} < \tau_{\text{supervisor}}$.

### FR-5: Evaluation, Splits & Telemetry Harness (`eval/`, `tracing/`, `analysis/`)
* **Dataset:** AgentDojo across all 4 suites (`workspace`, `banking`, `slack`, `travel`).
* **User-Task Split:** 25 Dev user tasks and 50 Test user tasks (benign runs and their attacked twins remain in the same split).
* **Pairing Ratio (1:2):** Every user task runs under 1 benign condition and 2 matched indirect prompt-injection conditions (75 Dev runs/arm; 150 Test runs/arm).
* **Step-Level Telemetry (`tracing/trace.py`):** Writes one validated JSONL line per step with all required schema fields.

---

## 5. Non-Functional & Statistical Requirements
* **Determinism:** All evaluations run with pinned seeds and `temperature=0`. Arm A is executed twice on the test split to establish the empirical API non-determinism floor.
* **Timing Precision:** All GPU gate latency measurements must call `torch.cuda.synchronize()` before stopping `time.perf_counter()`.
* **Statistical Analysis:** Paired comparisons across arms must use **cluster bootstrap over user tasks** ($95\%$ confidence intervals) to account for correlation between a user task's benign and injected runs.

---

## 6. Continuous Version Control & GitHub Sync

### Simultaneous Tracking
Every configuration change (`configs/arms.yaml`), schema update, and code iteration must be committed and pushed immediately to GitHub.

### Trace & Run Versioning
Benchmark execution runs must pin and log the exact `git commit hash` inside trace metadata (`tracing/trace.py`) prior to starting, ensuring strict reproducibility.

### Artifact Quarantine
Raw JSONL traces must be staged to Git LFS or synced to release tags simultaneously as arms finish execution.