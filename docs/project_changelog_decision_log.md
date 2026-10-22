# LAYA Agent Research Project — Changelog & Decision Record (The Past)

All major architectural evolutions, resolved design conflicts, and methodological pivots are recorded here with their technical rationale.

---

## [v0.5.0] — 2026-10-22: Infrastructure, Runtime Schemas & Final Parameter Lock
### Decisions Locked
* **Hosted LLM Provider:** Locked **NVIDIA Build API** (`integrate.api.nvidia.com/v1`) via the OpenAI SDK. Set `nemotron-3.5-super-vl-120b-a12b` as the primary Big Tier model and made both `BIG_TIER_MODEL` and `SMALL_TIER_MODEL` configurable via environment/YAML variables so models can be swapped cleanly.
* **LAYA Runtime Mechanics:** Inspected `NandhaKishorM/laya` and resolved the apparent mismatch between the earlier benchmark preset (`OBSERVE`, `FIND`, `INSPECT`, `ACT`, `VERIFY`) and the 5 architectural gates. Confirmed LAYA is a non-autoregressive `ModernBERT-large` engine that scores arbitrary runtime schemas using `choice`, `score`, and `noul` primitives in a single forward pass.
* **Context Firewall Chunking (Option A):** Because the English `ModernBERT-large` checkpoint has a $512$-token limit ($192$ tokens for criteria, $\sim 320$ tokens for state), we locked a **300-token sliding chunker** inside `firewall.py` and restricted all criteria strings to $< 15$ words.
* **Local Retrieval Stack:** Locked in-memory **`FAISS`** + **`BAAI/bge-small-en-v1.5`** embeddings (`sentence-transformers`) for the 40-rule domain policy store.
* **Benchmark Scope & Attack Ratio:** Locked **all 4 AgentDojo suites** (`workspace`, `banking`, `slack`, `travel`) with a **1:2 benign-to-injection ratio** per user task (25 Dev user tasks = 75 runs/arm; 50 Test user tasks = 150 runs/arm; 600 test runs across Arms A–D).

### Added
* GitHub repository synchronization policy: all gate changes, baseline scripts, and run configs are pushed simultaneously per phase.
* Commit hash logging in per-step JSONL trace schema for immutable run verification.

---

## [v0.4.0] — 2026-10-20: Rejection of Scope Cut & Evaluation Redesign
### Context & Conflict
* Build AI proposed cutting the Scorer, Filter, Router, and Supervisor gates to write a firewall-only paper, arguing that 4 separate benchmarks ($\tau$-bench, BFCL, AgentDojo, GAIA) across 70 tasks $\times$ 3 seeds lacked statistical power and that a 70B LLM could not fit on a Colab T4 GPU.
### Resolution & Rationale
* **Rejected Scope Cut:** Kept all 5 LAYA gates. We do not build throwaway prototypes or delete modules; all gates live in one unified pipeline controlled by `configs/arms.yaml`.
* **Clarified Compute Split:** Corrected the hardware misconception—Colab T4 (15 GB VRAM) runs **only** the ~846 MB local LAYA gate model and `bge-small-en-v1.5` embedder. Hosted LLMs run remotely via API.
* **Eliminated Pseudo-Replication:** Dropped $70\text{ tasks} \times 3\text{ identical seeds}$ in favor of deterministic `temperature=0` runs across a paired **25 Dev / 50 Test user-task split** (with 1 benign + 2 matched injection runs per task) and cluster bootstrap intervals. Added a duplicate Arm A test run to measure the API non-determinism floor.
* **Unified Benchmark + Policy Store:** Standardized on AgentDojo's 4 tool suites paired with a local 40-rule domain policy store so every task exercises retrieval, filtering, routing, firewalling, and supervision in one environment.
* **Explicit Firing Rate Logging:** Agreed to log per-gate firing rates so low trigger frequencies on short trajectories (e.g., Router or Supervisor) are transparently reported as empirical findings rather than hidden.

---

## [v0.3.0] — 2026-10-17: Handoff Merge & Resolution of 7 Core Conflicts
### Decisions Locked
1. **Cumulative Experiment Arms:** Confirmed Arms A $\to$ B $\to$ C $\to$ D are cumulative (`A` = baseline, `B` = A + Firewall + Supervisor, `C` = B + Scorer, `D` = C + Filter + Router), eliminating the need for a separate Arm E while adding baseline config flags (`B_regex`, `C_always_small`, `C_random`).
2. **Firewall Fallback Policy:** Differentiated utility fallback from security fallback. Utility decisions (Scorer, Filter, Router, Supervisor, and Firewall relevance) **fail open** on low confidence. Firewall prompt-injection checks (`noul` primitive) **redact** chunks exceeding $\tau_{\text{inj}}$ with a standardized placeholder string (`[FIREWALL: Tool output redacted due to suspected injection]`) so automated runs continue safely without human intervention.
3. **Dropped Jev & Speculative Pre-Run:** Removed the closed-source TypeSafe Jev dependency and Idea #2 (Speculative pre-run of read-only tools) from the active experimental arms to keep the architecture 100% reproducible on open-weight local LAYA.

---

## [v0.2.0] — 2026-10-14: Colab T4 Gate Latency Profiling ($N=1000$)
### Empirical Findings
* Executed a 1,000-call per-gate latency benchmark on a Tesla T4 GPU (`device="cuda"`), excluding model initialization, first prediction, and 10 warm-up calls.
* Measured flat latency across input state buckets ($\le 35$ words, $36\text{–}80$ words, $> 80$ words):
  * $p_{50} \approx 35.3\text{ ms}$
  * $p_{95} \approx 51.5\text{ ms}$
  * $p_{99} \approx 58.0\text{ ms}$
### Architectural Impact
* Voided earlier $70\text{–}500\text{ ms}$ network proxy estimates.
* Revised Supervisor execution frequency from "every 2–3 steps" to **every step** because a ~35 ms local GPU check adds negligible overhead relative to $0.5\text{–}3.0\text{ s}$ hosted LLM calls.

---

## [v0.1.0] — 2026-10-09: Initial Ideation & Baseline Mapping
* Extrapolated 7 candidate gate locations around a vector-policy ReAct agent.
* Decided to keep Query Embedding + Hybrid/FAISS Search intact and replace only the Rerank + Top-$K$ stage with the LAYA Applicability Filter (`[D]`).
* Dropped Idea #5 (Branch & Prune tree search) immediately due to multiplicative LLM token cost.