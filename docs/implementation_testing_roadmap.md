# LAYA Agent Research Project — Implementation & Testing Roadmap (The Plan)

This roadmap bridges the current locked design (`architecture.md`) to the completed empirical study (`requirements.md`). Every milestone builds directly into the final repository without throwaway code and enforces a strict pass/fail verification gate before proceeding.

---

## Milestone 0: Repository Skeleton, Config & Telemetry Contract
**Goal:** Establish the directory structure, configuration loader, and validated step-level JSONL logger.

### Tasks
1. Create the `laya_agent/` directory hierarchy and `requirements.txt` (`laya`, `openai`, `faiss-cpu`, `sentence-transformers`, `pydantic`, `pyyaml`, `pytest`, `agentdojo`).
2. Write `configs/arms.yaml` containing global settings (`temperature: 0`, environment variable hooks for `BIG_TIER_MODEL` and `SMALL_TIER_MODEL`, token pricing placeholders, `dev: 25` / `test: 50` split definitions) and the arm definitions (`A`, `B_regex`, `B`, `C_always_small`, `C_random`, `C`, `D`).
3. Implement `tracing/trace.py` using Pydantic/dataclasses to enforce the 18-field JSONL schema and atomic file appending.
4. Implement `agent/gates/base.py` defining `Decision(str, Enum)`, `GateResult`, and the `Gate` base class with timing, confidence comparison ($\text{conf} < \tau$), and exception-safe fallback handling.
5. **Pre-Run Checkpoint:** Commit the working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
6. **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

### Verification Gate 0
* **Test:** Run `pytest tests/test_trace_and_base.py` simulating 10 dummy steps and triggering a forced provider exception inside a mock gate.
* **Pass Criteria:** 100% of records validate against the 18-field schema; the forced exception cleanly sets `fallback_used=True` and returns `gate.default(ctx)` without crashing.

### Milestone Checkpoint Rule (Every Phase)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 1: Dual-Tier LLM Client, Policy Store & Base Agent (Arm A)
**Goal:** Stand up the unaugmented ReAct baseline agent (`Arm A`), the NVIDIA Build API client, and the frozen 40-rule FAISS policy store.

### Tasks
1. Implement `agent/llm/client.py` and `agent/llm/tiers.py` for NVIDIA Build (`https://integrate.api.nvidia.com/v1`), including exponential backoff for rate limits, latency timing (`llm_latency_ms`), and token/cost accounting.
2. Author `policies/policies.yaml` (40 domain & security policies across `workspace`, `banking`, `slack`, `travel`), `policies/annotations.jsonl`, and `policies/checks.py`.
3. Implement `agent/retrieval/embed.py` (`BAAI/bge-small-en-v1.5`) and `agent/retrieval/store.py` (in-memory `FAISS` index with a `.freeze()` lock).
4. Implement `agent/core/state.py` and `agent/core/loop.py` (the unified ReAct loop with no-op pass-through hooks when gates are set to `off`).
5. Implement `eval/adapters/agentdojo.py` and `eval/splits.py` to generate the frozen `splits.json` (25 Dev / 50 Test user tasks, 1 benign + 2 matched injections per task).

### Verification Gate 1
* **Test:** Run a 5-task smoke test of `Arm A` on the Dev split using NVIDIA Build (`nemotron-3.5-super-vl-120b-a12b`).
* **Pass Criteria:** Agent executes tools end-to-end, retrieves policies from FAISS, writes complete JSONL step traces, and records non-zero token counts and API latencies with zero unhandled exceptions.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 2: LAYA CUDA Provider & Long-Input Latency Sweep
**Goal:** Integrate the real `laya.Router` runtime on Colab T4 GPU and verify the 300-token chunker latency across long tool outputs.

### Tasks
1. Implement `agent/providers/laya.py` wrapping `laya.Router` on CUDA, ensuring `torch.cuda.synchronize()` is called immediately before stopping the latency timer.
2. Implement `agent/providers/mock.py` for offline CI/unit testing.
3. Write the 300-token sliding window chunker utility inside `agent/gates/firewall.py`.
4. Implement `scripts/profile_latency.py` to benchmark `firewall.py` on realistic tool payloads of $100$, $500$, $1000$, and $2000$ tokens ($N=200$ calls per bucket on Colab T4).

### Verification Gate 2
* **Test:** Execute `python scripts/profile_latency.py` on Colab T4 GPU.
* **Pass Criteria:** Per-chunk latency matches the baseline profile ($p_{50} \approx 35\text{–}40\text{ ms}$, $p_{95} < 65\text{ ms}$) with zero silent token truncation in the `ModernBERT-large` context window.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 3: Safety & Control Gates (Arm B & Arm B\_regex)
**Goal:** Implement the Context Firewall (`[B]`) and Supervisor (`[B]`) gates to defend against indirect prompt injections and execution loops.

### Tasks
1. Complete `agent/gates/firewall.py` supporting `mode: laya` (evaluating `FIREWALL_SCHEMA` with `noul` injection check and `choice` relevance check across 300-token chunks) and `mode: regex` (heuristic pattern baseline).
2. Implement the automated redaction rule: if any chunk scores $P(\text{injection}) > \tau_{\text{inj}}$, replace that chunk with `[FIREWALL: Tool output redacted due to suspected injection]` and log the decision.
3. Implement `agent/gates/supervisor.py` evaluating `SUPERVISOR_SCHEMA` (`continue`, `nudge`, `stop`) after every tool observation step.

### Verification Gate 3
* **Test:** Run unit tests (`tests/test_arm_b.py`) plus a 6-run Dev mini-batch (2 benign + 4 injected tasks) comparing `Arm A`, `Arm B_regex`, and `Arm B`.
* **Pass Criteria:** Known injection payloads trigger redaction in `Arm B`, redacted strings appear cleanly in the agent state without crashing the loop, and `ground_truth_label` accurately logs whether the chunk contained an injection.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 4: Cost & Efficiency Gates (Arms C & D)
**Goal:** Complete the 5-gate LAYA architecture by implementing the Scorer (`[C]`), Applicability Filter (`[D]`), and Action Router (`[D]`).

### Tasks
1. Implement `agent/gates/scorer.py` supporting `laya` (`SCORER_SCHEMA`), `always_small`, `always_big`, and `random` modes to set `state.model_tier` and `state.max_steps` before the loop starts.
2. Implement `agent/gates/filter.py` evaluating `FILTER_SCHEMA` (`keep` vs. `drop`) over the FAISS candidate policies, failing open (`keep`) when $\text{confidence} < \tau_{\text{filter}}$.
3. Implement `agent/gates/router.py` evaluating `ROUTER_SCHEMA` (`bypass_llm` vs. `call_llm`) before each controller step.
4. Implement `scripts/estimate_cost.py` to project total tokens, API calls, and runtime across the Dev and Test suites.

### Verification Gate 4
* **Test:** Run all 7 configured arms (`A`, `B_regex`, `B`, `C_always_small`, `C_random`, `C`, `D`) across 3 Dev user tasks (9 runs per arm) end-to-end.
* **Pass Criteria:** Every arm executes via `eval/runner.py` solely by changing the `--arm` CLI argument; trace logs confirm all 5 gates fire and log valid decisions in `Arm D`.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 5: Dev Split Calibration & Pre-Registration Freeze
**Goal:** Tune confidence thresholds ($\tau$) and criteria wording on the 25 Dev user tasks, then freeze the protocol before touching the Test split.

### Tasks
1. Run the full **Dev Split** (25 user tasks $\times$ 3 conditions = 75 runs per arm) across Arms A, B, C, and D.
2. Analyze Dev traces to calibrate $\tau_{\text{scorer}}$, $\tau_{\text{filter}}$, $\tau_{\text{router}}$, $\tau_{\text{inj}}$, $\tau_{\text{rel}}$, and $\tau_{\text{supervisor}}$ (balancing false-block rate on benign Dev runs against injection recall and token savings).
3. Lock the tuned thresholds inside `configs/arms.yaml`.
4. Complete `prereg/PREREGISTRATION.md` specifying the exact primary and exploratory claims, frozen thresholds, model IDs, and Git commit hash.

### Verification Gate 5
* **Test:** Audit `configs/arms.yaml` and `prereg/PREREGISTRATION.md`.
* **Pass Criteria:** Zero `null` values remain in `configs/arms.yaml`; Git working tree is clean and tagged `v1.0-prereg-frozen` prior to any Test split execution.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 6: Full Test Matrix Execution
**Goal:** Generate the immutable empirical dataset across the frozen 50-user-task Test split.

### Tasks
1. Execute **Arm A (Run 1 & Run 2)** on the Test split ($150 \times 2 = 300\text{ runs}$) at `temperature=0` to establish the baseline metrics and the hosted API non-determinism floor.
2. Execute **Arm B\_regex** and **Arm B** ($150\text{ runs each}$).
3. Execute **Arm C\_always\_small**, **Arm C\_random**, and **Arm C** ($150\text{ runs each}$).
4. Execute **Arm D (Full LAYA)** ($150\text{ runs}$).

### Verification Gate 6
* **Test:** Run log integrity verifier over `logs/test/`.
* **Pass Criteria:** All $150$ paired task trajectories per evaluated arm are complete, valid JSONL files with zero missing tasks, zero threshold modifications, and verified model version strings.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).

---

## Milestone 7: Statistical Analysis & Open-Science Release
**Goal:** Compute cluster-bootstrapped intervals, generate publication tables, and package the paper and artifacts for Zenodo/arXiv.

### Tasks
1. Run `analysis/metrics.py` and `analysis/bootstrap.py` (10,000 bootstrap resamples clustered by `user_task_id`) to compute $95\%$ confidence intervals for:
   * Attack Success Rate (ASR) & False-Block Rate
   * Benign Utility & Utility Under Attack
   * Input/Output Tokens, LLM Calls, and Cost per Task ($\$$)
   * Per-Gate Firing Rate, Fallback Rate, Precision/Recall, and Latency ($p_{50}, p_{95}, p_{99}$)
2. Run `analysis/report.py` to output formatted Markdown and LaTeX tables.
3. Draft the empirical systems paper reporting the primary pre-registered claim (Firewall safety/utility trade-off) and exploratory full-pipeline efficiency results, then archive the codebase, frozen JSONL logs, and PDF on Zenodo/arXiv.

### Verification Gate 7
* **Pass Criteria:** Every number in the manuscript tables is generated directly by `analysis/report.py` from the immutable `logs/test/` JSONL files with one-command reproducibility.

### Standard Execution Rule (After Every Milestone)
* **Pre-Run Checkpoint:** Commit working tree before launching any dev/test batch (`git commit -m "chore(eval): lock arm <X> config"`).
* **Post-Run Sync:** Immediately commit frozen JSONL logs and push to remote (`git push origin <branch>`).