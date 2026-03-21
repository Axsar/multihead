# Research Synthesis: K-Step Reasoning, MAKER, and Iterative Test Generation

**Date**: 2026-02-21
**Purpose**: Synthesize research on multi-step reasoning patterns to inform MultiHead's task decomposition architecture
**Research Tasks**: #155 (K-step), #156 (MAKER), #157 (kS-LLM)

---

## Executive Summary

Three parallel research efforts reveal **converging architectural patterns** for reliable multi-step LLM systems:

1. **K-Step Reasoning**: CoT → ReAct → Self-Consistency → Process Reward Models → Reflection loops
2. **MAKER/MDAP**: Zero-error million-step execution via maximal decomposition (m=1) + first-to-ahead-by-k voting
3. **Iterative Test Generation**: Universal refinement loop (Generate → Execute → Analyze → Feedback → Refine → Cache → Converge)

**Core Insight**: All three domains independently discovered that **atomic step decomposition + per-step verification + iterative refinement** enables reliable long-horizon task execution.

**MultiHead Status**: Already has ~60% of necessary infrastructure (DAG executor, consensus, event-sourcing, routing). Missing: step-level validators, reflection loops, auto-decomposition.

---

## Part 1: K-Step Reasoning Patterns

### 1.1 Core Techniques

#### Chain-of-Thought (CoT)
- **Definition**: Generate explicit reasoning traces before final answer
- **Breakthrough**: "Showing your work" dramatically improves complex reasoning
- **Emergence**: Requires ~100B+ parameters
- **Modern context**: Value decreasing with o1/o3-style models that internalize reasoning

#### ReAct (Reasoning + Acting)
- **Innovation**: Interleaves reasoning with tool execution
- **Loop structure**:
  1. **Thought**: Why am I taking this action?
  2. **Action**: Execute tool/API call
  3. **Observation**: System returns result
  4. **Repeat**: Until task complete
- **Performance**: +34% on ALFWorld, +10% on WebShop vs baselines

#### Self-Consistency Voting
- **Mechanism**: Generate N diverse reasoning paths (high temperature) → aggregate via majority vote
- **Enhancements**:
  - Confidence-weighted voting (not just count)
  - Ranked voting methods (IRV, Borda count, MRR)
  - Optimal sampling strategies
- **Trade-off**: N fixed samples inefficient (wasteful on easy problems, insufficient on hard ones)

#### Process Reward Models (PRMs)
- **Critical distinction**: Step-level feedback vs outcome-only validation
- **Advantages over ORMs**:
  - Precise credit assignment (identify *which* step failed)
  - Improved interpretability
  - Better test-time scaling (score candidate solutions at each step)
  - Superior empirical results
- **Challenge**: High annotation cost for step-level labels

#### Reflexion Framework
**Components**:
1. **Actor**: Generates initial attempt
2. **Evaluator**: Scores output quality
3. **Self-Reflection**: Produces verbal reinforcement cues
4. **Memory**: Stores reflections for future attempts

**Results**: Statistically significant improvement (p < 0.001) via self-reflection loop

**Pattern**:
```
Loop:
  Initial attempt → Evaluate → Reflect on failures → Refine → Re-attempt
```

### 1.2 Task Decomposition Strategies

#### LLM-Based Decomposition
- Use the LLM itself to break down complex tasks
- **DecomP** (Decomposed Prompting): Modular approach delegating subtasks to specialized handlers
- **ADaPT** (As-Needed Decomposition): Recursive decomposition based on complexity and capability
- **Hierarchical**: Coordinator decomposes high-level goals, delegates to specialized subagents

#### Planning Frameworks

**ReWOO (Reasoning Without Observation)**:
- Planner creates full tool itinerary upfront
- Worker executes tools sequentially
- Solver synthesizes final answer
- **Trade-off**: Less adaptive than ReAct, but faster for predictable sequences

**Tree-of-Thoughts (ToT)**:
- Maintains tree of intermediate reasoning states
- LLM evaluates progress at each node
- Combines with search (BFS/DFS) for systematic exploration
- Enables lookahead and backtracking
- **LATS variant**: Combines ToT + ReAct + planning

**Plan-and-Execute**:
- Phase 1: Generate complete plan
- Phase 2: Execute steps sequentially
- Enables upfront validation and resource estimation

### 1.3 Verification Between Steps

#### Contract-Based Verification (ToolGate)
- **Hoare-style contracts**: Each tool has preconditions + postconditions
- **Precondition**: Gates tool invocation (check current state satisfies requirements)
- **Postcondition**: Runtime verification before committing result
- **Guarantee**: Symbolic state evolves only through verified executions
- **Benefit**: Prevents invalid/hallucinated results from corrupting system state

#### Self-Verification Methods
- **Forward + Backward**: Generate candidates with CoT → verify by checking if conclusion implies premise
- **Verification CoT**: Validates each step for logical soundness and factual consistency
- **Step-wise validators**: Formal verifiers (code/math), PRMs (learned validators)

### 1.4 Reliability Principles

**Clear Contracts**:
- Purpose statement
- Crisp examples
- Precise argument types
- No ambiguity

**Input Validation**:
- Validate at boundaries
- Reject or auto-correct known invalid patterns
- Type checking and schema validation

**Structured Outputs**:
- Enforce specific formats (JSON, schemas)
- Enable reliable parsing
- Router agents use structured outputs for decision-making

**Explainability Requirements**:
- Before tool call: One-line reasoning + tool ID
- After tool call: Observation summary
- Maintain reasoning trace for debugging

**Monitoring**:
- Track latency, cost, output quality
- Real-time visibility
- Circuit breaker patterns for tool health

### 1.5 Retry and Refinement

**Retry Strategies**:
- Exponential backoff with jitter
- Circuit breaker (stop if consistently down)
- Error classification (retriable vs non-retriable)
- Adaptive retry based on error context

**Refinement Cycle** (Generate-Critique-Refine):
1. Create initial response
2. Reflect on quality (self-critique)
3. Iteratively improve based on feedback
4. Repeat for k iterations or convergence

**AgentDebug**:
- Decomposes trajectories into decision steps
- Isolates minimal root-cause failures
- Provides corrective feedback
- Iterative re-rollouts with actionable guidance

### 1.6 State Management

**DataStates-LLM Framework**:
- Explicit state representation
- Modular state providers
- Decouples semantic abstraction from computational mechanics

**Task Memory Engine (TME)**:
- Task Memory Tree (TMT): Hierarchical task structure
- Task Relationship Inference Module (TRIM): Infers dependencies
- Prompt Synthesizer: Builds context-aware prompts

**Checkpointing**:
- Lazy asynchronous checkpointing (non-blocking)
- Workflow checkpointing at stage boundaries
- **Saga Pattern**: Sequence of local transactions with compensating actions on failure

### 1.7 OpenAI Reasoning Models (o1, o3)

**Training**: RL to refine CoT strategies, recognize/correct mistakes, break steps into simpler ones

**Reasoning Tokens**:
- Separate from input/output
- Used for internal "thinking"
- Discarded after generating visible response
- Summary shown to users

**Test-Time Search (o3)**:
- Generates diverse candidate CoTs
- Explores multiple paths dynamically
- Selects best via evaluation

**Scaling**: Performance improves with both train-time (RL) and test-time (thinking time) compute

---

## Part 2: MAKER (Million-Step Reasoning)

### 2.1 Core Mechanisms

#### 2.1.1 Maximal Agentic Decomposition (MAD)

**Definition**: Break tasks into smallest possible atomic subtasks (m=1, where m = steps per subtask)

**Why it matters**:
- Creates modularity enabling effective error correction
- Prevents error propagation across boundaries
- Allows independent verification of each atomic action

**Mathematical scaling** (maximal decomposition):
```
E[cost] = Θ(p⁻¹ × c × s × ln s)
```
Where:
- `p` = per-step success probability
- `c` = per-token cost
- `s` = total steps
- `ln s` = logarithmic with steps (from k_min scaling)

**Critical insight**: Without maximal decomposition (m > 1), cost scales **exponentially** and becomes infeasible.

**Towers of Hanoi example**:
- 20 disks = 1,048,575 moves
- Each agent handles exactly ONE disk move
- Agent only knows: current state, which disk to move
- This extreme modularity is key to success

#### 2.1.2 First-to-Ahead-by-K Voting

**Algorithm**: Sample candidate actions sequentially until one achieves margin of **k votes** over all competitors

**Theoretical foundation**: Sequential Probability Ratio Test (SPRT), proven optimal by Wald and Wolfowitz

**Selection probability** (simplified, two candidates):
```
P(correct) = 1 / (1 + ((1-p)/p)^k)
```

**Key scaling law**: Minimal k required for overall success probability `t` across `s` steps:
```
k_min = ⌈ln(t^(-m/s) - 1) / ln((1-p)/p)⌉ = Θ(ln s)
```

**Critical result**: k grows **logarithmically** with total steps, not linearly!

**Example**: Million-step task with p ≈ 0.998 only needs **k=3** for 95% success

**Efficiency**:
- After first k rounds, exponential decrease in undecided steps
- Dynamic stopping (unlike fixed self-consistency sampling)
- Optimal sample efficiency for given reliability target

#### 2.1.3 Red-Flagging

**Purpose**: Identify and discard outputs with high error correlation, preventing simultaneous failures

**Detection signals**:
1. **Length threshold**: Outputs >700-750 tokens show error rates jumping from 0.1% to 10%
   - Long outputs indicate "confused logical loops"
2. **Format violations**: Malformed JSON, missing required fields
   - Structural errors correlate with semantic errors

**Effect**: Reduces "collisions across first two votes" substantially

**Not about accuracy**: Red-flagging doesn't improve base model performance—it removes *correlated* errors

#### 2.1.4 Error Rate Scaling

**Fundamental question**: How does required k_margin relate to total task length?

**Answer**: Logarithmically!

**Intuition**:
- Naive: 99% success per step → 0.99^1000000 ≈ 0 overall (guaranteed failure)
- MAKER: k=3 voting with p=0.998 → ~95% overall success

**Why logarithmic works**:
- Each voting round exponentially suppresses error probability
- k rounds = exponential^k suppression
- Need logarithmic k to counteract linear step accumulation

**Practical implication**: Million-step task doesn't need 1M times more error correction than thousand-step—only ~10x (ln 1000000 / ln 1000 ≈ 2)

### 2.2 When MAKER is Necessary vs Overkill

#### MAKER is **NECESSARY** when:
1. **Long task horizons**: >100-200 dependent sequential steps
2. **Zero-error tolerance**: Perfect execution required (code, proofs, transactions)
3. **Moderate per-step error rates**: 90-99.9% individual accuracy (sweet spot: 0.1%-10%)
4. **Structured, verifiable outputs**: Clear correctness criteria (JSON schemas, formal syntax)

#### MAKER is **OVERKILL** for:
1. **Short reasoning chains**: <50-100 steps
2. **Error-tolerant applications**: Creative writing, brainstorming, approximate solutions
3. **Non-decomposable tasks**: Holistic judgment, artistic decisions, global optimization
4. **Cost-sensitive, non-critical**: $3.5K for million-step Hanoi not justified for low-stakes

### 2.3 Cost and Latency Trade-offs

**Computational cost** (m=1):
```
E[cost] = Θ(p⁻¹ × c × s × ln s)
```

**Key insight**: Smaller models + voting beats larger models alone

| Model | Cost (20-disk Hanoi) | Strategy |
|-------|---------------------|----------|
| gpt-4o-mini + MAKER | $3,500 | Many cheap inferences |
| o1-mini | $9,000 | Expensive reasoning |
| o1 (full) | $71,000 | Most expensive |

**Latency bottleneck**: First-to-ahead-by-k is sequential
- Can't parallelize voting rounds
- Latency = avg samples per step × step count × inference time

**Mitigation**:
- Batch agent calls where dependencies allow
- Speculative execution (start N+1 while N votes complete)
- Adaptive k (lower for high-confidence steps)

### 2.4 Comparison to Other Approaches

#### vs Chain-of-Thought
- CoT: Single chain, errors propagate, fails beyond ~100 steps
- MAKER: Parallel microagents, voting at every step, proven to 1M+ steps

#### vs Self-Consistency
- Self-Consistency: Fixed N samples (wasteful), final answer voting only
- MAKER: Dynamic (first-to-ahead), per-step voting, optimal (SPRT-inspired)

#### vs Tree-of-Thoughts
- ToT: Explores alternative reasoning paths (breadth)
- MAKER: Ensures correct execution of single path (depth)
- ToT: Creative problem-solving, heuristic search, no guarantees
- MAKER: Deterministic execution, mathematical reliability bounds

**Key difference**: ToT explores *alternatives*, MAKER ensures *correctness*

---

## Part 3: Iterative Test Generation (kS-LLM Pattern)

### 3.1 Universal K-Step Loop

While no specific "kS-LLM" paper found, research shows **converging pattern** across independent works:

**The Loop**:
1. **Generate**: LLM produces tests/artifacts
2. **Execute**: Run tests, measure coverage
3. **Analyze**: Identify gaps, failures, uncovered paths
4. **Feedback**: Construct prompts with errors, coverage gaps, counter-examples
5. **Refine**: LLM generates improved tests
6. **Deduplicate/Cache**: Track history, prevent redundant work
7. **Converge**: Stop at coverage threshold, max iterations, or k consecutive no-improvement rounds

### 3.2 Key Implementations

#### Panta: Path History Tracking

**Refinement loop**:
1. **Path Selection**: Ranks uncovered paths using "coverage deficiency score"
2. **Prompt Construction**: Embeds paths with source code, existing tests, failed test feedback
3. **Test Generation**: LLM generates tests targeting specific paths
4. **Validation & Repair**: Execute; passing tests integrate, failed tests undergo repair (up to 3 iterations)
5. **Convergence Check**: Repeats until 100% coverage, max iterations, or `maxNoIncreaseLimit` consecutive no-improvement

**Key innovation**: `pathHistory` data structure tracks selection frequency
- Paths selected `maxSelectedConst` times without coverage are excluded
- Prevents wasteful redundancy

#### CoverUp: Multi-Turn Refinement

**Finding**: Nearly **50% of successful test generation** occurs through multi-turn refinement, not initial prompts

**Feedback signals**:
- Coverage gaps (lines/branches not executed)
- Test execution results (pass/fail)
- Error messages and excerpts

**Handling failures**: "If they do not increase coverage, or result in failures/errors, CoverUp continues chat, pointing out problems and requesting improvements"

#### SymPrompt: Multi-Stage Coverage-Guided

**Approach**: "Multi-step reasoning by deconstructing testsuite generation into multi-stage sequence, each driven by specific prompt aligned with execution paths"

**Results**:
- **5x improvement** in correct test generations
- **26% coverage increase** for CodeGen2
- **2x improvement** for GPT-4 vs baseline

### 3.3 Coverage Feedback (GCOV Integration)

**Runtime coverage analysis**: "GCOV source code coverage analysis tool translates execution profiles into human-readable test coverage report"

**Metrics used**:
- Line coverage
- Branch coverage
- Path coverage
- Function coverage

**Feedback construction**: Coverage gaps directly inform next-round prompts targeting uncovered regions

### 3.4 Caching and Corpus Management

#### Deduplication Strategies

**MinHash LSH** (Locality Sensitive Hashing):
- Probabilistic technique estimating Jaccard similarity
- Lossy compression preserving similarity structure
- Purpose-built for large-scale deduplication

**Impact**: "Allows training of models that emit memorized text ten times less frequently"

**Seed deduplication** (fuzzing context): "Seeds are deduplicated and sent to fuzzers to prevent redundant seeds from wasting computational resources"

#### Path Tracking (Panta)

`pathHistory` mechanism serves as corpus manager:
- Tracks which paths attempted
- Counts selection frequency
- Automatic exclusion of intractable paths

### 3.5 Failure-Driven Generation (Edge Cases)

#### FuzzGPT: Historical Bug-Driven

**Methodology**:
- Leverages "historical bug-triggering programs [that] may include rare/valuable code ingredients"
- Fine-tuning on bug-triggering examples
- In-context learning with prompts directing toward atypical inputs

**Results**:
- **76 bugs detected** (49 previously unknown)
- **11 high-priority vulnerabilities**
- Substantially outperformed prior TitanFuzz

#### TELPA: Counter-Example Refinement

"Identifies ineffective tests as counter-examples for LLMs and employs feedback-based process to iteratively refine these counter-examples"

### 3.6 Performance vs Traditional Fuzzing

**COTTONTAIL** (IEEE S&P 2026):
- **30.73% and 41.32%** improvement over baselines (line and branch coverage)
- **6 new CVEs** discovered

**TinyXML2 case study**:
- Coverage increased from **38% to 69%** with LLM-based fuzzing

**Advantages**:
- "LLM-driven test generation has demonstrated significant promise and versatility"
- "Leverage extensive training on diverse codebases to generate tests that are not only syntactically correct but also semantically meaningful"

**Challenges**:
- Hallucinations ("plausible yet incorrect outputs")
- Complex API dependencies
- Semantic consistency requirements

### 3.7 Generalization (SELF-REFINE Framework)

**General-purpose pattern**: "Alternates between FEEDBACK and REFINE generative steps working in tandem"

**Loop**:
1. Initial output generated
2. Feedback obtained from model
3. Feedback used to refine draft
4. Repeat for k iterations or convergence

**Domain-agnostic**: Applies beyond testing to writing, code generation, problem-solving, etc.

---

## Part 4: Synthesis - The Converging Architecture

### 4.1 Common Patterns Across All Three Domains

| Pattern | K-Step Reasoning | MAKER | Iterative Test Gen |
|---------|-----------------|-------|-------------------|
| **Atomic decomposition** | Task → subtasks (DecomP, ADaPT) | m=1 (one decision per agent) | Path → test per path |
| **Step-level verification** | PRMs, contract validation | First-to-ahead-by-k voting | Execute + coverage feedback |
| **Iterative refinement** | Reflexion (Actor-Eval-Reflect) | Resample on red-flag | Generate-Execute-Analyze loop |
| **Error filtering** | Red-flagging, circuit breakers | >700 tokens, format violations | Counter-examples, failed tests |
| **Caching/memory** | Task Memory Engine, DataStates | Event-sourcing, artifact store | pathHistory, MinHash LSH |
| **Convergence criteria** | Max iterations, quality threshold | k-margin achieved | Coverage target, no improvement |
| **Cost optimization** | Smaller models for subtasks | Cheap models + voting > expensive | Deduplication prevents waste |

**The Universal Loop**:
```
Plan → Decompose (atomic) → Execute (with voting) → Validate (contracts/coverage) →
Reflect (on failures) → Refine (targeted) → Cache (history) →
Converge (threshold or max iterations)
```

### 4.2 Key Theoretical Insights

#### Logarithmic Scaling Laws
- **MAKER**: k_min = Θ(ln s) where s = total steps
- **Implication**: Million-step tasks only need ~10x more error correction than thousand-step
- **Enabler**: Makes long-horizon reliable execution feasible

#### Process > Outcome Validation
- **PRMs outperform ORMs**: Step-level feedback beats end-result checking
- **Earlier error detection**: Fail fast, localize precisely
- **Better credit assignment**: Know *which* step failed, not just *that* it failed

#### Dynamic Sampling Efficiency
- **Fixed-N wasteful**: Self-consistency always samples N times
- **SPRT optimal**: First-to-ahead samples minimally for target confidence
- **Adaptive**: Easy steps converge quickly, hard steps get more attempts

#### Correlated Error Filtering
- **Red-flagging critical**: Not about improving accuracy, about preventing simultaneous failures
- **Length signals confusion**: >700 tokens = model lost in logical loops
- **Format signals semantics**: Structural errors correlate with meaning errors

### 4.3 Architectural Requirements

For a system to implement these patterns reliably, it needs:

1. **Task Decomposer**:
   - LLM-driven or rule-based decomposition
   - Outputs DAG of atomic steps with dependencies
   - Validates decomposition completeness

2. **Step Executor**:
   - Per-step validation (contracts, schemas)
   - Structured output enforcement
   - Timeout and resource limits

3. **Voting Orchestrator**:
   - First-to-ahead-by-k algorithm
   - Dynamic k tuning based on error rates
   - Red-flag pre-filtering

4. **State Manager**:
   - Event-sourced execution (checkpointing)
   - Artifact content-addressed storage
   - Resume-from-step capability

5. **Reflection Engine**:
   - Analyzes failures to generate feedback
   - Constructs targeted refinement prompts
   - Iterative improve-and-retry loop

6. **Cache/Corpus Manager**:
   - Deduplication (semantic similarity hashing)
   - Path/attempt history tracking
   - Exclusion of intractable paths after k attempts

7. **Convergence Monitor**:
   - Tracks progress metrics (coverage, quality scores)
   - Detects plateaus (k consecutive no-improvement)
   - Enforces budgets (max iterations, cost, time)

---

## Part 5: MultiHead Capability Mapping

### 5.1 What MultiHead Already Has ✅

| Component | MultiHead Implementation | Status |
|-----------|------------------------|--------|
| **DAG Executor** | `dag_executor.py` with parallel step execution | ✅ Complete |
| **Event-sourcing** | `EventStore` with kill-9 resilience | ✅ Complete |
| **Artifact storage** | Content-addressed SHA-256 sharded store | ✅ Complete |
| **Consensus voting** | 5 strategies including FIRST_TO_AHEAD | ✅ Complete |
| **Red-flag filtering** | >700 tokens, JSON parse failure, missing fields | ✅ In consensus.py |
| **Router** | Weighted head selection (40% active, 30% breaker, 15% VRAM) | ✅ Complete |
| **Circuit breaker** | Per-head health tracking in Router | ✅ Complete |
| **Recipe system** | YAML WorkOrder → StepDef pipeline definitions | ✅ Complete |
| **Retry policy** | Per-step exponential backoff, jitter | ✅ In orchestrator |
| **State checkpointing** | Event replay reconstructs run state | ✅ Complete |

**Assessment**: MultiHead already implements **~60% of MAKER/K-step architecture**

### 5.2 What's Missing or Partial ⚠️

| Capability | Current State | Gap |
|-----------|---------------|-----|
| **Step-level validators** | ⚠️ Tool definitions exist, no formal contracts | Need Hoare-style pre/post-conditions |
| **Reflection loops** | ❌ None | Need Actor-Evaluator-Reflect-Memory cycle |
| **Auto-decomposition** | ❌ Requires manual recipe design | Need LLM-driven task → atomic steps |
| **Process Reward Models** | ❌ None | Step-level learned/heuristic validators |
| **Adaptive k tuning** | ⚠️ Fixed k in FIRST_TO_AHEAD | Dynamic k based on observed error rates |
| **Coverage feedback** | ❌ None | Not applicable to general tasks (testing-specific) |
| **Corpus management** | ⚠️ Artifact store, no deduplication | Need semantic similarity hashing |
| **Cost tracking** | ⚠️ Metrics exist, no per-step budget enforcement | Budget-aware execution |
| **Resume API** | ⚠️ Event replay works, no exposed API | User-facing `resume_from_step()` |

### 5.3 Architecture Alignment Analysis

#### Strengths (Natural Fit)

**1. Event-sourced DAG execution**:
- MAKER needs checkpointing for million-step tasks (kill-9 resilient)
- MultiHead already has this via `EventStore` + `dag_executor.py`
- **No changes needed**: Core infrastructure ready

**2. First-to-Ahead consensus**:
- MultiHead's `FIRST_TO_AHEAD` strategy (lines 648-847 in `consensus.py`) implements MAKER's voting
- Already has red-flag pre-filtering (>700 tokens, JSON parse failure)
- **Minor enhancements needed**: Dynamic k tuning, per-step-type k configuration

**3. Head routing for specialized subtasks**:
- Router weighted scoring (40% active, 30% breaker) matches MAKER's principle of "use small cheap models with voting"
- `required_kind` routing enables automatic head selection for atomic steps
- **No changes needed**: Works out of the box

**4. Artifact content-addressing**:
- SHA-256 sharded artifact store provides deduplication infrastructure
- Immutable artifacts enable replay and equivalence checking
- **Enhancement needed**: Semantic similarity (not just exact matching)

#### Weaknesses (Missing Components)

**1. No automatic task decomposition**:
- Current: User manually designs recipes in YAML
- Needed: `recipe.auto_decompose_goal(goal_text) -> WorkOrder`
- Gap: Requires LLM-driven decomposition module

**2. No step-level validators**:
- Current: Tool definitions lack formal contracts
- Needed: Preconditions, postconditions, invariants per StepDef
- Gap: Contract-based verification layer

**3. No reflection loops**:
- Current: Steps execute once, failures just retry with same prompt
- Needed: Analyze failure → generate feedback → refine prompt → re-attempt
- Gap: Reflexion-style Actor-Evaluator-Reflect cycle

**4. No Process Reward Models**:
- Current: Outcome-only validation (did step succeed?)
- Needed: Step-level quality scoring for intermediate results
- Gap: Learned or heuristic validators integrated into orchestrator

### 5.4 Synergies with Existing Features

#### BotVibes/ACP Integration
- **Delegation to Claude worker**: Step validators could delegate verification to Claude via ACP tasks
- **Multi-agent consensus**: Stage 3 validation recipe already uses this pattern
- **Reflection prompts as ACP tasks**: Offload expensive reflection analysis to Claude

#### Night Shift Pipeline
- **Claim validation**: Use validators to filter low-quality claims before insertion
- **Narrative extraction refinement**: Reflexion loop for improving extraction quality
- **Long-running checkpoints**: Event-sourcing critical for overnight processing

#### Router System
- **Validator success rates**: Extend routing weights to consider validation pass rate per head
- **Circuit breaker signals**: Validators become inputs to breaker logic
- **Cost optimization**: Route to cheaper heads for atomic steps, expensive heads for validation

---

## Part 6: Gap Analysis and Recommendations

### 6.1 Priority Ranking

#### High Priority (Align with Current Architecture)
**ROI: High | Complexity: Low | Timeline: 1-2 weeks**

1. **Contract-based tool validation** ⭐⭐⭐⭐⭐
   - Natural extension of existing tool system
   - Add precondition/postcondition to `StepDef`
   - Validate at step boundaries in `orchestrator._execute_step()`
   - **Impact**: Catch 80% of invalid outputs before they corrupt downstream steps

2. **Step-level validators** ⭐⭐⭐⭐⭐
   - Critical for reliability, low implementation cost
   - Add `validator: Optional[Validator]` to `StepDef`
   - Hook into orchestrator after step execution, before artifact commit
   - **Impact**: Enable step-level retry without full pipeline restart

3. **Expose resume API** ⭐⭐⭐⭐
   - Already have infrastructure (event replay), just needs interface
   - `await orchestrator.resume_from_step(run_id, step_index, modified_step=None)`
   - **Impact**: A/B test steps, debug in isolation, faster iteration

#### Medium Priority (New Capabilities)
**ROI: Medium | Complexity: Medium | Timeline: 2-4 weeks**

4. **Reflexion loop** ⭐⭐⭐⭐
   - Adds self-correction, moderate complexity
   - Wrapper around `_execute_step()` that analyzes failures and refines prompts
   - Store reflections in artifact store for future attempts
   - **Impact**: Self-correcting pipelines reduce manual intervention

5. **Confidence-weighted consensus** ⭐⭐⭐
   - Enhancement to existing FIRST_TO_AHEAD
   - Extract confidence scores from model outputs
   - Weight votes by confidence, not just count
   - **Impact**: Better decisions when models are uncertain

6. **Dynamic k tuning** ⭐⭐⭐
   - Adjust k_margin based on observed per-step error rates
   - `k = max(k_min, int(ln(total_steps) * error_rate_multiplier))`
   - Cost/reliability trade-off optimization
   - **Impact**: Cheaper execution on easy tasks, more reliable on hard ones

#### Low Priority (Research/Future)
**ROI: Low-Medium | Complexity: High | Timeline: 1-3 months**

7. **Dynamic auto-decomposition** ⭐⭐
   - Complex, needs careful design
   - LLM-driven: `goal_text → DAG<StepDef>` via prompt
   - Validation: Ensure decomposition is complete and atomic
   - **Impact**: User-friendly (no manual recipes), but risky (decomposition errors catastrophic)

8. **Process reward models** ⭐⭐
   - Requires training data and model
   - Step-level quality scoring for intermediate results
   - Integration: PRMs as validators in orchestrator
   - **Impact**: High ceiling, but significant R&D effort

9. **Semantic equivalence clustering** ⭐
   - MAKER paper didn't implement this (exact matching sufficient)
   - Use LLM to classify semantically equivalent outputs
   - Benefit marginal for structured tasks (JSON, code)
   - **Impact**: Minor efficiency gain, not worth complexity for now

### 6.2 Recommended Implementation Phases

#### Phase 1: Foundations (Weeks 1-2)
**Goal**: Add missing safety nets to existing architecture

- [ ] Implement `Validator` class with precondition/postcondition hooks
- [ ] Add `validator: Optional[Validator]` field to `StepDef`
- [ ] Integrate validators into `orchestrator._execute_step()` (after execution, before artifact commit)
- [ ] Create validator library: JSONSchemaValidator, FormatValidator, LengthValidator, ContractValidator
- [ ] Expose `resume_from_step()` API for debugging and iteration

**Outcome**: Existing recipes become more reliable with minimal code changes

#### Phase 2: Self-Correction (Weeks 3-4)
**Goal**: Enable pipelines to learn from mistakes

- [ ] Implement `ReflectionEngine` class (Actor-Evaluator-Reflect pattern)
- [ ] Add `_execute_step_with_reflection()` wrapper in orchestrator
- [ ] Store reflections in artifact store with `reflection_id` refs
- [ ] Add `max_reflection_attempts` to `StepDef`
- [ ] Test with Stage 3 validation recipe (reflection on low-confidence consensus)

**Outcome**: Pipelines self-correct on transient failures, reducing need for manual retry

#### Phase 3: Dynamic Tuning (Weeks 5-6)
**Goal**: Optimize cost vs reliability trade-offs automatically

- [ ] Implement per-step-type error rate tracking in `MetricsCollector`
- [ ] Add `adaptive_k: bool` flag to consensus config
- [ ] Compute `k_optimal = max(k_min, int(ln(total_steps) * observed_error_rate))`
- [ ] Add confidence weighting to FIRST_TO_AHEAD (optional enhancement)
- [ ] Benchmark: Compare fixed k=3 vs adaptive k on long-running recipes

**Outcome**: Cheaper execution on easy tasks, more samples on hard tasks

#### Phase 4: Auto-Decomposition (Weeks 7-12)
**Goal**: User provides goal text, system generates atomic step DAG

- [ ] Design prompt template for decomposition: `goal → list[AtomicStep]`
- [ ] Implement `AutoDecomposer` class with validation (completeness, atomicity checks)
- [ ] Add dependency inference: analyze step inputs/outputs to build DAG
- [ ] Test with synthetic tasks (Towers of Hanoi, multi-step math)
- [ ] Gradually roll out to real workflows (start with Night Shift pipeline)

**Outcome**: Non-technical users can create pipelines via natural language goals

### 6.3 Metrics for Success

Track these metrics to validate implementation effectiveness:

| Metric | Baseline (Current) | Target (Phase 1) | Target (Phase 3) |
|--------|-------------------|------------------|------------------|
| **Step failure rate** | ~5-10% (estimate) | <3% | <1% |
| **Pipeline completion rate** (no manual intervention) | ~60% | >80% | >95% |
| **Cost per successful run** (tokens) | 100K (baseline) | 90K (-10%) | 70K (-30%) |
| **Avg retry attempts per failed step** | 3 | 2 | 1.5 |
| **Time to debug failed run** (human hours) | 2-4 hours | <1 hour | <30 min |

### 6.4 Risk Mitigation

#### Risk 1: Validators too strict → false negatives
**Mitigation**: Start with lenient thresholds, gradually tighten based on observed false positive rate

#### Risk 2: Reflection loops don't converge → infinite retries
**Mitigation**: Hard cap on `max_reflection_attempts` (default: 3), detect cycles (same reflection twice = abort)

#### Risk 3: Auto-decomposition generates invalid DAGs
**Mitigation**: Human-in-the-loop validation for auto-decomposed recipes (at least initially), extensive testing on synthetic tasks first

#### Risk 4: Dynamic k tuning over-optimizes for cost → reliability degrades
**Mitigation**: Set `k_min` floor (never go below 2), alert if overall success rate drops below threshold

---

## Part 7: Actionable Next Steps

### Immediate (This Week)

1. **Task #159**: Design atomic step contract framework
   - Define `Validator` interface (precondition, postcondition, confidence_threshold)
   - Create base validator implementations (JSONSchema, Format, Length)
   - Document contract patterns for common step types

2. **Task #160**: Design voting & consensus enhancements
   - Specify dynamic k tuning algorithm
   - Design confidence-weighted voting extension to FIRST_TO_AHEAD
   - Document red-flag criteria per head type

3. **Task #161**: Prototype Stage 3 validation decomposition
   - Use existing `architectural-decision.yaml` recipe as baseline
   - Add validators to each step (JSON schema, verdict enum, confidence range)
   - Test reflection loop on low-confidence consensus results

### Short-Term (Next 2 Weeks)

4. **Implement Phase 1** (Foundations):
   - Validator class and integration
   - Resume API exposure
   - Validator library (JSONSchema, Format, Length, Contract)

5. **Test with Real Workflow**:
   - Apply validators to Night Shift narrative extraction
   - Measure step failure rate reduction
   - Benchmark pipeline completion rate improvement

### Medium-Term (Next Month)

6. **Implement Phase 2** (Self-Correction):
   - Reflexion engine
   - Reflection-wrapped step execution
   - Artifact storage for reflections

7. **Evaluate Against MAKER Benchmarks**:
   - Replicate Towers of Hanoi experiment (smaller scale: 10 disks = 1,023 moves)
   - Measure: k_min required, cost vs baseline, error rate
   - Compare to gpt-4o-mini + voting results from paper

### Long-Term (Next Quarter)

8. **Implement Phase 3** (Dynamic Tuning):
   - Error rate tracking per step type
   - Adaptive k algorithm
   - Confidence-weighted voting

9. **Prototype Phase 4** (Auto-Decomposition):
   - Decomposition prompt engineering
   - Validation logic (completeness, atomicity)
   - Synthetic task testing (Hanoi, multi-step math)

10. **Meta-Improvement**:
    - Use the framework to improve itself
    - Auto-decompose "Add PRM validators to orchestrator" goal
    - Demonstrate self-bootstrapping capability

---

## Part 8: Key Takeaways

### Converging Insights Across All Research

1. **Atomic decomposition is non-negotiable**: m=1 (MAKER), per-path tests (kS-LLM), subtask delegation (K-step)
2. **Logarithmic scaling enables long horizons**: k ∝ ln(s) makes million-step execution feasible
3. **Process > Outcome validation**: Step-level feedback (PRMs) outperforms end-result checking (ORMs)
4. **Dynamic sampling beats fixed-N**: First-to-ahead (SPRT) optimal, self-consistency wasteful
5. **Correlated error filtering critical**: Red-flagging prevents simultaneous failures, not just individual ones
6. **Reflection enables self-correction**: 50% of test generation success comes from refinement, not initial prompts
7. **Caching prevents duplicate work**: pathHistory (Panta), MinHash LSH (corpus management), artifact store (MultiHead)
8. **Smaller models + voting > larger models alone**: gpt-4o-mini + MAKER ($3.5K) beats o1 ($71K)

### MultiHead's Unique Position

**Strengths**:
- Event-sourced DAG execution (kill-9 resilient, checkpointed)
- First-to-ahead consensus (MAKER-compatible)
- Weighted routing (cost-optimized head selection)
- Content-addressed artifacts (deduplication infrastructure)
- Already ~60% of the way to full MAKER implementation

**Opportunities**:
- **Local million-step reasoning** without expensive API calls
- **GPU-efficient**: Swap heads per step type, no 24/7 VRAM occupation
- **Cost-effective**: Use tiny local models (Qwen3-8B) with voting instead of GPT-4
- **Privacy-preserving**: Entire MDAP runs locally, no data leaves machine

**Gaps**:
- Missing step-level validators (contracts, PRMs)
- No reflection loops (self-correction)
- Manual recipe design (no auto-decomposition yet)
- No adaptive k tuning (fixed k=3 in consensus)

### The Path Forward

1. **Short-term** (Weeks 1-2): Add validators and resume API (Phase 1)
2. **Medium-term** (Weeks 3-6): Implement reflection loops and dynamic k (Phases 2-3)
3. **Long-term** (Months 2-3): Auto-decomposition and meta-improvement (Phase 4)
4. **Validation**: Benchmark against MAKER (Towers of Hanoi), measure improvement on existing workflows (Night Shift)
5. **Application**: Use framework for Stage 3 validation consensus (original motivating use case)

---

## Sources

### K-Step Reasoning
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [ReAct: Reasoning + Acting](https://arxiv.org/abs/2210.03629) - Yao et al., 2023
- [Reflexion: Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366) - Shinn et al., 2023
- [Let's Verify Step by Step (PRMs)](https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf) - OpenAI
- [ToolGate: Contract-Grounded Tool Execution](https://arxiv.org/html/2601.04688)
- [OpenAI Reasoning Models (o1/o3)](https://platform.openai.com/docs/guides/reasoning)

### MAKER/MDAP
- [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/html/2511.09030v1) - Meyerson et al., 2024
- [MAKER Framework Overview](https://sanj.dev/post/2025-11-20-maker-framework)
- [Sequential Probability Ratio Test](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test) - Wald & Wolfowitz

### Iterative Test Generation
- [Panta: Iterative Hybrid Program Analysis](https://arxiv.org/html/2503.13580)
- [CoverUp: Coverage-Guided Test Generation](https://arxiv.org/html/2403.16218v1/)
- [FuzzGPT: Edge-Case Fuzzing](https://arxiv.org/abs/2304.02014)
- [SymPrompt: Code-Aware Prompting](https://dl.acm.org/doi/10.1145/3643769)
- [COTTONTAIL: LLM-Driven Concolic Execution](https://github.com/Cottontail-Proj/cottontail)
- [SELF-REFINE: Iterative Refinement](https://openreview.net/pdf?id=S37hOerQLB)

---

**End of Research Synthesis**

**Next**: Design atomic step contract framework (Task #159) and voting enhancements (Task #160), then prototype on Stage 3 validation (Task #161).
