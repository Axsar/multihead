# Design: Intelligent Solver Selection & Multi-Head Orchestration

**Status**: Draft
**Created**: 2026-02-21
**Task**: #160
**Purpose**: Design MultiHead's core differentiator - intelligent routing to the RIGHT solver (not just "use an LLM")

---

## 1. The Vision: MultiHead as Universal Orchestrator

### 1.1 Problem Statement

**Current state**: Hardcoded heads in `config/heads.yaml`
```yaml
heads:
  - head_id: qwen-llm
    model: Qwen/Qwen3-8B-Instruct
  - head_id: qwen-vlm
    model: Qwen/Qwen3-VL-32B-Thinking
```

**Limitations**:
1. **Static registry**: New models require manual YAML updates
2. **No discovery**: Can't learn about new YOLO v12, SAM3, or specialized tools
3. **Manual routing**: User must know which model is best for task type
4. **Local-only**: Ignores vast capabilities of BotVibes marketplace
5. **No benchmarking**: Don't know if Qwen3-8B is actually best for reasoning tasks
6. **Expertise required**: User must understand model capabilities (YOLO vs SAM vs GroundingDINO)

**The Real Problem**:
> **MultiHead is meant to use the BEST tools, but it can't discover what those are or intelligently choose between them.**

### 1.2 The Solution: Self-Improving Solver Selection

**Core Principle**: Use MultiHead's own consensus mechanisms to decide which solvers to use

**Three-Layer Architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: EXTERNAL MARKETPLACE (BotVibes)                        │
│ - Provider discovery (what capabilities exist?)                 │
│ - Provider selection (which agent is best for task type?)      │
│ - Privacy-preserving delegation (safe data handling)           │
└─────────────────────────────────────────────────────────────────┘
                           ▲
                           │ delegates to
                           │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: DYNAMIC SOLVER REGISTRY                               │
│ - Model discovery (new YOLO, SAM, LLM releases)               │
│ - Continuous benchmarking (which solver performs best?)       │
│ - Solver selection via consensus (vote on best for task type) │
└─────────────────────────────────────────────────────────────────┘
                           ▲
                           │ selects from
                           │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: SOLVER TYPES (not just LLM!)                          │
│ - LLM (reasoning, NL understanding)                            │
│ - VLM (image understanding)                                     │
│ - Deterministic (math, transforms, validation)                 │
│ - Specialized Models (YOLO, SAM, UNet)                         │
│ - Tools (database, API, file ops)                              │
│ - External Services (BotVibes, APIs)                           │
└─────────────────────────────────────────────────────────────────┘
```

**Key Innovation**: Layer 2 uses Layer 1 to make decisions about itself (meta-reasoning)

---

## 2. Layer 1: Solver Type Taxonomy

### 2.1 Solver Categories

```python
class SolverType(str, enum.Enum):
    """Types of solvers MultiHead can route to."""

    # LLM-based
    LLM = "llm"              # Text reasoning (Qwen, GPT, Claude)
    VLM = "vlm"              # Vision-language (Qwen-VL, GPT-4V)

    # Specialized ML models
    OBJECT_DETECTION = "object_detection"  # YOLO, Faster-RCNN, GroundingDINO
    SEGMENTATION = "segmentation"          # SAM, UNet, Mask-RCNN
    EMBEDDING = "embedding"                # sentence-transformers, CLIP

    # Deterministic
    DETERMINISTIC = "deterministic"        # Pure Python functions (no ML)
    VALIDATOR = "validator"                # Schema checks, format validation

    # External
    TOOL = "tool"                          # Database, file ops, API calls
    EXTERNAL_SERVICE = "external_service"  # BotVibes, external APIs
    HUMAN = "human"                        # Human-in-the-loop
```

### 2.2 Capability Taxonomy

**Problem**: "LLM" is too broad. Qwen3-8B and GPT-4 are both LLMs, but capabilities differ.

**Solution**: Capability tags (multi-dimensional)

```python
class Capability(BaseModel):
    """Fine-grained capability description."""

    # Primary category
    solver_type: SolverType

    # Modality
    input_modalities: list[str]   # ["text", "image", "code", ...]
    output_modalities: list[str]  # ["text", "json", "image", ...]

    # Task types (what can it DO?)
    task_types: list[str]  # Examples:
    # - "coordinate_transform" (deterministic)
    # - "semantic_reasoning" (LLM)
    # - "object_detection" (YOLO)
    # - "instance_segmentation" (SAM)
    # - "text_classification" (LLM/specialized)
    # - "image_generation" (diffusion model)
    # - "code_generation" (code LLM)
    # - "data_query" (database tool)

    # Quality attributes
    accuracy_score: float | None = None      # Benchmark accuracy (0-1)
    latency_p50_ms: float | None = None      # Typical latency
    cost_per_call: float | None = None       # USD per invocation

    # Resource requirements
    requires_gpu: bool = False
    vram_mb: int = 0

    # Constraints
    max_input_tokens: int | None = None      # For LLMs
    max_image_size: tuple[int, int] | None = None  # For vision models
    supported_formats: list[str] = []        # e.g., ["json", "yaml", "xml"]
```

**Example: YOLO variants**
```python
yolo_v8 = Capability(
    solver_type=SolverType.OBJECT_DETECTION,
    input_modalities=["image"],
    output_modalities=["json"],  # bbox coordinates
    task_types=["object_detection", "bbox_prediction"],
    accuracy_score=0.89,  # mAP on COCO
    latency_p50_ms=45,
    requires_gpu=True,
    vram_mb=2000,
    supported_formats=["coco_json"]
)

yolo_v11 = Capability(
    solver_type=SolverType.OBJECT_DETECTION,
    input_modalities=["image"],
    output_modalities=["json"],
    task_types=["object_detection", "bbox_prediction", "instance_segmentation"],  # NEW in v11
    accuracy_score=0.92,  # Better than v8
    latency_p50_ms=38,    # Faster too!
    requires_gpu=True,
    vram_mb=2500,
    supported_formats=["coco_json", "yolo_format"]
)
```

**Router decision**: For "object_detection" task, compare candidates on accuracy, latency, cost → pick best

---

## 3. Layer 2: Dynamic Solver Registry

### 3.1 Solver Manifest (Extended)

**Current** (hardcoded):
```yaml
# config/heads.yaml
heads:
  - head_id: qwen-llm
    adapter: transformers
    model: Qwen/Qwen3-8B-Instruct
```

**Proposed** (dynamic, capability-aware):
```yaml
# config/solvers.yaml
solvers:
  # Local models
  - solver_id: qwen-llm-8b
    name: "Qwen3 8B Instruct"
    adapter: transformers
    model: Qwen/Qwen3-8B-Instruct
    capabilities:
      solver_type: llm
      task_types: [semantic_reasoning, text_generation, classification]
      input_modalities: [text]
      output_modalities: [text, json]
      max_input_tokens: 32768
      accuracy_benchmarks:
        mmlu: 0.71
        gsm8k: 0.83
      latency_p50_ms: 120
      cost_per_1k_tokens: 0.0  # Local, no cost
      requires_gpu: true
      vram_mb: 6000

  # Specialized vision model
  - solver_id: yolo-v11-local
    name: "YOLO v11"
    adapter: custom_yolo  # New adapter type
    model: ultralytics/yolov11
    capabilities:
      solver_type: object_detection
      task_types: [object_detection, bbox_prediction, instance_segmentation]
      input_modalities: [image]
      output_modalities: [json]
      accuracy_benchmarks:
        coco_map: 0.92
      latency_p50_ms: 38
      requires_gpu: true
      vram_mb: 2500

  # SAM for segmentation
  - solver_id: sam2-base
    name: "Segment Anything v2"
    adapter: custom_sam
    model: facebook/sam2-hiera-base-plus
    capabilities:
      solver_type: segmentation
      task_types: [instance_segmentation, mask_generation]
      input_modalities: [image]
      output_modalities: [mask_image, polygon]
      latency_p50_ms: 200
      requires_gpu: true
      vram_mb: 8000

  # Deterministic functions
  - solver_id: coordinate-transform-py
    name: "Coordinate Space Transformer"
    adapter: deterministic  # NEW adapter type
    implementation: "multihead.transforms.coordinate_transform"
    capabilities:
      solver_type: deterministic
      task_types: [coordinate_transform, spatial_transform]
      input_modalities: [json]
      output_modalities: [json]
      latency_p50_ms: 1  # Pure Python, microseconds
      cost_per_call: 0.0
      accuracy_score: 1.0  # Deterministic = perfect

  # External API
  - solver_id: openai-gpt4o
    name: "OpenAI GPT-4o"
    adapter: openai
    model: gpt-4o-2024-11-20
    capabilities:
      solver_type: llm
      task_types: [semantic_reasoning, text_generation, code_generation]
      input_modalities: [text, image]  # Multimodal
      output_modalities: [text, json]
      max_input_tokens: 128000
      accuracy_benchmarks:
        mmlu: 0.88
        gsm8k: 0.95
      latency_p50_ms: 800
      cost_per_1k_tokens: 0.0025  # $2.50 per 1M tokens

  # BotVibes marketplace providers
  - solver_id: botvibes-vision-expert
    name: "Vision Specialist Agent"
    adapter: botvibes  # NEW adapter type
    provider_id: "agent-vision-001"  # BotVibes agent ID
    capabilities:
      solver_type: external_service
      task_types: [image_analysis, visual_reasoning, ocr]
      input_modalities: [image, text]
      output_modalities: [text, json]
      latency_p50_ms: 5000  # Network + queue time
      cost_per_call: 0.10  # BotVibes credit system
      privacy_level: "encrypted"  # Data handling guarantee
```

### 3.2 Solver Discovery (Continuous Learning)

**Problem**: New models released constantly (YOLO v12, SAM3, Qwen4)

**Solution**: Automated discovery + benchmarking pipeline

```python
class SolverDiscovery:
    """Discovers and benchmarks new solvers."""

    async def discover_new_models(self) -> list[SolverCandidate]:
        """Poll various sources for new model releases."""
        candidates = []

        # HuggingFace Model Hub
        candidates += await self._discover_huggingface(
            tags=["text-generation", "image-segmentation", "object-detection"]
        )

        # Ollama library
        candidates += await self._discover_ollama()

        # BotVibes marketplace
        candidates += await self._discover_botvibes_providers()

        # Papers with Code (SOTA models)
        candidates += await self._discover_papers_with_code(
            tasks=["object-detection", "instance-segmentation", "llm"]
        )

        return candidates

    async def benchmark_candidate(
        self,
        candidate: SolverCandidate,
        benchmark_suite: str = "standard"
    ) -> BenchmarkResult:
        """Run benchmarks on new solver to assess capabilities."""

        if candidate.solver_type == SolverType.LLM:
            # LLM benchmarks: MMLU, GSM8K, HumanEval
            return await self._benchmark_llm(candidate)

        elif candidate.solver_type == SolverType.OBJECT_DETECTION:
            # Vision benchmarks: COCO, VOC, latency on test images
            return await self._benchmark_vision(candidate)

        elif candidate.solver_type == SolverType.DETERMINISTIC:
            # Deterministic: correctness tests, latency
            return await self._benchmark_deterministic(candidate)

        # etc.
```

**Trigger**: Run discovery weekly via Night Shift cron job

### 3.3 Solver Selection via Consensus (Meta-Reasoning!)

**The Key Insight**: Use MultiHead to decide which solver to use for a task type

**Example**: "What's the best object detection model for comic book panels?"

```yaml
# config/recipes/solver-selection.yaml
goal: "Select best object detection solver for comic panels"

steps:
  - name: gather_candidates
    head_id: deterministic-query  # Just fetch from registry
    prompt: "List all solvers with capability 'object_detection'"
    output: candidate_list

  - name: multi_model_evaluation
    consensus:
      heads:
        - head_id: qwen-llm
          prompt: |
            Evaluate these object detection models for comic book panels:
            {candidate_list}

            Consider: accuracy on small objects, speed, VRAM constraints.
            Comic panels have: speech bubbles, characters, text.

            Rank top 3 and explain reasoning. Return JSON:
            {"rankings": [{"solver_id": "...", "score": 0-1, "reasoning": "..."}]}

        - head_id: openai-gpt4o  # Get second opinion
          prompt: |
            [Same prompt as above - independent evaluation]

        - head_id: botvibes-ml-expert  # External expert via BotVibes
          prompt: |
            You are an ML engineer specializing in vision models.
            [Same evaluation task]

      strategy: weighted
      weights: [1.5, 2.0, 1.0]  # Trust GPT-4 + BotVibes expert more

  - name: benchmark_top_candidates
    # Actually RUN top 3 on sample comic images
    head_id: deterministic-benchmark
    input_refs: [multi_model_evaluation.rankings]
    prompt: "Benchmark top 3 on 100 comic panel test images"

  - name: final_selection
    head_id: qwen-llm
    prompt: |
      Based on consensus rankings and empirical benchmarks:
      {multi_model_evaluation}
      {benchmark_top_candidates}

      Make final recommendation: which solver should be default for
      'object_detection' task_type='comic_panels'?

      Return: {"selected_solver_id": "...", "confidence": 0.9, "reasoning": "..."}
```

**Output**:
```json
{
  "selected_solver_id": "yolo-v11-local",
  "confidence": 0.92,
  "reasoning": "YOLO v11 has best balance of accuracy (92% mAP) and speed (38ms). SAM2 is slower (200ms) and optimized for masks not bboxes. GroundingDINO requires text prompts which adds complexity."
}
```

**Action**: Update solver registry with task-specific preference:
```yaml
# config/solver_preferences.yaml
preferences:
  - task_type: object_detection
    context: comic_panels
    preferred_solver: yolo-v11-local
    confidence: 0.92
    selected_at: 2026-02-21T10:30:00Z
    selected_by: consensus-run-abc123
```

**Router uses this**: When step declares `task_types: ["object_detection"]` + context `comic_panels`, router automatically picks `yolo-v11-local`

---

## 4. Layer 3: BotVibes Marketplace Integration

### 4.1 Provider Discovery

**BotVibes exposes**:
- Agent capabilities (what can they do?)
- Performance metrics (success rate, latency)
- Cost (credit system)
- Reputation (ratings, reviews)
- Privacy guarantees (data handling policies)

**MultiHead queries**:
```python
async def discover_botvibes_providers(
    self,
    capability: str,
    min_reputation: float = 0.7
) -> list[BotVibesProvider]:
    """Find BotVibes agents capable of a task type."""

    response = await self.botvibes_client.search_agents(
        capabilities=[capability],
        min_reputation=min_reputation,
        privacy_levels=["encrypted", "private"],  # Data safety filter
        max_cost_per_task=0.50  # Budget constraint
    )

    return [
        BotVibesProvider(
            provider_id=agent.agent_id,
            name=agent.name,
            capabilities=agent.capabilities,
            reputation=agent.reputation,
            avg_latency_ms=agent.metrics.avg_latency,
            cost_per_task=agent.pricing.base_cost,
            privacy_level=agent.privacy_policy
        )
        for agent in response.agents
    ]
```

**Example providers**:
```json
[
  {
    "provider_id": "agent-vision-specialist-001",
    "name": "Vision Analysis Expert",
    "capabilities": ["image_analysis", "visual_reasoning", "ocr"],
    "reputation": 0.94,
    "avg_latency_ms": 4500,
    "cost_per_task": 0.08,
    "privacy_level": "encrypted",
    "success_rate": 0.91
  },
  {
    "provider_id": "agent-coordinate-fixer-007",
    "name": "Spatial Coordinate Specialist",
    "capabilities": ["coordinate_transform", "spatial_reasoning"],
    "reputation": 0.88,
    "avg_latency_ms": 2000,
    "cost_per_task": 0.05,
    "privacy_level": "private",  # Doesn't store data
    "success_rate": 0.96
  }
]
```

### 4.2 Provider Selection (With Privacy Constraints)

**Critical**: Not all data can leave local system!

```python
class DataSensitivity(str, enum.Enum):
    PUBLIC = "public"          # Can delegate to anyone
    INTERNAL = "internal"      # BotVibes encrypted only
    CONFIDENTIAL = "confidential"  # Local solvers only
    RESTRICTED = "restricted"  # Human approval required

class PrivacyConstraint(BaseModel):
    """Privacy constraints for solver selection."""

    data_sensitivity: DataSensitivity
    allowed_providers: list[str] | None = None  # Whitelist
    blocked_providers: list[str] | None = None  # Blacklist
    require_encryption: bool = True
    max_data_retention_days: int = 0  # 0 = immediate deletion
    geographic_restrictions: list[str] = []  # e.g., ["US", "EU"]
```

**Router respects privacy**:
```python
def route_with_privacy(
    self,
    step: StepDef,
    privacy: PrivacyConstraint
) -> str:
    """Route step to solver respecting privacy constraints."""

    candidates = self.get_capable_solvers(step.task_types)

    # Filter by privacy level
    if privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
        # Only local solvers allowed
        candidates = [c for c in candidates if c.is_local]

    elif privacy.data_sensitivity == DataSensitivity.INTERNAL:
        # BotVibes encrypted OK, but no public APIs
        candidates = [
            c for c in candidates
            if c.is_local or (c.is_botvibes and c.privacy_level == "encrypted")
        ]

    # Apply whitelist/blacklist
    if privacy.allowed_providers:
        candidates = [c for c in candidates if c.solver_id in privacy.allowed_providers]
    if privacy.blocked_providers:
        candidates = [c for c in candidates if c.solver_id not in privacy.blocked_providers]

    # Select best among privacy-compliant candidates
    return self.select_best(candidates, step)
```

**Example: H2V coordinate bug**
```yaml
# Stage 3 coordinate transform step
- name: transform_coordinates
  task_types: [coordinate_transform]
  privacy:
    data_sensitivity: internal  # Comic page data, not public
    require_encryption: true
    max_data_retention_days: 0

# Router decides:
# - Local deterministic function: PREFERRED (free, instant, perfect accuracy)
# - BotVibes coordinate-fixer: ALLOWED (encrypted, but costs $0.05)
# - OpenAI API: BLOCKED (data_sensitivity=internal, OpenAI stores data)
#
# Selection: coordinate-transform-py (deterministic, local, free, perfect)
```

### 4.3 BotVibes as Knowledge Source

**User's insight**: BotVibes knows how to build recipes better than we do!

**Use case**: Ask BotVibes agents to DESIGN recipes, not just execute tasks

```yaml
# Recipe: "Learn how to generate comic tails from BotVibes experts"
goal: "Design optimal recipe for comic tail generation"

steps:
  - name: query_botvibes_experts
    adapter: botvibes
    provider_search:
      capabilities: [recipe_design, computer_vision, comic_processing]
      min_reputation: 0.85
    prompt: |
      We need to generate speech tail paths for comic books.
      Input: balloon mask (PNG), character position, page resolution
      Output: SVG path for tail

      Current approach: UNet fixed 3975x6150, coordinate mismatches.

      Design optimal recipe:
      1. What task decomposition?
      2. Which models/tools for each step?
      3. What validation contracts?
      4. Expected accuracy/latency?

      Return detailed recipe YAML.

  - name: consensus_on_recipes
    consensus:
      heads:
        - head_id: qwen-llm  # Our local opinion
        - head_id: openai-gpt4o  # External validation
        - provider_search:  # BotVibes expert (from step 1)
            capabilities: [recipe_design]

      strategy: weighted
      prompt: |
        Evaluate this proposed tail generation recipe:
        {query_botvibes_experts.recipe}

        Does it solve coordinate mismatch issues?
        Is task decomposition atomic enough?
        Are validation contracts sufficient?

        Vote: approve | approve_with_changes | reject
        Suggest improvements.

  - name: benchmark_recipe
    head_id: deterministic-executor
    prompt: "Run proposed recipe on 100 test comic pages, measure accuracy + latency"

  - name: adopt_or_iterate
    head_id: qwen-llm
    prompt: |
      Based on benchmark results:
      {benchmark_recipe}

      Should we:
      (a) Adopt BotVibes recipe as-is
      (b) Adopt with modifications: [list changes]
      (c) Reject and iterate

      If (b), what specific changes?
```

**Outcome**: BotVibes experts contribute BETTER recipes than we could design manually!

**Knowledge loop**:
1. BotVibes experts share recipes → MultiHead evaluates → Adopts best
2. MultiHead's successful runs → Deposited to knowledge base → Shared with BotVibes
3. Network effect: Everyone's recipes improve everyone else's

---

## 5. Enhanced Router Architecture

### 5.1 Router Scoring (Expanded)

**Current** (from router.py):
```python
# Weighted scoring (40% active, 30% breaker, 15% VRAM, 10% error, 5% latency)
```

**Proposed** (capability + privacy + cost aware):
```python
def score_solver(
    self,
    solver: Solver,
    step: StepDef,
    privacy: PrivacyConstraint,
    budget: Budget
) -> float:
    """Score solver suitability for step (0.0-1.0)."""

    score = 0.0

    # 1. Capability match (40%) - Does it CAN do this?
    capability_match = self._score_capability_match(solver, step.task_types)
    score += 0.40 * capability_match

    # 2. Quality (20%) - How WELL can it do this?
    if solver.accuracy_score:
        score += 0.20 * solver.accuracy_score

    # 3. Active state (15%) - Avoid swap cost
    if solver.is_active:
        score += 0.15

    # 4. Circuit breaker (10%) - Health check
    breaker_state = self.get_breaker(solver.solver_id)
    if breaker_state == "closed":
        score += 0.10
    elif breaker_state == "half_open":
        score += 0.05
    # open = 0 (filtered out earlier)

    # 5. Cost efficiency (10%) - Within budget?
    cost_score = self._score_cost(solver, budget)
    score += 0.10 * cost_score

    # 6. Latency (5%) - Fast enough?
    latency_score = self._score_latency(solver, step.max_latency_ms)
    score += 0.05 * latency_score

    # FILTERS (hard constraints)
    if not self._satisfies_privacy(solver, privacy):
        return 0.0  # Privacy violation = immediate disqualification

    if not self._fits_in_budget(solver, budget):
        return 0.0  # Over budget = disqualified

    return score

def _score_capability_match(
    self,
    solver: Solver,
    required_tasks: list[str]
) -> float:
    """How well do solver capabilities match required tasks?"""

    # Exact match
    if all(task in solver.capabilities.task_types for task in required_tasks):
        return 1.0

    # Partial match
    overlap = set(required_tasks) & set(solver.capabilities.task_types)
    if overlap:
        return len(overlap) / len(required_tasks)

    # Related capability (use embedding similarity for fuzzy match)
    # e.g., "bbox_prediction" similar to "object_detection"
    max_similarity = 0.0
    for task in required_tasks:
        for solver_task in solver.capabilities.task_types:
            similarity = self._semantic_similarity(task, solver_task)
            max_similarity = max(max_similarity, similarity)

    return max_similarity * 0.8  # Penalty for not being exact match

def _satisfies_privacy(
    self,
    solver: Solver,
    privacy: PrivacyConstraint
) -> bool:
    """Check if solver meets privacy requirements."""

    # Confidential data: local only
    if privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
        return solver.is_local

    # Internal data: local or encrypted BotVibes
    if privacy.data_sensitivity == DataSensitivity.INTERNAL:
        if solver.is_local:
            return True
        if solver.adapter == "botvibes":
            return solver.privacy_level in ["encrypted", "private"]
        return False

    # Public data: anything goes
    return True
```

### 5.2 Routing Decision Flow

```
Step arrives
    ↓
Extract requirements:
  - task_types: [coordinate_transform, spatial_reasoning]
  - privacy: DataSensitivity.INTERNAL
  - budget: max $0.10 per step
  - latency: max 500ms
    ↓
Query Solver Registry:
  "Give me all solvers with capabilities matching task_types"
    ↓
Candidates:
  - coordinate-transform-py (deterministic, local, free, 1ms)
  - botvibes-coordinate-expert (external, $0.05, 2000ms)
  - qwen-llm (LLM fallback, local, free, 120ms)
    ↓
Apply filters:
  - Privacy: INTERNAL → allow local + encrypted BotVibes ✓ all pass
  - Budget: max $0.10 → all under budget ✓
  - Latency: max 500ms → botvibes too slow (2000ms) ✗ FILTERED OUT
    ↓
Remaining candidates:
  - coordinate-transform-py
  - qwen-llm
    ↓
Score each:
  coordinate-transform-py:
    - capability: 1.0 (exact match)
    - accuracy: 1.0 (deterministic)
    - active: 1.0 (always active)
    - cost: 1.0 (free)
    - latency: 1.0 (1ms is instant)
    → TOTAL: 0.95

  qwen-llm:
    - capability: 0.7 (can do it, but not specialized)
    - accuracy: 0.85 (might make mistakes)
    - active: 1.0 (currently loaded)
    - cost: 1.0 (free, local)
    - latency: 0.8 (120ms acceptable)
    → TOTAL: 0.81
    ↓
Select: coordinate-transform-py (highest score)
```

**Key**: Deterministic solver wins because it's perfect for this task type!

---

## 6. Implementation Phases

### 6.1 Phase 1: Capability-Aware Routing (Week 1)

**Goal**: Extend router to understand solver capabilities, not just head_id

- [ ] Add `Capability` model to `models.py`
- [ ] Extend `HeadManifest` with `capabilities` field
- [ ] Update `config/heads.yaml` → `config/solvers.yaml` with capabilities
- [ ] Modify `router.py`:
  - [ ] Add `_score_capability_match()` method
  - [ ] Replace head_id routing with capability-based scoring
  - [ ] Support `task_types` in `StepDef`
- [ ] Add deterministic solver support:
  - [ ] New adapter: `DeterministicAdapter` (runs Python functions)
  - [ ] Example: `coordinate_transform()` function
  - [ ] Register in solvers.yaml

**Test**: Route coordinate transform step → picks deterministic solver (not LLM)

### 6.2 Phase 2: Privacy-Aware Delegation (Week 2)

- [ ] Add `PrivacyConstraint` model
- [ ] Add `privacy` field to `StepDef`
- [ ] Modify router scoring:
  - [ ] Implement `_satisfies_privacy()` filter
  - [ ] Hard constraint: privacy violations = score 0.0
- [ ] Add privacy levels to solver manifests
- [ ] Test: Confidential data step → only local solvers allowed

### 6.3 Phase 3: BotVibes Provider Integration (Week 3)

- [ ] New adapter: `BotVibesAdapter`
- [ ] Implement `discover_botvibes_providers()` in router
- [ ] Add BotVibes provider search to routing:
  - [ ] Query marketplace for capable agents
  - [ ] Score based on reputation, cost, latency
  - [ ] Respect privacy constraints
- [ ] Test: Delegate vision task to BotVibes agent (encrypted data)

### 6.4 Phase 4: Solver Discovery & Benchmarking (Weeks 4-5)

- [ ] Implement `SolverDiscovery` class:
  - [ ] HuggingFace API integration
  - [ ] Ollama library polling
  - [ ] BotVibes marketplace polling
  - [ ] Papers with Code SOTA tracking
- [ ] Benchmarking pipeline:
  - [ ] LLM benchmarks (MMLU, GSM8K, HumanEval)
  - [ ] Vision benchmarks (COCO, latency tests)
  - [ ] Deterministic correctness tests
- [ ] Night Shift integration: Weekly discovery + benchmark job

### 6.5 Phase 5: Meta-Reasoning Solver Selection (Week 6)

- [ ] Create `solver-selection.yaml` recipe template
- [ ] Implement consensus-based solver evaluation:
  - [ ] Multi-model candidate ranking
  - [ ] Empirical benchmarking on sample data
  - [ ] Final selection with confidence score
- [ ] Update solver preferences automatically
- [ ] Router uses preferences for tie-breaking

**Test**: Run solver selection for "object_detection" on comic panels → picks YOLO v11

### 6.6 Phase 6: BotVibes Recipe Learning (Week 7)

- [ ] Implement recipe query from BotVibes experts
- [ ] Consensus evaluation of external recipes
- [ ] Benchmark proposed recipes on test data
- [ ] Adoption pipeline (approve → test → deploy)
- [ ] Feedback loop (successful recipes shared back to BotVibes)

---

## 7. Example: Routing Decisions for H2V Pipeline

### 7.1 Stage 3 Coordinate Transform

**Step definition**:
```yaml
- name: transform_tail_coordinates
  task_types: [coordinate_transform]
  input: tail_mask_unet  # 3975x6150
  output: tail_mask_page  # 4042x5929
  privacy:
    data_sensitivity: internal
  budget:
    max_cost: 0.01
    max_latency_ms: 100
```

**Router decision**:
```
Candidates:
  ✓ coordinate-transform-py (deterministic, local, free, 1ms, 1.0 accuracy)
  ✓ qwen-llm (LLM, local, free, 120ms, 0.85 accuracy)
  ✗ openai-gpt4o (blocked: data_sensitivity=internal, OpenAI external)
  ✗ botvibes-coord-expert (too slow: 2000ms > 100ms limit)

SELECTED: coordinate-transform-py
REASON: Deterministic, perfect accuracy, instant, free
```

### 7.2 Semantic Reasoning ("Is this a tail?")

**Step definition**:
```yaml
- name: classify_tail_candidate
  task_types: [semantic_reasoning, image_classification]
  input: cropped_region_image
  output: {is_tail: bool, confidence: float}
  privacy:
    data_sensitivity: internal
  budget:
    max_cost: 0.05
```

**Router decision**:
```
Candidates:
  ✓ qwen-vlm-7b (VLM, local, free, 150ms, 0.88 accuracy)
  ✓ qwen-vlm-32b (VLM, local, free, 400ms, 0.94 accuracy)
  ✓ openai-gpt4o (blocked by privacy: external)
  ✓ botvibes-vision-expert (external, $0.08, 4500ms, 0.91 accuracy)

Scoring:
  qwen-vlm-7b: capability=1.0, accuracy=0.88, active=1.0, cost=1.0, latency=0.9
    → SCORE: 0.91

  qwen-vlm-32b: capability=1.0, accuracy=0.94, active=0.0 (need swap), cost=1.0, latency=0.7
    → SCORE: 0.81 (swap penalty hurts)

SELECTED: qwen-vlm-7b
REASON: Best balance (good accuracy, already loaded, fast enough)
```

### 7.3 Instance Segmentation (Balloons)

**Step definition**:
```yaml
- name: segment_balloons
  task_types: [instance_segmentation]
  input: page_image
  output: balloon_masks
  privacy:
    data_sensitivity: internal
  budget:
    max_cost: 0.10
```

**Router decision**:
```
Candidates:
  ✓ sam2-base (segmentation, local, free, 200ms, 0.91 accuracy)
  ✓ yolo-v11 (detection+segmentation, local, free, 38ms, 0.89 accuracy)
  ✗ unet-custom (segmentation, local, but task_types=[tail_segmentation] not balloons)
  ✗ botvibes-segmentation-pro ($0.15 > $0.10 budget limit)

Scoring:
  sam2-base: capability=1.0, accuracy=0.91, active=0.0, cost=1.0, latency=0.8
    → SCORE: 0.86

  yolo-v11: capability=0.9 (can do it, but optimized for detection), accuracy=0.89, active=0.0, cost=1.0, latency=1.0
    → SCORE: 0.84

SELECTED: sam2-base
REASON: Specialized for segmentation, higher accuracy for masks
```

---

## 8. Privacy & Security Considerations

### 8.1 Data Sensitivity Classification

**Automatic classification** (heuristic + LLM):
```python
def classify_data_sensitivity(data: Any) -> DataSensitivity:
    """Classify data sensitivity level."""

    # Heuristic checks
    if contains_pii(data):  # Name, email, SSN, etc.
        return DataSensitivity.CONFIDENTIAL

    if contains_financial_data(data):
        return DataSensitivity.RESTRICTED

    if is_business_proprietary(data):
        return DataSensitivity.INTERNAL

    # LLM-based classification for ambiguous cases
    prompt = f"Classify data sensitivity: {data[:500]}..."
    result = llm.generate(prompt)

    return DataSensitivity[result.classification]
```

**User override**: User can manually tag steps with privacy level

### 8.2 BotVibes Trust Model

**Provider verification**:
- Reputation score (based on past task success rate)
- Privacy audit (data handling policies verified)
- Geographic restrictions (GDPR, data sovereignty)
- Encryption guarantees (data encrypted in transit + at rest)
- Retention policies (immediate deletion vs 30-day retention)

**Gradual trust building**:
```yaml
# Start with low-trust delegation
- phase: 1
  allowed_sensitivity: public
  max_cost_per_task: 0.10

# After 100 successful tasks, increase trust
- phase: 2
  allowed_sensitivity: internal  # Can now handle internal data
  max_cost_per_task: 0.50

# After 1000 successful tasks, full trust
- phase: 3
  allowed_sensitivity: confidential
  max_cost_per_task: 5.00
```

### 8.3 Encrypted Delegation Protocol

**For INTERNAL data sent to BotVibes**:
```python
async def delegate_to_botvibes(
    self,
    provider_id: str,
    task: TaskPayload,
    privacy: PrivacyConstraint
) -> TaskResult:
    """Delegate task to BotVibes with encryption."""

    # 1. Generate ephemeral key pair
    public_key, private_key = generate_keypair()

    # 2. Encrypt task payload with provider's public key
    encrypted_payload = encrypt(
        data=task.to_json(),
        recipient_public_key=provider.public_key
    )

    # 3. Send encrypted task via ACP
    acp_task = await self.acp_bridge.create_task(
        capability=task.task_type,
        payload_ref=encrypted_payload,
        target_agent_id=provider_id,
        metadata={
            "encryption": "rsa_2048",
            "return_encryption_key": public_key
        }
    )

    # 4. Provider processes encrypted data (can't see plaintext)
    # 5. Provider returns encrypted result

    # 6. Decrypt result with our private key
    encrypted_result = await self.acp_bridge.wait_for_result(acp_task.task_id)
    result = decrypt(encrypted_result, private_key)

    # 7. Provider deletes all data (verified via audit log)
    await self.acp_bridge.verify_deletion(acp_task.task_id)

    return TaskResult.from_json(result)
```

**Zero-knowledge protocol**: Provider processes data without seeing plaintext (for highly sensitive use cases, future)

---

## 9. Success Metrics

### 9.1 Routing Quality

| Metric | Baseline | Target |
|--------|----------|--------|
| **Correct solver selection** | 60% (manual) | >95% (automatic) |
| **Cost per 1M steps** | $3,500 (all LLM) | <$500 (mixed solvers) |
| **Avg step latency** | 150ms | <50ms (deterministic where possible) |
| **Privacy violations** | N/A | 0 (hard constraint) |

### 9.2 BotVibes Integration

| Metric | Target |
|--------|--------|
| **Provider discovery** | >50 capable agents |
| **Successful delegations** | >80% success rate |
| **Cost savings** | 30% cheaper than local-only for expensive tasks |
| **Recipe improvements** | 3+ external recipes adopted |

### 9.3 Solver Discovery

| Metric | Target |
|--------|--------|
| **New models discovered** | 10+ per month (automated) |
| **Benchmark coverage** | 100% of local models |
| **Selection accuracy** | >90% pick best for task type |

---

## 10. Open Questions & Next Steps

### 10.1 Open Questions

1. **Model versioning**: How to handle YOLO v8 → v11 upgrades?
   - **Proposal**: Shadow deployment (test v11 on 10% traffic, compare to v8)

2. **Solver conflicts**: What if two solvers both claim best for task type?
   - **Proposal**: Run mini-benchmark on representative data, empirical winner wins

3. **BotVibes pricing**: How to handle variable costs (credits, subscriptions)?
   - **Proposal**: Budget-aware routing, fallback to local if over budget

4. **Deterministic guarantees**: How to verify deterministic solvers are truly deterministic?
   - **Proposal**: Run same input 10x, verify output identical

### 10.2 Next Steps

**Immediate (This Week)**:
1. Implement `Capability` model and update `solvers.yaml`
2. Add `DeterministicAdapter` for Python functions
3. Test routing coordinate transform → deterministic solver

**Short-term (Weeks 2-3)**:
4. Add privacy constraints to router
5. Implement BotVibes adapter
6. Test encrypted delegation

**Medium-term (Month 2)**:
7. Solver discovery pipeline (Night Shift integration)
8. Benchmarking automation
9. Meta-reasoning solver selection recipe

**Long-term (Quarter 1)**:
10. BotVibes recipe learning
11. Zero-knowledge delegation protocol
12. Full MAKER-style benchmarks (Towers of Hanoi with mixed solvers)

---

## 11. The Full Vision

**MultiHead becomes**:
- **Universal orchestrator**: Routes to ANY solver (local, cloud, marketplace)
- **Self-improving**: Discovers new models, benchmarks, selects best automatically
- **Privacy-first**: Respects data sensitivity, never leaks confidential data
- **Cost-optimized**: Uses expensive solvers only when justified
- **Knowledge network**: Learns from BotVibes experts, shares back improvements

**User experience**:
```yaml
# User writes high-level goal
goal: "Process 10,000 comic pages for H2V pipeline"

# MultiHead automatically:
# 1. Discovers best solvers for each task type (YOLO v11, SAM2, coordinate-py, Qwen-VL)
# 2. Respects privacy (confidential data stays local)
# 3. Optimizes cost (uses deterministic where possible, LLM only for reasoning)
# 4. Delegates to BotVibes when beneficial (vision expert for hard cases)
# 5. Learns from failures (updates preferences, tries new solvers)
# 6. Improves over time (benchmarks new models, adopts better ones)
```

**The breakthrough**: MultiHead doesn't just execute tasks—it LEARNS which tools are best and IMPROVES itself continuously.

---

**Status**: Ready for Phase 1 implementation
**Next**: Implement capability-aware routing, test with deterministic solvers
