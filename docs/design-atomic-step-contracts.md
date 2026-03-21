# Design: Atomic Step Contract Framework

**Status**: Draft
**Created**: 2026-02-21
**Task**: #159
**Purpose**: Add Hoare-style contracts to MultiHead's step execution for reliable multi-step reasoning

---

## 1. Overview

### 1.1 Problem Statement

Current MultiHead architecture lacks **step-level verification**:
- Steps execute without precondition checks (invalid inputs can start execution)
- No postcondition validation (malformed outputs commit to artifact store)
- Failures only detected downstream (error propagation, hard to debug)
- No quality scoring for intermediate results (outcome-only validation)

**Impact**: Invalid outputs corrupt downstream steps, causing cascading failures that are expensive to debug and fix.

### 1.2 Solution: Contract-Based Verification

Introduce **Hoare-style contracts** at step boundaries:

```
{Precondition} → Execute Step → {Postcondition}
```

**Precondition**: State requirements that must be true before execution
- Input validation (schema, types, ranges)
- Dependency checks (required artifacts exist)
- Resource availability (VRAM, API quotas)

**Postcondition**: Output guarantees that must be true after execution
- Output schema validation (JSON, format)
- Quality thresholds (confidence scores, length bounds)
- Invariants (relationships between inputs/outputs)

**Benefits**:
- **Fail fast**: Catch invalid inputs before expensive execution
- **Error localization**: Know *which* step violated *which* contract
- **Self-documenting**: Contracts serve as executable specifications
- **Composability**: Downstream steps can trust upstream postconditions

### 1.3 Design Principles

1. **Non-invasive**: Works with existing StepDef/orchestrator, no breaking changes
2. **Optional**: Steps without validators execute as before (backward compatible)
3. **Composable**: Validators can be chained (e.g., schema + length + confidence)
4. **Extensible**: Easy to add custom validators for domain-specific contracts
5. **Observable**: Validation failures logged as events, queryable for debugging

---

## 2. Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      StepDef (models.py)                     │
│  + validator: Optional[Validator]  ← NEW FIELD              │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ used by
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Orchestrator._execute_step()                   │
│                                                              │
│  1. Check preconditions (validator.validate_precondition()) │
│  2. Execute step (generate output)                          │
│  3. Check postconditions (validator.validate_postcondition())│
│  4. Commit artifact (only if postcondition passes)          │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ uses
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Validator (validators.py)                   │
│                                                              │
│  Interface:                                                  │
│    - validate_precondition(state, inputs) → ValidationResult│
│    - validate_postcondition(state, output) → ValidationResult│
│                                                              │
│  Implementations:                                            │
│    - JSONSchemaValidator                                     │
│    - FormatValidator (length, regex, enum)                  │
│    - ConfidenceValidator (quality thresholds)               │
│    - ContractValidator (arbitrary Python predicates)        │
│    - CompositeValidator (chain multiple validators)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Without Validator** (current):
```
Input → Execute → Output → Commit Artifact → Next Step
         (may fail mid-execution, hard to debug)
```

**With Validator** (proposed):
```
Input → Precondition Check ──✓──→ Execute → Output → Postcondition Check ──✓──→ Commit → Next Step
           │                                             │
           └── ✗ FAIL FAST (don't execute)              └── ✗ DISCARD OUTPUT (don't commit)
```

**Validation Failure Outcomes**:
- **Precondition failure**: Skip execution, emit STEP_FAILED event with precondition violation details
- **Postcondition failure**: Discard output, retry if retry_policy allows, emit validation error event

---

## 3. Validator Interface

### 3.1 Base Classes

```python
# src/multihead/validators.py

from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of a validation check."""
    passed: bool
    confidence: float = 1.0  # 0.0-1.0, how confident are we in this validation?
    violations: list[str] = []  # List of constraint violations
    metadata: dict[str, Any] = {}  # Additional context (e.g., which field failed)


class Validator(ABC):
    """Base class for step contract validators."""

    name: str = "base_validator"

    @abstractmethod
    def validate_precondition(
        self,
        state: dict[str, Any],  # Current run state (step_results, artifacts)
        inputs: dict[str, Any]   # Step inputs being passed
    ) -> ValidationResult:
        """Check if preconditions are met before execution."""
        pass

    @abstractmethod
    def validate_postcondition(
        self,
        state: dict[str, Any],   # Updated run state
        output: Any              # Step output (str, dict, etc.)
    ) -> ValidationResult:
        """Check if postconditions are met after execution."""
        pass
```

### 3.2 ValidationResult Semantics

**`passed: bool`**:
- `True`: Validation passed, proceed
- `False`: Validation failed, abort/retry

**`confidence: float`** (0.0-1.0):
- 1.0: Certain (e.g., JSON schema validation, exact string match)
- 0.5-0.99: Uncertain (e.g., LLM-based semantic validation)
- <0.5: Low confidence (flag for human review)

**Use case**: Adaptive retry strategies based on confidence
- High confidence failure (0.9+) → Don't retry, likely fundamental issue
- Low confidence failure (<0.6) → Retry with different head/prompt

**`violations: list[str]`**:
- Human-readable descriptions of what failed
- Example: `["Missing required field: 'verdict'", "Confidence 0.3 below threshold 0.7"]`

**`metadata: dict`**:
- Machine-readable context for debugging
- Example: `{"field": "verdict", "expected": "approve|reject", "actual": "maybe"}`

---

## 4. Built-in Validator Implementations

### 4.1 JSONSchemaValidator

**Purpose**: Validate output against JSON schema

**Use case**: Structured outputs (consensus results, API responses)

```python
class JSONSchemaValidator(Validator):
    """Validates JSON output against a schema."""

    name = "json_schema"

    def __init__(self, schema: dict, require_precondition_schema: bool = False):
        self.schema = schema
        self.require_precondition_schema = require_precondition_schema

    def validate_precondition(self, state, inputs) -> ValidationResult:
        if not self.require_precondition_schema:
            return ValidationResult(passed=True)

        # Optionally validate inputs match expected schema
        # (useful for steps that transform JSON → JSON)
        return self._validate_json(inputs.get("input_data", {}))

    def validate_postcondition(self, state, output) -> ValidationResult:
        return self._validate_json(output)

    def _validate_json(self, data: Any) -> ValidationResult:
        from jsonschema import validate, ValidationError

        try:
            validate(instance=data, schema=self.schema)
            return ValidationResult(passed=True, confidence=1.0)
        except ValidationError as e:
            return ValidationResult(
                passed=False,
                confidence=1.0,  # We're certain it failed
                violations=[f"Schema violation: {e.message}"],
                metadata={"path": list(e.path), "validator": e.validator}
            )
```

**Example usage**:
```yaml
# Recipe YAML
steps:
  - name: multi_model_critique
    consensus:
      heads: [qwen-llm, openai-gpt4o]
      strategy: weighted
      output_schema:  # ← Becomes JSONSchemaValidator automatically
        type: object
        required: [verdict, concerns, suggestions, confidence]
        properties:
          verdict:
            type: string
            enum: [approve, approve_with_changes, reject]
          concerns:
            type: array
            items: {type: string}
          suggestions:
            type: array
            items: {type: string}
          confidence:
            type: number
            minimum: 0.0
            maximum: 1.0
```

### 4.2 FormatValidator

**Purpose**: Validate length, format, regex, enum membership

**Use case**: Simple output constraints (length bounds, allowed values)

```python
class FormatValidator(Validator):
    """Validates output format constraints."""

    name = "format"

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        regex: str | None = None,
        allowed_values: list[str] | None = None,
        check_precondition: bool = False
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.regex = regex
        self.allowed_values = allowed_values
        self.check_precondition = check_precondition

    def validate_precondition(self, state, inputs) -> ValidationResult:
        if not self.check_precondition:
            return ValidationResult(passed=True)
        return self._validate_format(str(inputs.get("prompt", "")))

    def validate_postcondition(self, state, output) -> ValidationResult:
        return self._validate_format(str(output))

    def _validate_format(self, text: str) -> ValidationResult:
        violations = []

        if self.min_length and len(text) < self.min_length:
            violations.append(f"Length {len(text)} below minimum {self.min_length}")

        if self.max_length and len(text) > self.max_length:
            violations.append(f"Length {len(text)} exceeds maximum {self.max_length}")

        if self.regex:
            import re
            if not re.match(self.regex, text):
                violations.append(f"Does not match regex: {self.regex}")

        if self.allowed_values and text not in self.allowed_values:
            violations.append(f"Value '{text}' not in allowed: {self.allowed_values}")

        return ValidationResult(
            passed=len(violations) == 0,
            confidence=1.0,
            violations=violations,
            metadata={"actual_length": len(text)}
        )
```

**Example usage**:
```python
# In recipe or code
validator = FormatValidator(
    max_length=750,  # Red-flag MAKER threshold
    regex=r"^\{.*\}$"  # Must be JSON object
)
```

### 4.3 ConfidenceValidator

**Purpose**: Validate quality thresholds (confidence scores, model certainty)

**Use case**: Filter low-confidence outputs, trigger refinement loops

```python
class ConfidenceValidator(Validator):
    """Validates confidence/quality thresholds."""

    name = "confidence"

    def __init__(
        self,
        min_confidence: float = 0.7,
        confidence_field: str = "confidence",  # Where to find score in output
        extract_from_json: bool = True
    ):
        self.min_confidence = min_confidence
        self.confidence_field = confidence_field
        self.extract_from_json = extract_from_json

    def validate_precondition(self, state, inputs) -> ValidationResult:
        # Confidence validation only makes sense for postconditions
        return ValidationResult(passed=True)

    def validate_postcondition(self, state, output) -> ValidationResult:
        confidence = self._extract_confidence(output)

        if confidence is None:
            return ValidationResult(
                passed=False,
                confidence=0.5,  # Uncertain about failure
                violations=[f"Could not extract confidence from field '{self.confidence_field}'"],
                metadata={"output_type": type(output).__name__}
            )

        passed = confidence >= self.min_confidence

        return ValidationResult(
            passed=passed,
            confidence=confidence,  # Use the output's own confidence as our confidence
            violations=[] if passed else [
                f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}"
            ],
            metadata={"actual_confidence": confidence, "threshold": self.min_confidence}
        )

    def _extract_confidence(self, output: Any) -> float | None:
        if isinstance(output, dict):
            return output.get(self.confidence_field)

        if self.extract_from_json and isinstance(output, str):
            import json
            try:
                data = json.loads(output)
                return data.get(self.confidence_field)
            except json.JSONDecodeError:
                return None

        return None
```

**Example usage**:
```yaml
# In architectural-decision.yaml
steps:
  - name: multi_model_critique
    consensus:
      heads: [qwen-llm, openai-gpt4o]
      # Automatically adds ConfidenceValidator(min_confidence=0.7)
      # if output_schema has "confidence" field
```

### 4.4 ContractValidator

**Purpose**: Arbitrary Python predicates for custom contracts

**Use case**: Domain-specific invariants, complex relationships

```python
class ContractValidator(Validator):
    """Validates arbitrary Python predicates."""

    name = "contract"

    def __init__(
        self,
        precondition: Callable[[dict, dict], tuple[bool, str]] | None = None,
        postcondition: Callable[[dict, Any], tuple[bool, str]] | None = None,
        name: str = "custom_contract"
    ):
        """
        Args:
            precondition: (state, inputs) -> (passed, reason)
            postcondition: (state, output) -> (passed, reason)
        """
        self.precondition_fn = precondition
        self.postcondition_fn = postcondition
        self.contract_name = name

    def validate_precondition(self, state, inputs) -> ValidationResult:
        if not self.precondition_fn:
            return ValidationResult(passed=True)

        try:
            passed, reason = self.precondition_fn(state, inputs)
            return ValidationResult(
                passed=passed,
                confidence=1.0,
                violations=[] if passed else [f"{self.contract_name}: {reason}"]
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                confidence=0.5,
                violations=[f"Precondition check raised exception: {e}"],
                metadata={"exception": str(e)}
            )

    def validate_postcondition(self, state, output) -> ValidationResult:
        if not self.postcondition_fn:
            return ValidationResult(passed=True)

        try:
            passed, reason = self.postcondition_fn(state, output)
            return ValidationResult(
                passed=passed,
                confidence=1.0,
                violations=[] if passed else [f"{self.contract_name}: {reason}"]
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                confidence=0.5,
                violations=[f"Postcondition check raised exception: {e}"],
                metadata={"exception": str(e)}
            )
```

**Example usage**:
```python
# Custom validator for Stage 3 coordinate space
def check_coordinate_space(state, output):
    """Ensure output includes coordinate_space metadata."""
    if isinstance(output, dict):
        coord_space = output.get("coordinate_space")
        if coord_space in ["page_resolution", "unet_fixed", "panel_relative"]:
            return True, "Valid coordinate space"
        return False, f"Invalid coordinate_space: {coord_space}"
    return False, "Output not a dict"

validator = ContractValidator(
    postcondition=check_coordinate_space,
    name="coordinate_space_contract"
)
```

### 4.5 CompositeValidator

**Purpose**: Chain multiple validators (AND/OR logic)

**Use case**: Steps requiring multiple constraints (schema + length + confidence)

```python
class CompositeValidator(Validator):
    """Combines multiple validators with AND/OR logic."""

    name = "composite"

    def __init__(
        self,
        validators: list[Validator],
        mode: Literal["all", "any"] = "all"
    ):
        """
        Args:
            validators: List of validators to combine
            mode: "all" (AND) - all must pass, "any" (OR) - at least one must pass
        """
        self.validators = validators
        self.mode = mode

    def validate_precondition(self, state, inputs) -> ValidationResult:
        return self._combine_results([
            v.validate_precondition(state, inputs) for v in self.validators
        ])

    def validate_postcondition(self, state, output) -> ValidationResult:
        return self._combine_results([
            v.validate_postcondition(state, output) for v in self.validators
        ])

    def _combine_results(self, results: list[ValidationResult]) -> ValidationResult:
        if self.mode == "all":
            # All must pass
            passed = all(r.passed for r in results)
            confidence = min(r.confidence for r in results)  # Weakest link
            violations = [v for r in results for v in r.violations]
        else:
            # Any must pass
            passed = any(r.passed for r in results)
            confidence = max(r.confidence for r in results)  # Strongest validator
            violations = [v for r in results for v in r.violations if not r.passed]

        return ValidationResult(
            passed=passed,
            confidence=confidence,
            violations=violations,
            metadata={"mode": self.mode, "validator_count": len(results)}
        )
```

**Example usage**:
```python
validator = CompositeValidator(
    validators=[
        JSONSchemaValidator(schema={"type": "object", "required": ["verdict"]}),
        FormatValidator(max_length=750),  # MAKER red-flag
        ConfidenceValidator(min_confidence=0.7)
    ],
    mode="all"  # Must satisfy all three constraints
)
```

---

## 5. Integration with MultiHead

### 5.1 StepDef Extension

**Add validator field** to `StepDef` in `models.py`:

```python
class StepDef(BaseModel):
    step_id: str = ""
    name: str
    head_id: str = ""
    prompt_template: str = ""
    input_refs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    required_kind: str | None = None
    output_schema: dict[str, Any] = Field(default_factory=dict)
    retry_policy: dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 1, "backoff_ms": 1000})
    checkpoint_mode: CheckpointMode = CheckpointMode.SYNC
    extra: dict[str, Any] = Field(default_factory=dict)
    consensus: Any = None
    fallback: list[str] = Field(default_factory=list)

    # ─── NEW FIELD ───
    validator: Any = None  # Validator instance or config dict
    # Type hint is Any to avoid circular import; actual type is validators.Validator
```

**Backward compatibility**: `validator: Any = None` means existing recipes work unchanged.

### 5.2 Orchestrator Integration

**Modify `orchestrator._execute_step()`** to check contracts:

```python
async def _execute_step(
    self,
    run_id: str,
    step: StepDef,
    work_order: WorkOrder,
    state: RunState,
    run_artifacts_dir: Path
) -> StageResult:
    """Execute a single step with optional contract validation."""

    # ─── 1. PRECONDITION CHECK ───
    if step.validator:
        precondition_result = step.validator.validate_precondition(
            state=self._build_validation_state(state),
            inputs=self._build_step_inputs(step, state)
        )

        if not precondition_result.passed:
            # Precondition failed: don't execute, fail immediately
            self._emit_event(
                run_id=run_id,
                kind=EventKind.STEP_FAILED,
                step_id=step.step_id,
                data={
                    "error": "Precondition validation failed",
                    "violations": precondition_result.violations,
                    "confidence": precondition_result.confidence,
                    "metadata": precondition_result.metadata
                }
            )

            return StageResult(
                step_id=step.step_id,
                head_id=step.head_id,
                status=StepStatus.FAILED,
                error=f"Precondition failed: {'; '.join(precondition_result.violations)}",
                warnings=[f"Validation confidence: {precondition_result.confidence:.2f}"]
            )

    # ─── 2. EXECUTE STEP (existing logic) ───
    self._emit_event(run_id, EventKind.STEP_STARTED, step.step_id)

    try:
        # ... existing execution logic (prompt building, head invocation, etc.) ...
        output = await self._generate_output(step, state)

    except Exception as e:
        # Execution exception (network error, GPU OOM, etc.)
        return self._handle_execution_error(run_id, step, e)

    # ─── 3. POSTCONDITION CHECK ───
    if step.validator:
        postcondition_result = step.validator.validate_postcondition(
            state=self._build_validation_state(state),
            output=output
        )

        if not postcondition_result.passed:
            # Postcondition failed: discard output, retry if policy allows
            self._emit_event(
                run_id=run_id,
                kind=EventKind.STEP_FAILED,
                step_id=step.step_id,
                data={
                    "error": "Postcondition validation failed",
                    "violations": postcondition_result.violations,
                    "confidence": postcondition_result.confidence,
                    "metadata": postcondition_result.metadata,
                    "output_discarded": True
                }
            )

            # Check if retry allowed
            attempt = state.step_results.get(step.step_id, {}).get("attempt", 0) + 1
            max_attempts = step.retry_policy.get("max_attempts", 1)

            if attempt < max_attempts:
                # Retry with backoff
                await self._retry_step(run_id, step, state, attempt)
            else:
                # Max retries exceeded
                return StageResult(
                    step_id=step.step_id,
                    head_id=step.head_id,
                    status=StepStatus.FAILED,
                    error=f"Postcondition failed after {attempt} attempts: {'; '.join(postcondition_result.violations)}",
                    warnings=[f"Validation confidence: {postcondition_result.confidence:.2f}"]
                )

    # ─── 4. COMMIT ARTIFACT (existing logic) ───
    artifact_ref = await self._store_artifact(run_id, step.step_id, output)

    self._emit_event(
        run_id,
        EventKind.STEP_OUTPUT_WRITTEN,
        step.step_id,
        data={"artifact_id": artifact_ref.artifact_id}
    )

    # ─── 5. RETURN SUCCESS ───
    return StageResult(
        step_id=step.step_id,
        head_id=step.head_id,
        status=StepStatus.COMMITTED,
        output_artifacts=[artifact_ref],
        metrics={"validation_confidence": postcondition_result.confidence if step.validator else 1.0}
    )
```

**Helper methods**:

```python
def _build_validation_state(self, state: RunState) -> dict[str, Any]:
    """Build state dict for validators."""
    return {
        "run_id": state.run_id,
        "step_results": {
            step_id: {
                "outputs": result.outputs,
                "status": result.status.value,
                "artifacts": [a.artifact_id for a in result.output_artifacts]
            }
            for step_id, result in state.step_results.items()
        },
        "current_step_index": state.current_step_index
    }

def _build_step_inputs(self, step: StepDef, state: RunState) -> dict[str, Any]:
    """Build inputs dict for precondition validation."""
    inputs = {}

    for ref in step.input_refs:
        if ref == "user_input":
            inputs["user_input"] = state.work_order.inputs if state.work_order else {}
        else:
            # ref is artifact_id or step_id
            if ref in state.step_results:
                inputs[ref] = state.step_results[ref].outputs

    return inputs
```

### 5.3 Auto-Validator from output_schema

**Enhance `plan_normalizer.py`** to auto-create validators:

```python
def _auto_create_validators(steps: list[StepDef]) -> None:
    """Auto-create validators from output_schema if validator not explicitly set."""
    for step in steps:
        if step.validator is None and step.output_schema:
            # Create JSONSchemaValidator from output_schema
            step.validator = JSONSchemaValidator(schema=step.output_schema)

            # If schema has "confidence" field, wrap with ConfidenceValidator
            if "properties" in step.output_schema:
                if "confidence" in step.output_schema["properties"]:
                    step.validator = CompositeValidator(
                        validators=[
                            step.validator,
                            ConfidenceValidator(min_confidence=0.7)
                        ],
                        mode="all"
                    )
```

**Call in `normalize()`**:

```python
def normalize(work_order, head_manager, metrics=None, resource_monitor=None) -> WorkOrder:
    wo = work_order.model_copy(deep=True)
    _route_heads(wo.steps, head_manager, metrics, resource_monitor)
    _validate_head_ids(wo.steps, head_manager)
    _infer_dependencies(wo.steps)
    _auto_assign_fallbacks(wo.steps, head_manager)
    _auto_create_validators(wo.steps)  # ← NEW
    return wo
```

---

## 6. Example Contracts for Common Step Types

### 6.1 LLM Text Generation

```python
text_generation_validator = CompositeValidator(
    validators=[
        FormatValidator(min_length=10, max_length=5000),
        ContractValidator(
            postcondition=lambda state, output: (
                isinstance(output, str),
                "Output must be string"
            )
        )
    ],
    mode="all"
)
```

### 6.2 JSON Consensus Output

```python
consensus_validator = CompositeValidator(
    validators=[
        JSONSchemaValidator(schema={
            "type": "object",
            "required": ["verdict", "confidence"],
            "properties": {
                "verdict": {"type": "string", "enum": ["approve", "reject"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }),
        FormatValidator(max_length=750),  # MAKER red-flag
        ConfidenceValidator(min_confidence=0.7)
    ],
    mode="all"
)
```

### 6.3 Code Generation

```python
def check_valid_python(state, output):
    """Validate generated Python code compiles."""
    import ast
    try:
        ast.parse(output)
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

code_validator = CompositeValidator(
    validators=[
        FormatValidator(min_length=5),
        ContractValidator(
            postcondition=check_valid_python,
            name="python_syntax"
        )
    ],
    mode="all"
)
```

### 6.4 Coordinate Space (H2V Stage 3)

```python
def check_coordinate_metadata(state, output):
    """Ensure Stage 3 outputs include coordinate_space metadata."""
    if not isinstance(output, dict):
        return False, "Output must be dict with metadata"

    coord_space = output.get("coordinate_space")
    valid_spaces = ["page_resolution", "unet_fixed", "panel_relative"]

    if coord_space not in valid_spaces:
        return False, f"coordinate_space must be one of {valid_spaces}, got {coord_space}"

    if "resolution" not in output or len(output["resolution"]) != 2:
        return False, "Must include resolution as [width, height]"

    return True, "Valid coordinate metadata"

stage3_validator = ContractValidator(
    postcondition=check_coordinate_metadata,
    name="stage3_coordinate_contract"
)
```

---

## 7. Migration Strategy

### 7.1 Phase 1: Infrastructure (Week 1)

- [ ] Create `src/multihead/validators.py` with base classes
- [ ] Implement built-in validators (JSONSchema, Format, Confidence, Contract, Composite)
- [ ] Add `validator: Any = None` field to `StepDef` in `models.py`
- [ ] Update `orchestrator._execute_step()` to check preconditions/postconditions
- [ ] Add `_auto_create_validators()` to `plan_normalizer.py`
- [ ] Write unit tests for each validator type

### 7.2 Phase 2: Recipe Migration (Week 2)

**Automatic migration** (existing recipes work unchanged):
- Recipes with `output_schema` automatically get `JSONSchemaValidator`
- No code changes required for backward compatibility

**Manual enhancement** (opt-in improvements):
- Add explicit validators to critical steps in `architectural-decision.yaml`
- Wrap consensus steps with `CompositeValidator` (schema + confidence + length)
- Add custom `ContractValidator` for domain-specific invariants

**Example**: Update `config/recipes/architectural-decision.yaml`

```yaml
steps:
  - name: multi_model_critique
    consensus:
      heads: [qwen-llm, openai-gpt4o]
      strategy: weighted
      output_schema:  # ← Auto-creates JSONSchemaValidator
        type: object
        required: [verdict, concerns, suggestions, confidence]
        properties:
          verdict:
            type: string
            enum: [approve, approve_with_changes, reject]
          concerns:
            type: array
          suggestions:
            type: array
          confidence:
            type: number
            minimum: 0.0
            maximum: 1.0

    # ─── OPTIONAL: Explicit validator for additional constraints ───
    validator:
      type: composite
      mode: all
      validators:
        - type: json_schema  # Auto-created from output_schema above
        - type: format
          max_length: 750  # MAKER red-flag threshold
        - type: confidence
          min_confidence: 0.7
```

### 7.3 Phase 3: Metrics and Monitoring (Week 3)

- [ ] Add validation metrics to `MetricsCollector`
- [ ] Track: precondition_failures, postcondition_failures, avg_confidence
- [ ] Dashboard: Validation failure rates per step type
- [ ] Alerts: Spike in validation failures (> 10% for 5 consecutive runs)

---

## 8. Success Metrics

Track these to validate contract framework effectiveness:

| Metric | Baseline | Target (Phase 2) |
|--------|----------|-----------------|
| **Precondition failures** (% of starts) | N/A | <5% |
| **Postcondition failures** (% of completions) | ~10% (estimated) | <3% |
| **Downstream failures** (cascading errors) | ~20% | <5% |
| **Avg validation confidence** | N/A | >0.85 |
| **Time to debug failed run** | 2-4 hours | <30 min |

**Leading indicators**:
- Validation failures localize errors to specific steps (not vague "pipeline failed")
- Retry attempts decrease (higher quality outputs on first try)
- Downstream steps trust upstream postconditions (no defensive validation)

---

## 9. Future Enhancements

### 9.1 LLM-Based Validators (Phase 4)

**Semantic validation** via LLM calls:

```python
class SemanticValidator(Validator):
    """Validates output semantics using an LLM."""

    def __init__(self, head_id: str, validation_prompt: str):
        self.head_id = head_id
        self.validation_prompt = validation_prompt

    async def validate_postcondition(self, state, output) -> ValidationResult:
        # Delegate to LLM: "Does this output satisfy X constraint?"
        prompt = self.validation_prompt.format(output=output)
        response = await head_manager.generate(self.head_id, prompt)

        # Parse LLM's verdict (e.g., JSON {"valid": true, "confidence": 0.9})
        result = json.loads(response)
        return ValidationResult(
            passed=result["valid"],
            confidence=result.get("confidence", 0.5),
            violations=result.get("violations", [])
        )
```

**Use case**: Validate natural language outputs (e.g., "Does this summary capture the main points?")

### 9.2 Process Reward Models (Phase 5)

**Learned validators** from training data:

```python
class PRMValidator(Validator):
    """Process Reward Model validator (learned from examples)."""

    def __init__(self, model_path: Path):
        self.model = load_prm_model(model_path)

    def validate_postcondition(self, state, output) -> ValidationResult:
        # Score output quality using trained model
        score = self.model.score(output, context=state)

        return ValidationResult(
            passed=score > 0.7,
            confidence=score,
            violations=[] if score > 0.7 else [f"Quality score {score:.2f} below 0.7"]
        )
```

**Training**: Collect (input, output, quality_label) tuples from past runs, train classifier

### 9.3 Equivalence Clustering (Phase 6)

**Group semantically equivalent outputs** for consensus voting:

```python
class EquivalenceClusterer:
    """Clusters semantically equivalent outputs."""

    def cluster(self, outputs: list[str]) -> dict[str, list[str]]:
        """Group outputs into equivalence classes."""
        # Use embedding similarity or LLM-based classification
        embeddings = [self._embed(o) for o in outputs]
        clusters = self._cluster_by_similarity(embeddings, threshold=0.9)
        return clusters  # {"cluster_0": ["output_1", "output_2"], ...}
```

**Integration**: Enhance FIRST_TO_AHEAD to count votes per cluster, not exact string match

---

## 10. Open Questions

1. **Validator serialization**: How to serialize validators to YAML for recipes?
   - Option A: Validator registry with string IDs (e.g., `validator: "json_schema"`)
   - Option B: YAML-to-Python constructor (e.g., `validator: {type: composite, validators: [...]}`)
   - **Recommendation**: Option B (more flexible, easier debugging)

2. **Precondition vs Postcondition weighting**: Should precondition failures be cheaper (don't execute)?
   - **Recommendation**: Yes—precondition failures should abort without execution (no compute cost, just validation overhead)

3. **Confidence-based retry strategies**: Should low-confidence failures retry more than high-confidence?
   - **Recommendation**: Yes—adaptive retry based on `ValidationResult.confidence`:
     - confidence < 0.6: Retry with different head or prompt variation
     - confidence >= 0.9: Don't retry (likely fundamental issue, not transient)

4. **Validator composition order**: Does order matter in `CompositeValidator`?
   - **Recommendation**: Run cheap validators first (format, length) before expensive ones (LLM-based semantic validation)

---

**Status**: Ready for implementation
**Next**: Implement `validators.py` module, integrate into `orchestrator.py`, test with existing recipes
