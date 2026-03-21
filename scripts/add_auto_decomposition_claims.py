#!/usr/bin/env python3
"""Add claims about auto-decomposition feature to knowledge database."""

import os
from pathlib import Path
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    Provenance,
    Record,
    ScopeType,
    Stability,
    ValueObject,
)

# Initialize knowledge store
knowledge_db_path = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "knowledge.db"
ks = KnowledgeStore(knowledge_db_path)

# Project ID for MultiHead
project_id = "multihead"

# Create a record for the source code
now = datetime.now(timezone.utc)
record = Record(
    uri="file://src/multihead/auto_decomposition.py",
    sha256="",
    mime="text/x-python",
    captured_at=now,
    created_at=now,
)
record = ks.insert_record(record)

# Create evidence pointer for the record
evidence = EvidencePointer(
    record_id=record.record_id,
    uri=record.uri,
    quote="AutoDecomposer implementation with DAG inference",
    captured_at=now,
)
evidence = ks.insert_evidence(evidence)

# Standard provenance for all claims
provenance = Provenance(
    produced_by={"kind": "user", "id": "manual_script"},
    toolchain=[{"tool": "scripts/add_auto_decomposition_claims.py", "version": "1.0"}],
)

# Standard scope for all claims
scope = ClaimScope(
    scope_type=ScopeType.PROJECT,
    scope_id=project_id,
    visibility="private",
    valid_from=now,
)

# Claims about auto-decomposition
claims_data = [
    {
        "key": "multihead.auto_decomposition.architecture",
        "subject": {"entity_type": "module", "entity_id": "auto_decomposition", "label": "AutoDecomposer"},
        "predicate": "wraps_and_enhances",
        "object": {"value_type": "string", "value": "TaskDecomposer with DAG inference, atomicity validation, completeness validation, and research feature integration"},
        "statement": "AutoDecomposer wraps TaskDecomposer and adds DAG inference, atomicity validation, completeness validation, and research feature integration",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.95,
    },
    {
        "key": "multihead.auto_decomposition.dag_inference",
        "subject": {"entity_type": "class", "entity_id": "StepDependencyAnalyzer", "label": "StepDependencyAnalyzer"},
        "predicate": "infers_dependencies_from",
        "object": {"value_type": "string", "value": "file operations, action ordering, and artifact flow to enable parallel execution"},
        "statement": "StepDependencyAnalyzer infers DAG dependencies from file operations, action ordering, and artifact flow to enable parallel execution",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.9,
    },
    {
        "key": "multihead.auto_decomposition.dag_creation",
        "subject": {"entity_type": "method", "entity_id": "to_work_order_with_dag", "label": "AutoDecomposer.to_work_order_with_dag()"},
        "predicate": "creates",
        "object": {"value_type": "string", "value": "WorkOrders with true DAG dependencies instead of sequential chains"},
        "statement": "AutoDecomposer.to_work_order_with_dag() creates WorkOrders with true DAG dependencies instead of sequential chains",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.9,
    },
    {
        "key": "multihead.auto_decomposition.atomicity_validation",
        "subject": {"entity_type": "class", "entity_id": "AtomicityValidator", "label": "AtomicityValidator"},
        "predicate": "validates",
        "object": {"value_type": "string", "value": "steps are atomic (m=1) following MAKER research by checking single targets, no multi-action words, and length heuristics"},
        "statement": "AtomicityValidator validates steps are atomic (m=1) following MAKER research by checking single targets, no multi-action words, and length heuristics",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.85,
    },
    {
        "key": "multihead.auto_decomposition.completeness_validation",
        "subject": {"entity_type": "class", "entity_id": "CompletenessValidator", "label": "CompletenessValidator"},
        "predicate": "ensures",
        "object": {"value_type": "string", "value": "decomposition covers goal by checking keyword coverage, standard phases, and detecting under-decomposition"},
        "statement": "CompletenessValidator ensures decomposition covers goal by checking keyword coverage, standard phases, and detecting under-decomposition",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.8,
    },
    {
        "key": "multihead.auto_decomposition.research_features",
        "subject": {"entity_type": "class", "entity_id": "ResearchFeatureIntegrator", "label": "ResearchFeatureIntegrator"},
        "predicate": "auto_enables",
        "object": {"value_type": "string", "value": "ToT for exploratory steps, PRM for implementation steps, and Reflection for verification steps"},
        "statement": "ResearchFeatureIntegrator auto-enables ToT for exploratory steps, PRM for implementation steps, and Reflection for verification steps",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.9,
    },
    {
        "key": "multihead.auto_decomposition.file_dependencies",
        "subject": {"entity_type": "algorithm", "entity_id": "dependency_inference", "label": "File dependency inference"},
        "predicate": "uses_rule",
        "object": {"value_type": "string", "value": "write-after-read: if step B writes file X and step A reads file X, then B depends on A"},
        "statement": "File dependencies are inferred as write-after-read: if step B writes file X and step A reads file X, then B depends on A",
        "type": ClaimType.DEFINITION,
        "confidence": 1.0,
        "importance": 0.85,
    },
    {
        "key": "multihead.auto_decomposition.action_ordering",
        "subject": {"entity_type": "algorithm", "entity_id": "dependency_inference", "label": "Action ordering inference"},
        "predicate": "uses_rule",
        "object": {"value_type": "string", "value": "test steps depend on edit/create steps, verify steps depend on test steps"},
        "statement": "Action ordering dependencies: test steps depend on edit/create steps, verify steps depend on test steps",
        "type": ClaimType.DEFINITION,
        "confidence": 1.0,
        "importance": 0.8,
    },
    {
        "key": "multihead.auto_decomposition.atomic_phrases",
        "subject": {"entity_type": "class", "entity_id": "AtomicityValidator", "label": "AtomicityValidator"},
        "predicate": "whitelists",
        "object": {"value_type": "string", "value": "atomic phrases like 'read and understand', 'run tests', 'read test cases' as single atomic actions"},
        "statement": "Atomic phrases like 'read and understand', 'run tests', 'read test cases' are whitelisted as single atomic actions",
        "type": ClaimType.DECISION,
        "confidence": 1.0,
        "importance": 0.7,
    },
    {
        "key": "multihead.auto_decomposition.multiple_targets",
        "subject": {"entity_type": "rule", "entity_id": "atomicity_m1", "label": "Atomicity m=1 rule"},
        "predicate": "requires",
        "object": {"value_type": "string", "value": "steps with multiple target files are considered non-atomic and violate the m=1 principle"},
        "statement": "Steps with multiple target files are considered non-atomic and violate the m=1 principle",
        "type": ClaimType.CONSTRAINT,
        "confidence": 1.0,
        "importance": 0.85,
    },
    {
        "key": "multihead.auto_decomposition.research_milestone",
        "subject": {"entity_type": "feature", "entity_id": "auto_decomposition", "label": "Auto-decomposition"},
        "predicate": "implements",
        "object": {"value_type": "string", "value": "all six research features: Reflection, ToT, PRM, Auto-Decomposition, Recipe Learning, and Auto-Benchmarking"},
        "statement": "Auto-decomposition implements all six research features: Reflection, ToT, PRM, Auto-Decomposition, Recipe Learning, and Auto-Benchmarking",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.95,
    },
    {
        "key": "multihead.auto_decomposition.research_basis",
        "subject": {"entity_type": "feature", "entity_id": "auto_decomposition", "label": "Auto-decomposition"},
        "predicate": "based_on",
        "object": {"value_type": "string", "value": "MAKER (Maximal Agentic Decomposition m=1), K-Step Reasoning (atomic steps with verification), and ADaPT (As-Needed Decomposition)"},
        "statement": "Auto-decomposition is based on MAKER (Maximal Agentic Decomposition m=1), K-Step Reasoning (atomic steps with verification), and ADaPT (As-Needed Decomposition)",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.9,
    },
    {
        "key": "multihead.auto_decomposition.parallel_execution",
        "subject": {"entity_type": "feature", "entity_id": "parallel_execution", "label": "Parallel execution"},
        "predicate": "enabled_by",
        "object": {"value_type": "string", "value": "analyzing step independence: steps with no dependencies can execute concurrently"},
        "statement": "Parallel execution is enabled by analyzing step independence: steps with no dependencies can execute concurrently",
        "type": ClaimType.FACT,
        "confidence": 1.0,
        "importance": 0.9,
    },
    {
        "key": "multihead.auto_decomposition.feature_config_storage",
        "subject": {"entity_type": "implementation", "entity_id": "feature_configs", "label": "Research feature configs"},
        "predicate": "stored",
        "object": {"value_type": "string", "value": "separately from TaskNode (which doesn't support extra field) and applied during WorkOrder conversion"},
        "statement": "Research feature configs are stored separately from TaskNode (which doesn't support extra field) and applied during WorkOrder conversion",
        "type": ClaimType.DECISION,
        "confidence": 1.0,
        "importance": 0.75,
    },
]

# Add claims to knowledge store
print(f"Adding {len(claims_data)} claims about auto-decomposition to knowledge database...")
added_count = 0

for claim_data in claims_data:
    try:
        # Create claim
        claim = Claim(
            claim_status=ClaimStatus.PROPOSED,  # Start as proposed
            claim_type=claim_data["type"],
            scope=scope,
            canonical=ClaimCanonical(
                claim_key=claim_data["key"],
                subject=EntityRef(**claim_data["subject"]),
                predicate=claim_data["predicate"],
                object=ValueObject(**claim_data["object"]),
            ),
            statement=claim_data["statement"],
            rationale="Extracted from auto-decomposition implementation",
            confidence=claim_data["confidence"],
            stability=Stability.STABLE,
            importance=claim_data["importance"],
            provenance=provenance,
        )

        # Insert claim
        claim = ks.insert_claim(claim)

        # Link evidence
        ks.link_claim_evidence(claim.claim_id, evidence.evidence_id, stance="supports")

        # Accept claim (will auto-promote to ACCEPTED since it has evidence)
        ks.accept_claim(claim.claim_id)

        added_count += 1
        print(f"✓ Added: {claim_data['statement'][:80]}...")

    except Exception as e:
        print(f"✗ Failed to add claim: {e}")
        print(f"  Statement: {claim_data['statement'][:60]}...")

print(f"\n✅ Successfully added {added_count}/{len(claims_data)} claims to knowledge database")
print(f"📍 Database location: {knowledge_db_path}")
