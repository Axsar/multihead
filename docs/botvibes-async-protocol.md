# BotVibes ↔ MultiHead Async Messaging Protocol

**Version**: 1.1
**Date**: 2026-02-21
**Status**: Active
**Change**: Consolidated to main knowledge.db

---

## Standard Protocol: Claims-Based Messaging

**Method**: Structured claims in shared knowledge database
**Why**: Leaves audit trail, works offline, queryable, structured

---

## ✅ MAIN DATABASE (Use This)

**Default**: `$MULTIHEAD_DATA_DIR/knowledge.db` (defaults to `~/.multihead/knowledge.db`)

This is MultiHead's **unified knowledge database**. All async messaging uses this single database.

---

## 📬 For BotVibes: Poll for Messages FROM MultiHead

### Python (Recommended)

```python
from pathlib import Path
from multihead.knowledge_store import KnowledgeStore

db_path = Path(os.environ.get("MULTIHEAD_DATA_DIR", "~/.multihead")).expanduser() / "knowledge.db"
store = KnowledgeStore(db_path=db_path)

# Get unread messages
claims = store.list_claims(status='proposed', limit=10)

# Filter for BotVibes-related scopes
botvibes_msgs = [c for c in claims if 'botvibes' in c.scope.scope_id.lower()]

for msg in botvibes_msgs:
    print(f"Message: {msg.claim_id}")
    print(f"Subject: {msg.canonical.claim_key}")
    print(msg.statement)

    # Mark as read
    store.update_claim_status(msg.claim_id, ClaimStatus.ACCEPTED)
```

### SQL (Alternative)

```sql
SELECT claim_id, claim_status, scope_id, claim_key, statement, created_at
FROM claims
WHERE claim_status = 'proposed'
  AND scope_id LIKE '%botvibes%'
ORDER BY created_at DESC;
```

---

## 📮 For BotVibes: Send Messages TO MultiHead

Create claims with:
- **database**: Same `$MULTIHEAD_DATA_DIR/knowledge.db`
- **claim_status**: `PROPOSED` (new messages)
- **scope_id**: Include "multihead" (e.g., "botvibes-multihead-response")

```python
from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import (
    Claim, ClaimType, ClaimStatus, ClaimScope, ClaimCanonical,
    ScopeType, EntityRef, ValueObject, Provenance, Stability
)

db_path = Path(os.environ.get("MULTIHEAD_DATA_DIR", "~/.multihead")).expanduser() / "knowledge.db"
store = KnowledgeStore(db_path=db_path)

claim = Claim(
    claim_type=ClaimType.FACT,
    claim_status=ClaimStatus.PROPOSED,
    scope=ClaimScope(
        scope_type=ScopeType.PROJECT,
        scope_id="botvibes-multihead-response",  # Include 'multihead'
        visibility="shared"
    ),
    canonical=ClaimCanonical(
        claim_key="botvibes.response.topic",
        subject=EntityRef(...),
        predicate="your_predicate",
        object=ValueObject(...)
    ),
    statement="Your message here",
    # ... rest of claim
)

store.insert_claim(claim)
```

MultiHead polls for `scope_id LIKE '%multihead%'` in the same database.

---

## 📋 Message Lifecycle

1. **PROPOSED** → New, unread message
2. **ACCEPTED** → Read and acknowledged
3. **CONTESTED** → Disagreement or need clarification
4. **REJECTED** → Not applicable or declined

---

## 🗑️ Deprecated (v1.0)

These separate databases were overengineering and are now **ignored**:
- (previously `<data_dir>/knowledge/botvibes/knowledge.db`)
- (previously `<data_dir>/knowledge/multihead/knowledge.db`)

**Use only the main `knowledge.db` going forward.**

---

## ✅ Advantages of This Protocol

- **Audit Trail**: All messages permanently logged
- **Offline**: Works without server/network
- **Structured**: Queryable, filterable, searchable
- **Unified**: Single database, simple architecture
- **Flexible**: Supports questions, facts, decisions
- **Timestamped**: Automatic created_at/updated_at
- **Provenance**: Track who said what when

---

## 🔄 Fallback: ACP Tasks (When Online)

When both systems have ACP connection:
```python
# Faster, real-time messaging via ACP
task = await acp_bridge.create_task(
    capability="com.botvibes.notification",
    payload=message_data
)
```

But claims-based protocol is **primary** for reliability.

---

**Version History**:
- v1.0 (2026-02-21): Initial protocol with separate databases
- v1.1 (2026-02-21): Consolidated to main knowledge.db
