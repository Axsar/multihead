"""User signal classification for conversation extraction.

Detects emotional valence, intent, and importance signals from user
messages in Claude Code conversation transcripts. Pure heuristic —
no LLM cost, runs on every message.

Signals extracted:
- valence: frustrated / correction / excited / neutral / low_signal
- intent: directive / question / approval / rejection / context
- weight: 0.0-1.0 extraction priority
"""

from __future__ import annotations

import re

# --- Valence detection ---

_FRUSTRATED_PATTERNS = [
    re.compile(r'\b(I already told you|I said|I asked|as I mentioned)\b', re.I),
    re.compile(r'\b(stop|quit|enough)\s+(doing|assuming|wasting|trying)', re.I),
    re.compile(r'\b(still|again|already)\s+(not|broken|failing|wrong|same)', re.I),
    re.compile(r'\b(this|that)\s+(doesn\'t|isn\'t|won\'t|can\'t)\s+work', re.I),
    re.compile(r'\bwhy\s+(would|did|do|are|is)\s+you\b', re.I),
    re.compile(r'\bso annoying\b', re.I),
]

_EXCITED_PATTERNS = [
    re.compile(r'(?:^|\s)(!{2,})', re.M),
    re.compile(r'\b(perfect|amazing|excellent|brilliant|awesome|great job)\b', re.I),
    re.compile(r'\b(that\'s exactly|this is great|love it|nailed it)\b', re.I),
    re.compile(r'\b(yes!|nice!|wow)\b', re.I),
    re.compile(r'\b(breakthrough|finally|it works)\b', re.I),
]

_CORRECTION_PATTERNS = [
    re.compile(r'^no[,.\s]', re.I | re.M),
    re.compile(r'\b(that\'s wrong|that\'s not|that\'s incorrect|not what I)\b', re.I),
    re.compile(r'\b(I meant|I said|what I actually|I don\'t mean)\b', re.I),
    re.compile(r'\b(don\'t|do not)\s+(do|change|modify|add|remove|use)\b', re.I),
    re.compile(r'\b(wrong|incorrect|mistake|misunderstand)\b', re.I),
    re.compile(r'\bnot\s+(?:that|this|what|the)\b.*\binstead\b', re.I),
]

_DIRECTION_CHANGE_PATTERNS = [
    re.compile(r'\b(actually|wait|hold on|scratch that|forget that|never mind)\b', re.I),
    re.compile(r'\b(let me rethink|on second thought|changed my mind)\b', re.I),
]

# --- Intent detection ---

_DIRECTIVE_STARTS = re.compile(
    r'^(fix|add|create|update|implement|change|remove|run|test|deploy|build|'
    r'write|make|refactor|move|rename|delete|install|configure|set up|wire|'
    r'solve|commit|push|do|go|start|stop|restart|check|look|find|read|show)',
    re.I | re.M,
)

_QUESTION_PATTERN = re.compile(r'\?(?:\s|$)')
_QUESTION_START = re.compile(
    r'^(what|how|why|where|when|can|could|is|are|do|does|did|will|would|should|which)',
    re.I | re.M,
)

_APPROVAL_PATTERNS = [
    re.compile(r'^(yes|yep|yah|yeah|ok|okay|sure|lgtm|approved?|go|go ahead|ship it|sounds good|looks good|correct|right|exactly)\b', re.I),
    re.compile(r'\b(commit|push|do it|proceed|continue)\s*$', re.I),
]

_REJECTION_PATTERNS = [
    re.compile(r'^(no|nope|nah|disagree)\b', re.I),
    re.compile(r'\b(reject|don\'t|stop|cancel|abort|revert)\b', re.I),
]

# --- Caps detection ---
_CAPS_WORDS = re.compile(r'\b[A-Z]{3,}\b')


def classify_valence(text: str) -> str:
    """Classify emotional valence of a user message.

    Returns: frustrated, correction, excited, direction_change, neutral, low_signal
    """
    if not text or len(text.strip()) < 3:
        return "low_signal"

    # Strip tool results — these are machine output, not user emotion
    clean = re.sub(r'\[Tool result\]:.*?(?=\[(?:User|Tool|Assistant)\]:|$)', '', text, flags=re.DOTALL).strip()
    # Strip system notifications and context continuations
    clean = re.sub(r'<task-notification>.*?</task-notification>', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'<system-reminder>.*?</system-reminder>', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'This session is being continued from a previous conversation.*?$', '', clean, flags=re.DOTALL).strip()

    if not clean or len(clean.strip()) < 3:
        return "low_signal"

    # Use cleaned text for classification
    text = clean

    # Short acknowledgments
    stripped = text.strip().lower().rstrip(".,!?")
    if stripped in ("ok", "yah", "yeah", "yep", "sure", "k", "thanks", "got it",
                    "sounds good", "right", "fine", "cool"):
        return "low_signal"

    # Check for caps shouting (3+ capitalized words)
    caps_count = len(_CAPS_WORDS.findall(text))

    # Frustration (highest priority — captures corrections too)
    frustration_score = sum(1 for p in _FRUSTRATED_PATTERNS if p.search(text))
    if frustration_score >= 1 or caps_count >= 3:
        return "frustrated"

    # Correction
    if any(p.search(text) for p in _CORRECTION_PATTERNS):
        return "correction"

    # Direction change
    if any(p.search(text) for p in _DIRECTION_CHANGE_PATTERNS):
        return "direction_change"

    # Excitement
    if any(p.search(text) for p in _EXCITED_PATTERNS):
        return "excited"

    return "neutral"


def classify_intent(text: str) -> str:
    """Classify the intent of a user message.

    Returns: directive, question, approval, rejection, context
    """
    if not text or len(text.strip()) < 3:
        return "context"

    # Approval
    if any(p.search(text) for p in _APPROVAL_PATTERNS):
        return "approval"

    # Rejection
    if any(p.search(text) for p in _REJECTION_PATTERNS):
        return "rejection"

    # Question
    if _QUESTION_PATTERN.search(text) or _QUESTION_START.match(text.strip()):
        return "question"

    # Directive
    if _DIRECTIVE_STARTS.match(text.strip()):
        return "directive"

    return "context"


def compute_weight(
    valence: str,
    intent: str,
    text_length: int,
    is_user: bool = True,
) -> float:
    """Compute extraction weight for a message.

    Higher weight = more important for knowledge extraction.
    User messages get higher base weight than assistant messages.
    """
    if not is_user:
        # Assistant output — lower base, boosted by user validation
        return 0.3

    # Base weight by valence
    valence_weights = {
        "frustrated": 0.95,      # Frustration = correction = ground truth
        "correction": 0.90,      # Explicit correction
        "excited": 0.80,         # Validates preceding output
        "direction_change": 0.70, # Important pivot
        "neutral": 0.60,         # Standard input
        "low_signal": 0.10,      # "ok", "yah"
    }

    # Intent modifier
    intent_modifiers = {
        "directive": 0.10,    # User explicitly asked for something
        "rejection": 0.10,   # User said no — important
        "question": -0.05,   # Questions = gaps, not facts
        "approval": 0.0,     # Neutral
        "context": 0.0,      # Neutral
    }

    # Length bonus — longer messages = more investment
    length_bonus = min(0.10, text_length / 1000)

    weight = valence_weights.get(valence, 0.5)
    weight += intent_modifiers.get(intent, 0.0)
    weight += length_bonus

    return round(min(1.0, max(0.0, weight)), 2)


def classify_message(text: str, is_user: bool = True) -> dict:
    """Full classification of a message.

    Returns dict with valence, intent, weight, and raw text length.
    """
    valence = classify_valence(text) if is_user else "neutral"
    intent = classify_intent(text) if is_user else "context"
    weight = compute_weight(valence, intent, len(text), is_user)

    return {
        "valence": valence,
        "intent": intent,
        "weight": weight,
        "text_length": len(text),
        "is_user": is_user,
    }
