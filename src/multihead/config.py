"""Configuration loading for MultiHead."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .models import AdapterKind, HeadManifest, StepDef, WorkOrder

logger = logging.getLogger(__name__)

# Key written/read when persisting the shared data directory.
_DATA_DIR_ENV_KEY = "MULTIHEAD_DATA_DIR"

# User-level config location
_USER_CONFIG_DIR = Path.home() / ".multihead" / "config"


def _bundled_config_path() -> Path:
    """Return path to the bundled default_config/ inside the package."""
    return Path(__file__).parent / "default_config"


def resolve_config_dir(explicit: str | None = None) -> Path:
    """Find the best config directory.

    Resolution order:
    1. Explicit path (--config-dir or MULTIHEAD_CONFIG_DIR)
    2. ./config/ if it contains heads.yaml (repo checkout / dev mode)
    3. ~/.multihead/config/ if it contains heads.yaml (user config)
    4. Bundled default_config/ (fresh install — auto-copied to user dir)
    """
    # 1. Explicit override
    if explicit:
        return Path(explicit)

    # 2. Repo checkout (dev mode)
    local = Path("config")
    if (local / "heads.yaml").exists() or (local / "solvers.yaml").exists():
        return local

    # 3. User config dir
    if (_USER_CONFIG_DIR / "heads.yaml").exists() or (_USER_CONFIG_DIR / "solvers.yaml").exists():
        return _USER_CONFIG_DIR

    # 4. Bundled defaults — copy to user dir for writable config
    bundled = _bundled_config_path()
    if (bundled / "heads.yaml").exists():
        _bootstrap_user_config(bundled)
        return _USER_CONFIG_DIR

    # Nothing found — return user dir anyway (init wizard will handle)
    return _USER_CONFIG_DIR


def _bootstrap_user_config(bundled: Path) -> None:
    """Copy bundled defaults to ~/.multihead/config/ on first run."""
    import shutil

    if _USER_CONFIG_DIR.exists():
        return  # Don't overwrite existing (even partial) user config

    logger.info("First run: copying default config to %s", _USER_CONFIG_DIR)
    try:
        shutil.copytree(bundled, _USER_CONFIG_DIR)
    except Exception as e:
        logger.warning("Failed to copy default config: %s", e)


class Settings(BaseSettings):
    data_dir: Path = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead")))
    api_host: str = "127.0.0.1"
    api_port: int = 7337
    default_checkpoint_mode: str = "sync"
    config_dir: Path = Path("config")
    mesh_secret: str | None = None
    core_head_id: str = "mock-llm"

    model_config = {"env_prefix": "MULTIHEAD_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("api_host")
    @classmethod
    def warn_public_binding(cls, v: str) -> str:
        if v in ("0.0.0.0", "::"):
            logger.warning(
                "API server binding to all interfaces (%s) — "
                "no authentication on endpoints! Use 127.0.0.1 for local-only access.",
                v,
            )
        return v

    @field_validator("api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"api_port must be 1-65535, got {v}")
        return v

    @field_validator("mesh_secret")
    @classmethod
    def validate_secret(cls, v: str | None) -> str | None:
        if v is not None and len(v) < 16:
            raise ValueError("mesh_secret must be at least 16 characters")
        return v

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def knowledge_db_path(self) -> Path:
        return self.data_dir / "knowledge.db"

    @property
    def packs_dir(self) -> Path:
        return self.data_dir / "packs"

    @property
    def nightshift_output_dir(self) -> Path:
        return self.data_dir / "nightshift"

    @property
    def memory_file(self) -> Path:
        return self.data_dir / "MEMORY_MULTIHEAD.md"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def context_dir(self) -> Path:
        return self.data_dir / "context"

    def ensure_dirs(self) -> None:
        """Create standard data directory structure.

        $MULTIHEAD_DATA_DIR/
          knowledge.db      — claims, events, evidence
          state.db          — run state, head state
          runs/             — run event logs (JSONL)
          artifacts/        — content-addressed artifact storage
          packs/            — context packs for agents
          nightshift/       — pipeline output, reports, batches
          skills/           — registered skills
          sessions/         — conversation harvest manifests
          embeddings/       — vector embedding cache
          context/          — generated context files
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        self.nightshift_output_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def resolve_core_head_id(self, available_heads: dict[str, Any]) -> str:
        """Resolve core_head_id to an actual available head.

        Priority: configured value > first LLM head > first head > "mock-llm".
        """
        if self.core_head_id in available_heads:
            return self.core_head_id
        # First LLM-kind head
        for hid, manifest in available_heads.items():
            if getattr(manifest, "kind", None) == "llm":
                return hid
        # Any head
        if available_heads:
            return next(iter(available_heads))
        return "mock-llm"


def check_adapter_available(adapter: str) -> tuple[bool, str]:
    """Check if an adapter's dependencies are available.

    Returns (available, reason) tuple.
    """
    import shutil

    adapter = adapter.lower()

    if adapter in ("mock", "deterministic"):
        return True, "built-in"

    if adapter == "transformers":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True, "torch + transformers found"
        except ImportError:
            return False, "requires torch and transformers (pip install .[gpu])"

    if adapter == "claude":
        if shutil.which("claude"):
            return True, "claude CLI found"
        return False, "requires Claude CLI on PATH (https://docs.anthropic.com/claude-code)"

    if adapter == "claude_agent_sdk":
        try:
            import claude_code_sdk  # noqa: F401
            return True, "claude-agent-sdk found"
        except ImportError:
            return False, "requires claude-agent-sdk (pip install .[claude])"

    if adapter == "claude_session":
        if shutil.which("claude"):
            return True, "claude CLI found"
        return False, "requires Claude CLI on PATH"

    if adapter == "openai":
        import os
        if os.environ.get("OPENAI_API_KEY"):
            return True, "OPENAI_API_KEY set"
        return False, "requires OPENAI_API_KEY environment variable"

    if adapter == "ollama":
        if shutil.which("ollama"):
            return True, "ollama found"
        return False, "requires ollama on PATH (https://ollama.ai)"

    if adapter == "vllm":
        try:
            import vllm  # noqa: F401
            return True, "vllm found"
        except ImportError:
            return False, "requires vllm (pip install vllm)"

    if adapter == "embedding":
        try:
            import sentence_transformers  # noqa: F401
            return True, "sentence-transformers found"
        except ImportError:
            return False, "requires sentence-transformers"

    if adapter in ("mesh", "botvibes"):
        return True, "network adapter (availability checked at runtime)"

    return True, "unknown adapter, assuming available"


def validate_heads(heads: dict[str, "HeadManifest"]) -> tuple[dict[str, "HeadManifest"], list[str]]:
    """Validate adapter availability for all heads.

    Returns (valid_heads, warnings) where valid_heads has broken heads removed
    and warnings is a list of human-readable warning strings.
    """
    valid = {}
    warnings = []
    for head_id, manifest in heads.items():
        adapter_str = str(manifest.adapter.value) if hasattr(manifest.adapter, "value") else str(manifest.adapter)
        available, reason = check_adapter_available(adapter_str)
        if available:
            valid[head_id] = manifest
        else:
            warnings.append(f"Head '{head_id}': adapter '{adapter_str}' not available — {reason}")
    return valid, warnings


def set_data_dir_in_env(shared_dir: Path | str, env_path: Path | str = ".env") -> None:
    """Write MULTIHEAD_DATA_DIR to the .env file.

    If the key already exists it is updated in-place; otherwise it is appended.
    The file is created with a minimal header when it does not yet exist.

    Args:
        shared_dir: Absolute path to the shared data directory.
        env_path: Path to the .env file (default: ``.env`` in the current
            working directory, which mirrors pydantic-settings' default).
    """
    env_path = Path(env_path)
    new_value = str(shared_dir)
    key_line = f"{_DATA_DIR_ENV_KEY}={new_value}"

    if env_path.exists():
        original = env_path.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"^{re.escape(_DATA_DIR_ENV_KEY)}\s*=.*$", re.MULTILINE
        )
        if pattern.search(original):
            updated = pattern.sub(key_line, original)
        else:
            # Append after a blank line if the file doesn't end with one.
            sep = "" if original.endswith("\n\n") else ("\n" if original.endswith("\n") else "\n\n")
            updated = original + sep + key_line + "\n"
        env_path.write_text(updated, encoding="utf-8")
    else:
        env_path.write_text(
            "# MultiHead environment config\n"
            f"{key_line}\n",
            encoding="utf-8",
        )

    logger.info("Wrote %s=%s to %s", _DATA_DIR_ENV_KEY, new_value, env_path)


def reload_settings(env_path: Path | str = ".env") -> "Settings":
    """Return a fresh Settings instance that reflects the current .env file.

    Call this after :func:`set_data_dir_in_env` (or any other .env mutation)
    to pick up the new values without restarting the process.  The returned
    object is independent — callers are responsible for propagating it to the
    rest of the application (e.g. replacing a module-level ``settings``
    singleton).

    Args:
        env_path: Path to the .env file passed through to Settings.

    Returns:
        A newly constructed :class:`Settings` instance.
    """
    return Settings(_env_file=str(env_path))  # type: ignore[call-arg]


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ``${ENV_VAR}`` patterns in config values.

    Supports strings, dicts, and lists.  Unresolvable variables are left
    as-is (so the adapter can raise a clear error).
    """
    if isinstance(value, str):
        import os as _os
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: _os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_heads(config_dir: Path) -> dict[str, HeadManifest]:
    """Load head/solver manifests from solvers.yaml (Phase 1) or heads.yaml (legacy).

    Phase 1: Prefers solvers.yaml with capability descriptions.
    Fallback: Uses heads.yaml for backward compatibility.

    Environment variables in ``extra`` fields (``${VAR}``) are resolved at
    load time so adapters receive concrete values.
    """
    # Phase 1: Try solvers.yaml first (new format with capabilities)
    solvers_file = config_dir / "solvers.yaml"
    if solvers_file.exists():
        try:
            with open(solvers_file) as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict) and "solvers" in data:
                logger.debug("Loading solvers from %s", solvers_file)
                heads = {}
                for i, item in enumerate(data.get("solvers", [])):
                    try:
                        # Convert solver_id to head_id for backward compatibility
                        if "solver_id" in item:
                            item["head_id"] = item.pop("solver_id")

                        # Resolve ${ENV_VAR} patterns in extra and endpoint
                        if "extra" in item:
                            item["extra"] = _resolve_env_vars(item["extra"])
                        if "endpoint" in item:
                            item["endpoint"] = _resolve_env_vars(item["endpoint"])

                        # Parse capabilities if present
                        if "capabilities" in item:
                            from .models import Capability
                            item["capabilities"] = Capability(**item["capabilities"])

                        manifest = HeadManifest(**item)
                        heads[manifest.head_id] = manifest
                    except Exception as e:
                        logger.error("Invalid solver at index %d in %s: %s", i, solvers_file, e)
                        raise ValueError(f"Invalid solver at index {i}: {e}") from e
                logger.debug("Loaded %d solvers from %s", len(heads), solvers_file)
                return heads
        except yaml.YAMLError as e:
            logger.error("Failed to parse %s: %s", solvers_file, e)
            raise ValueError(f"Invalid YAML in {solvers_file}: {e}") from e

    # Fallback: Try heads.yaml (legacy format)
    heads_file = config_dir / "heads.yaml"
    if not heads_file.exists():
        logger.warning("Neither solvers.yaml nor heads.yaml found in %s", config_dir)
        return {}

    try:
        with open(heads_file) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Failed to parse %s: %s", heads_file, e)
        raise ValueError(f"Invalid YAML in {heads_file}: {e}") from e

    if not isinstance(data, dict) or "heads" not in data:
        logger.warning("heads.yaml has no 'heads' key, returning empty")
        return {}

    logger.info("Loading heads from %s (legacy format, no capabilities)", heads_file)
    heads = {}
    for i, item in enumerate(data.get("heads", [])):
        try:
            manifest = HeadManifest(**item)
            heads[manifest.head_id] = manifest
        except Exception as e:
            logger.error("Invalid head manifest at index %d in %s: %s", i, heads_file, e)
            raise ValueError(f"Invalid head at index {i}: {e}") from e
    logger.info("Loaded %d heads from %s", len(heads), heads_file)
    return heads


def load_recipe(config_dir: Path, recipe_name: str) -> WorkOrder:
    """Load a recipe YAML and return a WorkOrder."""
    recipe_file = config_dir / "recipes" / f"{recipe_name}.yaml"
    if not recipe_file.exists():
        raise FileNotFoundError(f"Recipe not found: {recipe_file}")
    try:
        with open(recipe_file) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Failed to parse recipe %s: %s", recipe_file, e)
        raise ValueError(f"Invalid YAML in recipe {recipe_name}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Recipe {recipe_name} must be a YAML mapping, got {type(data).__name__}")

    steps = data.get("steps", [])
    if not steps:
        logger.warning("Recipe %s has no steps defined", recipe_name)

    try:
        parsed_steps = [StepDef(**s) for s in steps]
    except Exception as e:
        raise ValueError(f"Invalid step in recipe {recipe_name}: {e}") from e

    # Hydrate validators from YAML config dicts into Validator instances
    for step_dict, step_def in zip(steps, parsed_steps):
        validator_config = step_dict.get("validator")
        if isinstance(validator_config, dict):
            try:
                from .validators import create_validator_from_config

                step_def.validator = create_validator_from_config(validator_config)
                logger.debug(
                    "Hydrated validator for step %s: %s",
                    step_def.name, type(step_def.validator).__name__,
                )
            except Exception as e:
                logger.warning(
                    "Failed to hydrate validator for step %s in recipe %s: %s",
                    step_def.name, recipe_name, e,
                )
                step_def.validator = None

    return WorkOrder(
        goal=data.get("goal", recipe_name),
        inputs=data.get("inputs", {}),
        steps=parsed_steps,
        budgets=data.get("budgets", {}),
    )
