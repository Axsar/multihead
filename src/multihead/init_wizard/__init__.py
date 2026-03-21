"""Init wizard: detect hardware, check adapters, and generate config."""

from .capabilities import (
    DEFAULT_LLM_CAPABILITIES,
    DEFAULT_VLM_CAPABILITIES,
    MODEL_CAPABILITIES,
    infer_capabilities,
)
from .config_gen import (
    generate_env_file,
    generate_heads_yaml,
    generate_solvers_yaml,
    suggest_config,
)
from .hardware import check_adapters, detect_hardware
from .mesh import MeshSetupWizard
from .models import SENTINEL_FILENAME, AdapterStatus, HardwareProfile
from .wizard import InitWizard

__all__ = [
    # Models / constants
    "AdapterStatus",
    "HardwareProfile",
    "SENTINEL_FILENAME",
    # Hardware
    "check_adapters",
    "detect_hardware",
    # Capabilities
    "DEFAULT_LLM_CAPABILITIES",
    "DEFAULT_VLM_CAPABILITIES",
    "MODEL_CAPABILITIES",
    "infer_capabilities",
    # Config generation
    "generate_env_file",
    "generate_heads_yaml",
    "generate_solvers_yaml",
    "suggest_config",
    # Wizards
    "InitWizard",
    "MeshSetupWizard",
]
