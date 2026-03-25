"""Tests for subprocess_utils.no_window_flags()."""
import subprocess
import sys
from unittest.mock import patch

from multihead.subprocess_utils import no_window_flags


def test_no_window_flags_windows():
    # CREATE_NO_WINDOW only exists on Windows; mock it if missing
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    with patch.object(sys, "platform", "win32"), \
         patch.object(subprocess, "CREATE_NO_WINDOW", flag, create=True):
        assert no_window_flags() == flag


def test_no_window_flags_non_windows():
    with patch.object(sys, "platform", "linux"):
        assert no_window_flags() == 0
