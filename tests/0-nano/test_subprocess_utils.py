"""Tests for subprocess_utils.no_window_flags()."""
import subprocess
import sys
from unittest.mock import patch

from multihead.subprocess_utils import no_window_flags


def test_no_window_flags_windows():
    with patch.object(sys, "platform", "win32"):
        assert no_window_flags() == subprocess.CREATE_NO_WINDOW


def test_no_window_flags_non_windows():
    with patch.object(sys, "platform", "linux"):
        assert no_window_flags() == 0
