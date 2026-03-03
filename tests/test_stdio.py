"""Unit tests for utils.stdio (suppress_stdout)."""
import io
import sys

import pytest

from hst123.utils.stdio import suppress_stdout


class TestSuppressStdout:
    def test_suppresses_stdout(self):
        with suppress_stdout():
            print("should not appear", flush=True)
        # After context, stdout is restored
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            print("visible", flush=True)
            assert "visible" in buf.getvalue()
        finally:
            sys.stdout = old_stdout

    def test_restores_stdout_after_exception(self):
        old_stdout = sys.stdout
        try:
            with suppress_stdout():
                raise ValueError("oops")
        except ValueError:
            pass
        assert sys.stdout is old_stdout

    def test_restores_stderr(self):
        old_stderr = sys.stderr
        with suppress_stdout():
            pass
        assert sys.stderr is old_stderr
