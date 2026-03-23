"""Unit tests for utils.stdio (suppress_stdout)."""
import io
import sys

import pytest

from hst123.utils.stdio import suppress_stdout, suppress_stdout_fd


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


class TestSuppressStdoutFd:
    def test_context_runs_and_restores(self):
        with suppress_stdout_fd():
            pass

    def test_os_write_stdout_hidden_from_pipe(self):
        import os
        r, w = os.pipe()
        saved = os.dup(1)
        try:
            os.dup2(w, 1)
            with suppress_stdout_fd():
                os.write(1, b"hidden\n")
            os.write(1, b"visible\n")
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(w)
        chunk = os.read(r, 200)
        os.close(r)
        assert b"hidden" not in chunk
        assert b"visible" in chunk
