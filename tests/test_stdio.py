"""Tests for hst123.utils.stdio helpers."""

import os

from hst123.utils.stdio import limit_blas_threads_when_parallel


def test_limit_blas_threads_restores_env():
    key = "OMP_NUM_THREADS"
    prior = os.environ.get(key)
    try:
        os.environ[key] = "16"
        with limit_blas_threads_when_parallel(4):
            assert os.environ.get(key) == "1"
        assert os.environ.get(key) == "16"
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def test_limit_blas_threads_skips_when_single_core():
    key = "OMP_NUM_THREADS"
    prior = os.environ.get(key)
    try:
        os.environ[key] = "4"
        with limit_blas_threads_when_parallel(1):
            assert os.environ.get(key) == "4"
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior
