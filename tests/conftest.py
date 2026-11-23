"""
Pytest configuration and global fixtures for ModelMux.
"""

import pytest

from app.reliability.circuit_breaker import reset_circuit_breakers


@pytest.fixture(autouse=True)
def clean_circuit_breakers():
    """Ensure clean circuit breaker state before and after each test."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
