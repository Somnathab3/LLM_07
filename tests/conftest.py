"""Pytest configuration and fixtures"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture  
def sample_scat_data():
    """Sample SCAT data for testing"""
    return {
        "callsign": "AAL123",
        "fpl_plan_update": {
            "callsign": "AAL123",
            "route": "KORD DCT KLAS",
            "rfl": 370
        },
        "plots": [
            {
                "time_of_track": 1000,
                "I062/105": {"latitude": 41.978, "longitude": -87.904},
                "I062/136": {"measured_flight_level": 370}
            }
        ]
    }


@pytest.fixture
def mock_bluesky_client():
    """Mock BlueSky client for testing"""
    client = Mock()
    client.connected = True
    client.connect.return_value = True
    client.create_aircraft.return_value = True
    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = Mock()
    client.detect_conflicts.return_value = {"conflicts_detected": False, "conflicts": []}
    return client
