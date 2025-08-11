# LLM_ATC7: AI-Driven Air Traffic Control Conflict Detection & Resolution

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

A comprehensive system for simulating and quantifying ML-based hallucination effects on safety margins in enroute air traffic control using the Swedish Civil Air Traffic Control (SCAT) dataset, BlueSky simulator, and Large Language Models.

## 🎯 Overview

LLM_ATC7 implements a complete conflict detection and resolution pipeline that combines:

- **Real Flight Data**: SCAT dataset with ~170k flights and actual ATC clearances
- **High-Fidelity Simulation**: BlueSky open-source ATC simulator
- **AI Decision Making**: Multi-stage LLM reasoning (Detector → Resolver → Verifier) 
- **Safety Analysis**: Comprehensive metrics including TBAS, LAT, RAT scores
- **Memory System**: Experience library for continuous learning

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Somnathab3/LLM_ATC7.git
cd LLM_ATC7

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install BlueSky simulator
pip install bluesky-simulator[full]

# Setup Ollama for LLM (or configure OpenAI)
ollama pull llama3.1:8b
```

### Basic Usage

```bash
# Health check - verify all components
atc-llm health-check

# Run single scenario
atc-llm run-e2e --scat-path data/sample_scat.json --visualize

# Batch processing
atc-llm batch --scat-dir data/SCAT_extracted --output-dir results/

# View results
atc-llm metrics --results-dir results/latest/
```

## 📋 Features

### Core Capabilities
- ✅ SCAT dataset parsing with ASTERIX I062 support
- ✅ BlueSky integration with full command interface
- ✅ Multi-provider LLM support (OpenAI, Ollama, local models)
- ✅ Real-time conflict detection with 100 NM/±5000 ft surveillance
- ✅ Intelligent conflict resolution (waypoint, heading, altitude)
- ✅ Experience memory with similarity search
- ✅ Comprehensive safety metrics (TBAS, LAT, RAT)
- ✅ Dynamic intruder injection for scenario testing
- ✅ Real-time visualization and trajectory analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SCAT Data     │───▶│  Scenario       │───▶│   BlueSky       │
│   Adapter       │    │  Builder        │    │   Simulator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Experience    │◀───│   LLM Client    │◀───│   CDR Pipeline  │
│   Memory        │    │   (Multi-stage) │    │   (Orchestrator)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Visualization  │◀───│    Metrics      │◀───│   Validation    │
│   & Reports     │    │   Calculator    │    │   & Safety      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cdr --cov-report=html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Swedish Civil Aviation Authority for the SCAT dataset
- TU Delft for the BlueSky simulator
- OpenAP project for aircraft performance data

---

**⚠️ Disclaimer**: This is a research tool for academic purposes. Not certified for operational air traffic control use.
