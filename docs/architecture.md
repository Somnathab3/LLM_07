# System Architecture

## Overview

LLM_ATC7 implements a multi-stage conflict detection and resolution system:

1. **Data Layer**: SCAT dataset parsing and flight plan extraction
2. **Simulation Layer**: BlueSky integration for realistic flight dynamics
3. **AI Layer**: LLM-based conflict detection and resolution
4. **Analysis Layer**: Safety metrics and performance evaluation

## Component Interaction

```
SCAT Data → Scenario Builder → BlueSky Sim → CDR Pipeline → Metrics
                                 ↑              ↓
                           LLM Client ← Experience Memory
```

## Key Components

### SCAT Adapter
- Parses ASTERIX I062 surveillance data
- Extracts flight plans and clearances
- Finds neighboring aircraft within detection range

### BlueSky Client
- Interfaces with BlueSky ATC simulator
- Sends commands (HDG, ALT, SPD)
- Retrieves aircraft states

### LLM Client
- Multi-provider support (OpenAI, Ollama)
- Structured prompts for conflict detection and resolution
- JSON response parsing and validation

### CDR Pipeline
- Main orchestration component
- Conflict detection and resolution loop
- Safety validation and metrics collection

## Safety Considerations

- Separation standards: 5 NM horizontal, 1000 ft vertical
- Lookahead time: 10 minutes for conflict detection
- Resolution constraints: Max 45° heading, 2000 ft altitude changes
- Fallback mechanisms: Deterministic resolution if LLM fails

## Data Flow

1. **Input**: SCAT data files with surveillance tracks
2. **Processing**: Conflict detection using geometric algorithms + LLM
3. **Resolution**: LLM-generated maneuvers with safety validation
4. **Output**: Trajectory data, conflict logs, safety metrics

## Metrics

### Safety Metrics
- TBAS (Time-Based Avoidance System) score
- LAT (Look-Ahead Time) score  
- RAT (Resolution Action Time) score
- Separation violations count
- Minimum separation achieved

### Performance Metrics
- Resolution success rate
- LLM response time
- Additional distance/time due to maneuvers
- Memory system hit rate
