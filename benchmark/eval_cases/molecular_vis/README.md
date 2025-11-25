# SciVisAgentBench: Molecular Visualization

This benchmark is designed to evaluate agents on molecular visualization tasks, specifically focusing on tools like VMD (Visual Molecular Dynamics).

## Overview

The benchmark assesses agent capabilities across different levels of complexity:
- **Basic Actions (Easy)**: Simple, atomic commands and operations (Currently implemented).
- **Workflows (Medium)**: Sequences of actions forming a coherent pipeline (Planned).
- **Scientific Tasks (Hard)**: Complex, goal-oriented scientific analysis and visualization (Planned).

## Setup

1. **Data Preparation**:
   - Download the PDB file `1CRN` from the [RCSB PDB](https://www.rcsb.org/structure/1CRN).
   - Place the downloaded file into the `data/` folder in this directory.

## Usage

You can run the evaluation cases defined in the YAML files (located in the `actions/` directory) using either:
- **Promptfoo**
- The **SciVisAgentBench Evaluation Framework**

Refer to the [main repository](https://github.com/KuangshiAi/SciVisAgentBench/tree/main) for detailed instructions on running the evaluation harness.

## Future Extensions

### SciVisAgentBench: MD Simulations
A placeholder for Molecular Dynamics (MD) simulation tasks is currently reserved. This will be expanded to include setting up, running, and analyzing simulations.
