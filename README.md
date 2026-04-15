# GASPmini
Simpler genetic algorithm playground for learning the basics

## About

GASPmini is a simple, debuggable Python + Qt simulation that demonstrates:

- A 2D grid world with walls, food, and creatures
- Local sensing for the current tile plus adjacent tiles, with eating allowed only on the current tile
- Creatures with a **genome** (inherited traits) and **LifetimeState** (per-epoch state)
- Within-epoch learning via temporal credit assignment
- Between-epoch evolution (selection, crossover, mutation)
- A PySide6 Qt GUI for visualisation and step-by-step debugging

## Project structure

```
gaspmini/
├── app/
│   ├── config.py           # All tunable constants
│   ├── models.py           # Core dataclasses (Genome, Creature, WorldState, …)
│   ├── world.py            # World/biome generation, ASCII rendering
│   ├── sensors.py          # Local sensor snapshot for each creature
│   ├── gene_logic.py       # Wildcard gene matching and gene selection
│   ├── learning.py         # Within-epoch reward propagation
│   ├── simulation.py       # Per-tick action execution
│   ├── simulation_runner.py# High-level tick/epoch controller
│   ├── evolution.py        # Fitness, selection, crossover, mutation
│   ├── logging_utils.py    # Simple print/log helpers
│   └── qt_ui.py            # PySide6 GUI
└── tests/
    ├── test_gene_match.py
    ├── test_learning.py
    └── test_evolution.py
main.py                     # Entry point
```

## Requirements

- Python 3.11+
- PySide6 (`pip install PySide6`)

## Running

### GUI mode (default)
```bash
python main.py
```

### Console mode (no Qt needed)
```bash
python main.py --console
```

### Tests
```bash
cd gaspmini
python -m pytest tests/ -v
```

## GUI controls

| Button | Action |
|--------|--------|
| ▶ Start | Run simulation automatically |
| ⏸ Pause | Pause |
| Step Tick | Advance exactly one tick |
| Step Epoch | Run to end of epoch, then evolve |
| Reset | Restart with current seed |
| New Seed | Pick a random seed and restart |

Click any creature cell in the grid to inspect its sensor data, gene scores, learned adjustments, and history.

## Key design principles (v1)

- **Genome** = inherited stable traits (genes, learning parameters)
- **LifetimeState** = temporary per-epoch state (position, energy, history, learned adjustments)
- **WorldState** = current biome / grid contents
- Readability over performance
- Fully deterministic with a fixed seed

