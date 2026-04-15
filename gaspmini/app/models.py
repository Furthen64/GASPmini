# app/models.py
# Core data classes for GASPmini v1.
# Genome = inherited traits
# LifetimeState = per-epoch temporary state
# WorldState = grid contents at a moment in time

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class CellType(Enum):
    EMPTY = auto()
    WALL = auto()
    FOOD = auto()
    CREATURE = auto()


class Direction(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()


class ActionType(Enum):
    MOVE_FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    IDLE = auto()


# ── Sensor / Gene data classes ────────────────────────────────────────────────

@dataclass
class SensorField:
    """Snapshot of what a creature currently perceives."""
    current_cell: CellType
    front_cell: CellType
    left_cell: CellType
    right_cell: CellType
    back_cell: CellType
    last_action: Optional[ActionType]
    last_action_success: bool
    # 0=full, 1=medium, 2=low, 3=critical
    hunger_bucket: int


@dataclass
class GenePattern:
    """Pattern to match against a SensorField.  None means wildcard."""
    current_cell: Optional[CellType]
    front_cell: Optional[CellType]
    left_cell: Optional[CellType]
    right_cell: Optional[CellType]
    back_cell: Optional[CellType]
    last_action: Optional[ActionType]
    last_action_success: Optional[bool]
    hunger_bucket: Optional[int]


@dataclass
class Gene:
    gene_id: int
    pattern: GenePattern
    action: ActionType
    base_priority: float = 0.0


@dataclass
class Genome:
    genes: list[Gene] = field(default_factory=list)
    learning_rate: float = 0.25
    reward_decay: float = 0.7
    exploration_rate: float = 0.05
    history_length: int = 6


# ── History ───────────────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    sensor: SensorField
    gene_id: int
    action: ActionType
    reward: float
    action_success: bool
    tick_index: int


@dataclass
class RunHistorySample:
    age_ticks: int
    energy: float
    food_eaten: int
    failed_actions: int
    alive: bool


# ── LifetimeState ─────────────────────────────────────────────────────────────

@dataclass
class LifetimeState:
    """Resets every epoch.  Not inherited."""
    x: int
    y: int
    direction: Direction
    energy: float
    food_eaten: int = 0
    alive: bool = True
    age_ticks: int = 0
    failed_actions: int = 0
    stationary_ticks: int = 0
    last_action: Optional[ActionType] = None
    last_action_success: bool = False
    # gene_id -> cumulative learned adjustment
    learned_gene_adjustments: dict[int, float] = field(default_factory=dict)
    history: list[HistoryEntry] = field(default_factory=list)
    run_history: list[RunHistorySample] = field(default_factory=list)


# ── Creature ──────────────────────────────────────────────────────────────────

@dataclass
class Creature:
    creature_id: int
    genome: Genome
    lifetime: LifetimeState


# ── World ─────────────────────────────────────────────────────────────────────

@dataclass
class WorldState:
    width: int
    height: int
    walls: set[tuple[int, int]] = field(default_factory=set)
    food_positions: set[tuple[int, int]] = field(default_factory=set)
    creatures: list[Creature] = field(default_factory=list)
    tick_index: int = 0
    epoch_index: int = 0
    random_seed: int = 42


# ── Per-epoch result ──────────────────────────────────────────────────────────

@dataclass
class CreatureEpochResult:
    creature_id: int
    food_eaten: int
    age_ticks: int
    failed_actions: int
    fitness: float


# ── Action result ─────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    success: bool
    reward: float
    notes: str = ""
