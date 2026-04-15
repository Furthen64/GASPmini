# app/models.py
# Core data classes for GASPmini v1.
# Genome = inherited traits
# LifetimeState = per-epoch temporary state
# WorldState = grid contents at a moment in time

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from collections import deque
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


# ── Sensor-context history ───────────────────────────────────────────────────

_ACTION_CODE: dict[Optional[ActionType], int] = {
    None: 0,
    ActionType.MOVE_FORWARD: 1,
    ActionType.TURN_LEFT: 2,
    ActionType.TURN_RIGHT: 3,
    ActionType.IDLE: 4,
}


def action_to_code(action: Optional[ActionType]) -> int:
    """Encode an action into a compact integer feature."""
    return _ACTION_CODE[action]


@dataclass(frozen=True)
class HistoryTuple:
    food_ahead: int
    food_left: int
    food_right: int
    front_blocked: int
    hunger_bucket: int
    previous_action_code: int
    previous_success_flag: int


@dataclass
class HistoryBuffer:
    """Fixed-size FIFO for compact per-tick context tuples."""
    length: int = 3
    _entries: deque[HistoryTuple] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bounded_length = max(2, min(4, self.length))
        self.length = bounded_length
        self._entries = deque(maxlen=bounded_length)

    @staticmethod
    def _is_food(cell: CellType) -> int:
        return 1 if cell == CellType.FOOD else 0

    @staticmethod
    def _is_blocked(cell: CellType) -> int:
        return 1 if cell in {CellType.WALL, CellType.CREATURE} else 0

    def push(self, sensor: SensorField, action: ActionType, success: bool) -> None:
        """Append a compact context tuple for the tick."""
        self._entries.append(HistoryTuple(
            food_ahead=self._is_food(sensor.front_cell),
            food_left=self._is_food(sensor.left_cell),
            food_right=self._is_food(sensor.right_cell),
            front_blocked=self._is_blocked(sensor.front_cell),
            hunger_bucket=sensor.hunger_bucket,
            previous_action_code=action_to_code(action),
            previous_success_flag=1 if success else 0,
        ))

    def recent_first(self) -> list[HistoryTuple]:
        return list(reversed(self._entries))

    def flattened(self) -> list[int]:
        """
        Flatten tuples newest-first to a fixed-size vector.
        Missing (older) slots are zero-padded.
        """
        out: list[int] = []
        recent = self.recent_first()
        for idx in range(self.length):
            if idx < len(recent):
                item = recent[idx]
                out.extend(dataclasses.astuple(item))
            else:
                out.extend((0, 0, 0, 0, 0, 0, 0))
        return out


# ── History ───────────────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    sensor: SensorField
    gene_id: int
    action: ActionType
    reward: float
    action_success: bool
    tick_index: int


@dataclass(frozen=True)
class TransitionTuple:
    """
    Compact transition used for reward-event trajectory credit assignment.
    """
    state_features: tuple[int, ...]
    action: ActionType
    reward: float
    gene_id: int
    tick_index: int


@dataclass
class TransitionRingBuffer:
    """Fixed-size FIFO ring buffer for transition tuples."""
    length: int = 6
    _entries: deque[TransitionTuple] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bounded_length = max(1, self.length)
        self.length = bounded_length
        self._entries = deque(maxlen=bounded_length)

    def push(self, transition: TransitionTuple) -> None:
        self._entries.append(transition)

    def recent_first(self, max_steps_back: int | None = None) -> list[TransitionTuple]:
        recent = list(reversed(self._entries))
        if max_steps_back is None:
            return recent
        return recent[:max(0, max_steps_back)]


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
    history_buffer: HistoryBuffer = field(default_factory=HistoryBuffer)
    transition_buffer: TransitionRingBuffer = field(default_factory=TransitionRingBuffer)


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
