# app/sensors.py
# Build the SensorField for a creature given the current world state.
# Only local (4-directional) sensing for v1.

from __future__ import annotations

import app.config as config
from app.models import CellType, Direction, SensorField, Creature, WorldState
from app.world import get_cell_type


# ── Direction helpers ─────────────────────────────────────────────────────────

# Clockwise ordering: N, E, S, W
_DIRECTIONS_CW = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

# Delta (dx, dy) for each direction.  y increases downward.
_DIRECTION_DELTA: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: (0, -1),
    Direction.EAST:  (1,  0),
    Direction.SOUTH: (0,  1),
    Direction.WEST:  (-1, 0),
}


def _turn_left(direction: Direction) -> Direction:
    idx = _DIRECTIONS_CW.index(direction)
    return _DIRECTIONS_CW[(idx - 1) % 4]


def _turn_right(direction: Direction) -> Direction:
    idx = _DIRECTIONS_CW.index(direction)
    return _DIRECTIONS_CW[(idx + 1) % 4]


def _turn_back(direction: Direction) -> Direction:
    idx = _DIRECTIONS_CW.index(direction)
    return _DIRECTIONS_CW[(idx + 2) % 4]


def _cell_in_direction(
    x: int, y: int, direction: Direction, world: WorldState
) -> CellType:
    dx, dy = _DIRECTION_DELTA[direction]
    return get_cell_type(world, x + dx, y + dy)


# ── Hunger buckets ────────────────────────────────────────────────────────────

def _hunger_bucket(energy: float) -> int:
    """Map energy level to a hunger bucket 0–3."""
    # Bucket boundaries relative to INITIAL_ENERGY
    if energy >= config.INITIAL_ENERGY * 0.75:
        return 0  # full
    elif energy >= config.INITIAL_ENERGY * 0.40:
        return 1  # medium
    elif energy >= config.INITIAL_ENERGY * 0.15:
        return 2  # low
    else:
        return 3  # critical


# ── Public API ────────────────────────────────────────────────────────────────

def build_sensor_data(creature: Creature, world: WorldState) -> SensorField:
    """Return a SensorField snapshot for the given creature."""
    lt = creature.lifetime
    x, y = lt.x, lt.y
    facing = lt.direction

    left_dir  = _turn_left(facing)
    right_dir = _turn_right(facing)
    back_dir  = _turn_back(facing)

    return SensorField(
        front_cell=_cell_in_direction(x, y, facing,    world),
        left_cell= _cell_in_direction(x, y, left_dir,  world),
        right_cell=_cell_in_direction(x, y, right_dir, world),
        back_cell= _cell_in_direction(x, y, back_dir,  world),
        last_action=lt.last_action,
        last_action_success=lt.last_action_success,
        hunger_bucket=_hunger_bucket(lt.energy),
    )
