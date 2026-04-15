# app/sensors.py
# Build the SensorField for a creature given the current world state.
# Only local (4-directional) sensing for v1.

from __future__ import annotations

import app.config as config
from app.models import CellType, SensorField, Creature, WorldState
from app.world import get_cell_type, is_inside


def _cell_at_offset(x: int, y: int, dx: int, dy: int, world: WorldState) -> CellType:
    return get_cell_type(world, x + dx, y + dy)


def _current_cell_type(x: int, y: int, world: WorldState) -> CellType:
    if not is_inside(world, x, y):
        return CellType.WALL
    if (x, y) in world.walls:
        return CellType.WALL
    if (x, y) in world.food_positions:
        return CellType.FOOD
    return CellType.EMPTY


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

    return SensorField(
        current_cell=_current_cell_type(x, y, world),
        north_cell=_cell_at_offset(x, y, 0, -1, world),
        east_cell=_cell_at_offset(x, y, 1, 0, world),
        south_cell=_cell_at_offset(x, y, 0, 1, world),
        west_cell=_cell_at_offset(x, y, -1, 0, world),
        last_action=lt.last_action,
        last_action_success=lt.last_action_success,
        hunger_bucket=_hunger_bucket(lt.energy),
    )
