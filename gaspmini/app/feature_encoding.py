from __future__ import annotations

import app.config as config
from app.models import CellType, GenePattern, SensorField

# Compact encodings
# cardinal food direction: 0=none, 1=north, 2=east, 3=south, 4=west
# hunger tier: 0=low, 1=med, 2=high

_FOOD_NORTH = 1
_FOOD_EAST = 2
_FOOD_SOUTH = 3
_FOOD_WEST = 4


# ── Shared helpers ───────────────────────────────────────────────────────────

def _is_blocked(cell: CellType) -> bool:
    return cell in {CellType.WALL, CellType.CREATURE}


def _hunger_tier(bucket: int) -> int:
    # Bucket 0 (full) -> low hunger, 1/2 -> medium, 3 -> high
    if bucket <= 0:
        return 0
    if bucket >= 3:
        return 2
    return 1


def _food_direction_code(sensor: SensorField) -> int:
    if sensor.north_cell == CellType.FOOD:
        return _FOOD_NORTH
    if sensor.east_cell == CellType.FOOD:
        return _FOOD_EAST
    if sensor.south_cell == CellType.FOOD:
        return _FOOD_SOUTH
    if sensor.west_cell == CellType.FOOD:
        return _FOOD_WEST
    return 0


def _obstacle_mask(sensor: SensorField) -> int:
    # 4-bit: north=1, east=2, south=4, west=8
    mask = 0
    if _is_blocked(sensor.north_cell):
        mask |= 0b0001
    if _is_blocked(sensor.east_cell):
        mask |= 0b0010
    if _is_blocked(sensor.south_cell):
        mask |= 0b0100
    if _is_blocked(sensor.west_cell):
        mask |= 0b1000
    return mask


def _nearby_creature(sensor: SensorField) -> int:
    adjacent = (sensor.north_cell, sensor.east_cell, sensor.south_cell, sensor.west_cell)
    return 1 if any(cell == CellType.CREATURE for cell in adjacent) else 0


def encode_sensor_legacy(sensor: SensorField) -> tuple[object, ...]:
    return (
        sensor.current_cell,
        sensor.north_cell,
        sensor.east_cell,
        sensor.south_cell,
        sensor.west_cell,
        sensor.last_action,
        sensor.last_action_success,
        sensor.hunger_bucket,
    )


def encode_sensor_compact(sensor: SensorField) -> tuple[object, ...]:
    return (
        _food_direction_code(sensor),
        _obstacle_mask(sensor),
        _nearby_creature(sensor),
        _hunger_tier(sensor.hunger_bucket),
    )


def _pattern_food_direction_code(pattern: GenePattern) -> int | None:
    direction_fields = [
        pattern.north_cell,
        pattern.east_cell,
        pattern.south_cell,
        pattern.west_cell,
    ]
    food_directions = [i for i, cell in enumerate(direction_fields, start=1) if cell == CellType.FOOD]

    if len(food_directions) == 1:
        return food_directions[0]
    if len(food_directions) > 1:
        return None

    # No explicit food directions. If all 4 directions are explicitly non-food, encode "none".
    if all(cell is not None and cell != CellType.FOOD for cell in direction_fields):
        return 0
    return None


def _pattern_obstacle_mask(pattern: GenePattern) -> int | None:
    direction_fields = [pattern.north_cell, pattern.east_cell, pattern.south_cell, pattern.west_cell]
    if any(cell is None for cell in direction_fields):
        return None

    mask = 0
    if _is_blocked(direction_fields[0]):
        mask |= 0b0001
    if _is_blocked(direction_fields[1]):
        mask |= 0b0010
    if _is_blocked(direction_fields[2]):
        mask |= 0b0100
    if _is_blocked(direction_fields[3]):
        mask |= 0b1000
    return mask


def _pattern_nearby_creature(pattern: GenePattern) -> int | None:
    direction_fields = [pattern.north_cell, pattern.east_cell, pattern.south_cell, pattern.west_cell]
    if any(cell == CellType.CREATURE for cell in direction_fields):
        return 1
    if all(cell is not None and cell != CellType.CREATURE for cell in direction_fields):
        return 0
    return None


def _pattern_hunger_tier(pattern: GenePattern) -> int | None:
    if pattern.hunger_bucket is None:
        return None
    return _hunger_tier(pattern.hunger_bucket)


def encode_pattern_legacy(pattern: GenePattern) -> tuple[object, ...]:
    return (
        pattern.current_cell,
        pattern.north_cell,
        pattern.east_cell,
        pattern.south_cell,
        pattern.west_cell,
        pattern.last_action,
        pattern.last_action_success,
        pattern.hunger_bucket,
    )


def encode_pattern_compact(pattern: GenePattern) -> tuple[object, ...]:
    return (
        _pattern_food_direction_code(pattern),
        _pattern_obstacle_mask(pattern),
        _pattern_nearby_creature(pattern),
        _pattern_hunger_tier(pattern),
    )


def encode_sensor_for_matching(sensor: SensorField) -> tuple[object, ...]:
    if config.SENSOR_ENCODER_MODE == 'compact':
        return encode_sensor_compact(sensor)
    return encode_sensor_legacy(sensor)


def encode_pattern_for_matching(pattern: GenePattern) -> tuple[object, ...]:
    if config.SENSOR_ENCODER_MODE == 'compact':
        return encode_pattern_compact(pattern)
    return encode_pattern_legacy(pattern)


def encode_sensor_for_learning(sensor: SensorField) -> tuple[int, ...]:
    # Reuse matching encoder so telemetry and learning traces line up.
    encoded = encode_sensor_for_matching(sensor)
    out: list[int] = []
    for item in encoded:
        if item is None:
            out.append(0)
        elif isinstance(item, bool):
            out.append(1 if item else 0)
        elif isinstance(item, int):
            out.append(item)
        else:
            out.append(item.value)
    return tuple(out)
