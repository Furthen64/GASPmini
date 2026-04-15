from __future__ import annotations

from dataclasses import dataclass

from app.custom_maps import hidden_food, simple_1


@dataclass(frozen=True)
class CustomMapDefinition:
    map_id: str
    label: str
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    food_positions: frozenset[tuple[int, int]]


def _build_definition(map_id: str, label: str, layout_rows: tuple[str, ...]) -> CustomMapDefinition:
    if not layout_rows:
        raise ValueError(f"Custom map '{map_id}' must define at least one row.")

    width = len(layout_rows[0])
    if width == 0:
        raise ValueError(f"Custom map '{map_id}' cannot have empty rows.")

    walls: set[tuple[int, int]] = set()
    food_positions: set[tuple[int, int]] = set()

    for y, row in enumerate(layout_rows):
        if len(row) != width:
            raise ValueError(f"Custom map '{map_id}' has inconsistent row widths.")
        for x, cell in enumerate(row):
            if cell == 'W':
                walls.add((x, y))
            elif cell == 'F':
                food_positions.add((x, y))
            elif cell != '#':
                raise ValueError(
                    f"Custom map '{map_id}' contains unsupported tile '{cell}' at {(x, y)}."
                )

    return CustomMapDefinition(
        map_id=map_id,
        label=label,
        width=width,
        height=len(layout_rows),
        walls=frozenset(walls),
        food_positions=frozenset(food_positions),
    )


_CUSTOM_MAPS: tuple[CustomMapDefinition, ...] = (
    _build_definition(
        hidden_food.MAP_ID,
        hidden_food.LABEL,
        hidden_food.LAYOUT_ROWS,
    ),
    _build_definition(
        simple_1.MAP_ID,
        simple_1.LABEL,
        simple_1.LAYOUT_ROWS,
    ),
)

CUSTOM_MAP_DEFINITIONS: dict[str, CustomMapDefinition] = {
    definition.map_id: definition
    for definition in _CUSTOM_MAPS
}


def get_custom_map(map_id: str | None) -> CustomMapDefinition | None:
    if not map_id:
        return None
    return CUSTOM_MAP_DEFINITIONS.get(map_id)


def get_custom_map_items() -> list[tuple[str, str]]:
    return [
        (definition.map_id, definition.label)
        for definition in _CUSTOM_MAPS
    ]