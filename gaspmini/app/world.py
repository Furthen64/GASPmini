# app/world.py
# World generation and grid helpers for GASPmini v1.
# Biome is regenerated fresh each epoch.

from __future__ import annotations

import random
from typing import Optional

import app.config as config
from app.custom_maps import get_custom_map
from app.models import (
    CellType, ActionType,
    Gene, GenePattern, Genome, LifetimeState, Creature, WorldState, RunHistorySample, HistoryBuffer,
)


# ── RNG helper ────────────────────────────────────────────────────────────────

def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


# ── Grid position helpers ─────────────────────────────────────────────────────

def is_inside(world: WorldState, x: int, y: int) -> bool:
    return 0 <= x < world.width and 0 <= y < world.height


def is_walkable(world: WorldState, x: int, y: int) -> bool:
    """True if the cell is inside the grid and not a wall."""
    return is_inside(world, x, y) and (x, y) not in world.walls


def get_cell_type(world: WorldState, x: int, y: int) -> CellType:
    """Return the CellType at (x, y).  Creatures take priority over food."""
    if not is_inside(world, x, y):
        return CellType.WALL
    if (x, y) in world.walls:
        return CellType.WALL
    creature_positions = {(c.lifetime.x, c.lifetime.y) for c in world.creatures if c.lifetime.alive}
    if (x, y) in creature_positions:
        return CellType.CREATURE
    if (x, y) in world.food_positions:
        return CellType.FOOD
    return CellType.EMPTY


def creature_at(world: WorldState, x: int, y: int) -> Optional[Creature]:
    """Return the living creature at (x, y) or None."""
    for c in world.creatures:
        if c.lifetime.alive and c.lifetime.x == x and c.lifetime.y == y:
            return c
    return None


# ── ASCII debug render ────────────────────────────────────────────────────────

CELL_SYMBOLS = {
    CellType.EMPTY: '.',
    CellType.WALL: '#',
    CellType.FOOD: 'F',
    CellType.CREATURE: 'C',
}


def render_ascii(world: WorldState) -> str:
    """Return a multi-line ASCII string of the current world."""
    lines: list[str] = []
    for y in range(world.height):
        row = ''
        for x in range(world.width):
            ct = get_cell_type(world, x, y)
            row += CELL_SYMBOLS[ct]
        lines.append(row)
    return '\n'.join(lines)


# ── Biome generation ──────────────────────────────────────────────────────────

def _place_border_walls(world: WorldState) -> None:
    """Fill the outermost ring with walls."""
    for x in range(world.width):
        world.walls.add((x, 0))
        world.walls.add((x, world.height - 1))
    for y in range(1, world.height - 1):
        world.walls.add((0, y))
        world.walls.add((world.width - 1, y))


def _place_interior_walls(world: WorldState, count: int, rng: random.Random) -> None:
    """Place random interior walls (not on border)."""
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 10:
        x = rng.randint(1, world.width - 2)
        y = rng.randint(1, world.height - 2)
        if (x, y) not in world.walls:
            world.walls.add((x, y))
            placed += 1
        attempts += 1


def _apply_custom_map(world: WorldState, map_id: str) -> None:
    definition = get_custom_map(map_id)
    if definition is None:
        raise ValueError(f"Unknown custom map: {map_id}")

    world.walls = set(definition.walls)
    world.food_positions = set(definition.food_positions)


def spawn_food(world: WorldState, food_count: int, rng: random.Random) -> None:
    """Place food cells on non-wall, non-occupied cells."""
    placed = 0
    attempts = 0
    occupied = world.walls.copy()
    while placed < food_count and attempts < food_count * 20:
        x = rng.randint(1, world.width - 2)
        y = rng.randint(1, world.height - 2)
        if (x, y) not in occupied:
            world.food_positions.add((x, y))
            occupied.add((x, y))
            placed += 1
        attempts += 1


def _make_random_gene_pattern(rng: random.Random) -> GenePattern:
    """Create a gene pattern with random fields or wildcards."""
    cell_types = list(CellType)
    actions = list(ActionType)

    def rand_cell() -> Optional[CellType]:
        return rng.choice(cell_types) if rng.random() < 0.6 else None

    def rand_action() -> Optional[ActionType]:
        return rng.choice(actions) if rng.random() < 0.4 else None

    def rand_bool() -> Optional[bool]:
        return rng.choice([True, False]) if rng.random() < 0.4 else None

    def rand_hunger() -> Optional[int]:
        return rng.randint(0, 3) if rng.random() < 0.4 else None

    return GenePattern(
        current_cell=rand_cell(),
        north_cell=rand_cell(),
        east_cell=rand_cell(),
        south_cell=rand_cell(),
        west_cell=rand_cell(),
        last_action=rand_action(),
        last_action_success=rand_bool(),
        hunger_bucket=rand_hunger(),
    )


def make_random_genome(rng: random.Random, gene_count: int | None = None) -> Genome:
    """Create a genome with random genes and slightly varied parameters."""
    if gene_count is None:
        gene_count = config.INITIAL_GENE_COUNT

    genes = [
        Gene(
            gene_id=i,
            pattern=_make_random_gene_pattern(rng),
            action=rng.choice(list(ActionType)),
            base_priority=rng.uniform(-0.5, 0.5),
        )
        for i in range(gene_count)
    ]
    return Genome(
        genes=genes,
        learning_rate=max(0.01, config.DEFAULT_LEARNING_RATE + rng.uniform(-0.05, 0.05)),
        reward_decay=max(0.1, min(0.99, config.DEFAULT_REWARD_DECAY + rng.uniform(-0.05, 0.05))),
        exploration_rate=max(0.0, config.DEFAULT_EXPLORATION_RATE + rng.uniform(-0.02, 0.02)),
        history_length=config.HISTORY_LENGTH,
    )


def spawn_creatures(
    world: WorldState,
    genomes: list[Genome],
    rng: random.Random,
) -> None:
    """Spawn creatures with the given genomes at random free positions."""
    occupied = world.walls | world.food_positions
    positions: list[tuple[int, int]] = []
    attempts = 0
    while len(positions) < len(genomes) and attempts < len(genomes) * 100:
        x = rng.randint(1, world.width - 2)
        y = rng.randint(1, world.height - 2)
        pos = (x, y)
        if pos not in occupied and pos not in set(positions):
            positions.append(pos)
        attempts += 1

    for idx, genome in enumerate(genomes):
        if idx >= len(positions):
            break
        x, y = positions[idx]
        lifetime = LifetimeState(
            x=x,
            y=y,
            energy=float(config.INITIAL_ENERGY),
            history_buffer=HistoryBuffer(config.SENSOR_HISTORY_CONTEXT_LENGTH),
        )
        lifetime.run_history.append(
            RunHistorySample(
                age_ticks=0,
                energy=lifetime.energy,
                food_eaten=0,
                failed_actions=0,
                alive=True,
            )
        )
        world.creatures.append(Creature(
            creature_id=idx,
            genome=genome,
            lifetime=lifetime,
        ))


def generate_world(
    epoch_index: int = 0,
    seed: int | None = None,
    genomes: Optional[list[Genome]] = None,
    population_size: int | None = None,
    food_count: int | None = None,
    interior_wall_count: int | None = None,
    width: int | None = None,
    height: int | None = None,
    custom_map_id: str | None = None,
) -> WorldState:
    """Create a fresh world for the given epoch."""
    custom_map = get_custom_map(custom_map_id)
    if seed is None:
        seed = config.DEFAULT_SEED
    if population_size is None:
        population_size = config.POPULATION_SIZE
    if food_count is None:
        food_count = config.FOOD_COUNT
    if interior_wall_count is None:
        interior_wall_count = config.INTERIOR_WALL_COUNT
    if custom_map is not None:
        width = custom_map.width
        height = custom_map.height
    else:
        if width is None:
            width = config.GRID_WIDTH
        if height is None:
            height = config.GRID_HEIGHT

    # Use epoch-derived seed so each epoch has a different but reproducible layout.
    rng = make_rng(seed + epoch_index * 1000)

    world = WorldState(
        width=width,
        height=height,
        random_seed=seed,
        epoch_index=epoch_index,
    )

    if custom_map is not None:
        _apply_custom_map(world, custom_map_id)
    else:
        _place_border_walls(world)
        _place_interior_walls(world, interior_wall_count, rng)
        spawn_food(world, food_count, rng)

    if genomes is None:
        genomes = [make_random_genome(rng) for _ in range(population_size)]

    spawn_creatures(world, genomes, rng)
    return world
