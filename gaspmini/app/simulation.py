# app/simulation.py
# Per-tick simulation step: sense → choose gene → execute action → learn.

from __future__ import annotations

import app.config as config
from app.models import (
    ActionType, CellType, Creature, WorldState, ActionResult, HistoryEntry, RunHistorySample,
)
from app.sensors import build_sensor_data
from app.gene_logic import choose_gene, score_gene
from app.learning import apply_reward_to_history, record_history
from app.logging_utils import debug_log, log
from app.world import get_cell_type, creature_at, is_walkable
from app.sensors import _DIRECTION_DELTA


# ── Direction helpers (re-used from sensors) ──────────────────────────────────

def _front_position(creature: Creature) -> tuple[int, int]:
    lt = creature.lifetime
    dx, dy = _DIRECTION_DELTA[lt.direction]
    return lt.x + dx, lt.y + dy


def _consume_food_at(creature: Creature, world: WorldState, x: int, y: int) -> bool:
    if (x, y) not in world.food_positions:
        return False
    world.food_positions.discard((x, y))
    creature.lifetime.food_eaten += 1
    creature.lifetime.energy += config.ENERGY_GAIN_FROM_FOOD
    return True


def _record_run_history_sample(creature: Creature) -> None:
    lt = creature.lifetime
    lt.run_history.append(
        RunHistorySample(
            age_ticks=lt.age_ticks,
            energy=lt.energy,
            food_eaten=lt.food_eaten,
            failed_actions=lt.failed_actions,
            alive=lt.alive,
        )
    )


# ── Action execution ──────────────────────────────────────────────────────────

def execute_action(
    creature: Creature,
    action: ActionType,
    world: WorldState,
) -> ActionResult:
    """Apply `action` to `creature` and update world state. Return result."""
    lt = creature.lifetime

    if action == ActionType.TURN_LEFT:
        from app.sensors import _turn_left
        lt.direction = _turn_left(lt.direction)
        return ActionResult(success=True, reward=0.0, notes="turned left")

    elif action == ActionType.TURN_RIGHT:
        from app.sensors import _turn_right
        lt.direction = _turn_right(lt.direction)
        return ActionResult(success=True, reward=0.0, notes="turned right")

    elif action == ActionType.MOVE_FORWARD:
        fx, fy = _front_position(creature)
        if not is_walkable(world, fx, fy):
            lt.failed_actions += 1
            return ActionResult(success=False, reward=config.REWARD_FAILED_MOVE, notes="wall block")
        if creature_at(world, fx, fy) is not None:
            lt.failed_actions += 1
            return ActionResult(success=False, reward=config.REWARD_FAILED_MOVE, notes="creature block")
        lt.x, lt.y = fx, fy
        return ActionResult(success=True, reward=0.0, notes=f"moved to ({fx},{fy})")

    elif action == ActionType.EAT:
        if _consume_food_at(creature, world, lt.x, lt.y):
            return ActionResult(success=True, reward=config.REWARD_EAT_FOOD, notes="ate food underfoot")
        lt.failed_actions += 1
        return ActionResult(success=False, reward=config.REWARD_FAILED_EAT, notes="no food underfoot")

    elif action == ActionType.IDLE:
        return ActionResult(success=True, reward=config.REWARD_IDLE, notes="idle")

    return ActionResult(success=False, reward=0.0, notes="unknown action")


# ── Single creature tick ──────────────────────────────────────────────────────

def tick_creature(creature: Creature, world: WorldState) -> None:
    """Advance one creature by one tick."""
    lt = creature.lifetime

    if not lt.alive:
        return

    # 1. Build sensor snapshot
    sensor = build_sensor_data(creature, world)
    if config.DEBUG_SENSORS:
        debug_log(f"Creature {creature.creature_id} sensor: {sensor}")

    # 2. Choose gene
    gene = choose_gene(creature, sensor)
    if config.DEBUG_GENE_SCORING:
        debug_log(
            f"Creature {creature.creature_id} chose gene {gene.gene_id} "
            f"→ {gene.action.name}"
        )

    # 3. Execute action
    action = gene.action
    result = execute_action(creature, action, world)

    if config.DEBUG_ACTIONS:
        debug_log(
            f"Creature {creature.creature_id} action={action.name}  "
            f"success={result.success}  reward={result.reward:.2f}  "
            f"notes={result.notes}"
        )

    # 4. Energy drain
    lt.energy -= config.ENERGY_LOSS_PER_TICK
    lt.age_ticks += 1

    # 5. Record history
    entry = HistoryEntry(
        sensor=sensor,
        gene_id=gene.gene_id,
        action=action,
        reward=result.reward,
        action_success=result.success,
        tick_index=world.tick_index,
    )
    record_history(creature, entry)

    # 6. Apply reward to history
    if result.reward != 0.0:
        apply_reward_to_history(creature, result.reward)

    # 7. Update last action fields
    lt.last_action = action
    lt.last_action_success = result.success

    # 8. Check death by starvation
    if lt.energy <= 0:
        lt.alive = False
        apply_reward_to_history(creature, config.REWARD_DEATH)
        log(
            f"Creature {creature.creature_id} died at tick {world.tick_index} "
            f"(food eaten: {lt.food_eaten})"
        )

    _record_run_history_sample(creature)


# ── World tick ────────────────────────────────────────────────────────────────

def tick_world(world: WorldState) -> None:
    """Advance all living creatures by one tick."""
    for creature in world.creatures:
        if creature.lifetime.alive:
            tick_creature(creature, world)
    world.tick_index += 1
