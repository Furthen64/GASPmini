# app/gene_logic.py
# Wildcard gene-matching and gene selection for GASPmini v1.

from __future__ import annotations

import random

import app.config as config
from app.models import Gene, Genome, GenePattern, SensorField, Creature, HistoryTuple, action_to_code
from app.logging_utils import debug_log


# ── Field scoring ─────────────────────────────────────────────────────────────

def _score_field(pattern_value, sensor_value) -> float:
    if pattern_value is None:
        return config.GENE_WILDCARD_SCORE
    if pattern_value == sensor_value:
        return config.GENE_MATCH_SCORE
    return config.GENE_MISMATCH_SCORE


# ── Pattern matching ──────────────────────────────────────────────────────────

def score_gene_match(sensor: SensorField, gene_pattern: GenePattern) -> float:
    """Return a match score for a gene pattern against the current sensor data."""
    total = 0.0
    total += _score_field(gene_pattern.current_cell,       sensor.current_cell)
    total += _score_field(gene_pattern.front_cell,         sensor.front_cell)
    total += _score_field(gene_pattern.left_cell,          sensor.left_cell)
    total += _score_field(gene_pattern.right_cell,         sensor.right_cell)
    total += _score_field(gene_pattern.back_cell,          sensor.back_cell)
    total += _score_field(gene_pattern.last_action,        sensor.last_action)
    total += _score_field(gene_pattern.last_action_success, sensor.last_action_success)
    total += _score_field(gene_pattern.hunger_bucket,      sensor.hunger_bucket)
    return total


def _expected_food_flag(cell_pattern) -> int | None:
    if cell_pattern is None:
        return None
    return 1 if cell_pattern.name == "FOOD" else 0


def _expected_front_blocked_flag(cell_pattern) -> int | None:
    if cell_pattern is None:
        return None
    return 1 if cell_pattern.name in {"WALL", "CREATURE"} else 0


def _score_history_tuple(pattern: GenePattern, history: HistoryTuple) -> float:
    total = 0.0
    total += _score_field(_expected_food_flag(pattern.front_cell), history.food_ahead)
    total += _score_field(_expected_food_flag(pattern.left_cell), history.food_left)
    total += _score_field(_expected_food_flag(pattern.right_cell), history.food_right)
    total += _score_field(_expected_front_blocked_flag(pattern.front_cell), history.front_blocked)
    total += _score_field(pattern.hunger_bucket, history.hunger_bucket)
    expected_action_code = None if pattern.last_action is None else action_to_code(pattern.last_action)
    total += _score_field(expected_action_code, history.previous_action_code)
    expected_success = None if pattern.last_action_success is None else int(pattern.last_action_success)
    total += _score_field(expected_success, history.previous_success_flag)
    return total


def score_history_context(creature: Creature, gene_pattern: GenePattern) -> float:
    if not config.USE_SENSOR_HISTORY_CONTEXT:
        return 0.0

    history = creature.lifetime.history_buffer.recent_first()
    total = 0.0
    for idx, item in enumerate(history):
        weight = config.HISTORY_CONTEXT_DECAY ** idx
        total += _score_history_tuple(gene_pattern, item) * weight
    return total


# ── Gene selection ────────────────────────────────────────────────────────────

def score_gene(
    gene: Gene,
    sensor: SensorField,
    learned_adjustments: dict[int, float],
    creature: Creature,
) -> float:
    """Total score for a gene: match score + base priority + learned adjustment."""
    match_score = score_gene_match(sensor, gene.pattern)
    history_score = score_history_context(creature, gene.pattern)
    adjustment = learned_adjustments.get(gene.gene_id, 0.0)
    return match_score + history_score + gene.base_priority + adjustment


def choose_gene(creature: Creature, sensor: SensorField) -> Gene:
    """
    Choose the best gene for this tick.
    With probability exploration_rate, pick randomly from the top 3 genes.
    """
    genome: Genome = creature.genome
    learned = creature.lifetime.learned_gene_adjustments

    scored = [
        (score_gene(g, sensor, learned, creature), g)
        for g in genome.genes
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    if config.DEBUG_GENE_SCORING:
        for sc, g in scored[:5]:
            debug_log(
                f"  Gene {g.gene_id}: total={sc:.2f}  "
                f"match={score_gene_match(sensor, g.pattern):.2f}  "
                f"history={score_history_context(creature, g.pattern):.2f}  "
                f"base={g.base_priority:.2f}  "
                f"adj={learned.get(g.gene_id, 0.0):.3f}  "
                f"action={g.action.name}"
            )

    # Exploration: pick randomly from top-3 genes with small probability
    if random.random() < genome.exploration_rate and len(scored) >= 2:
        top_n = min(3, len(scored))
        return scored[random.randint(0, top_n - 1)][1]

    return scored[0][1]
