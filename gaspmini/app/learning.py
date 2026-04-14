# app/learning.py
# Within-epoch temporal credit assignment for GASPmini v1.
# Rewards/penalties propagate backward through recent history with decay.

from __future__ import annotations

from app.config import DEBUG_REWARDS, HISTORY_LENGTH
from app.models import Creature, HistoryEntry
from app.logging_utils import debug_log


def record_history(
    creature: Creature,
    entry: HistoryEntry,
) -> None:
    """Append a history entry and trim to max length."""
    lt = creature.lifetime
    max_len = creature.genome.history_length or HISTORY_LENGTH
    lt.history.append(entry)
    if len(lt.history) > max_len:
        lt.history = lt.history[-max_len:]


def apply_reward_to_history(
    creature: Creature,
    reward: float,
    max_steps_back: int | None = None,
) -> None:
    """
    Propagate `reward` backward through the creature's recent history.

    For step i steps back from the most recent entry:
        adjustment += reward * (reward_decay ** i) * learning_rate

    This updates `lifetime.learned_gene_adjustments` in place.
    """
    genome = creature.genome
    lt = creature.lifetime

    lr = genome.learning_rate
    decay = genome.reward_decay
    history = lt.history

    if not history:
        return

    if max_steps_back is None:
        max_steps_back = len(history)
    else:
        max_steps_back = min(max_steps_back, len(history))

    recent = history[-max_steps_back:]

    for steps_back, entry in enumerate(reversed(recent)):
        adjustment = reward * (decay ** steps_back) * lr
        gene_id = entry.gene_id
        current = lt.learned_gene_adjustments.get(gene_id, 0.0)
        lt.learned_gene_adjustments[gene_id] = current + adjustment

        if DEBUG_REWARDS:
            debug_log(
                f"  Learning: gene={gene_id}  steps_back={steps_back}  "
                f"reward={reward:.2f}  decay^n={decay**steps_back:.3f}  "
                f"adj+={adjustment:.4f}  total={lt.learned_gene_adjustments[gene_id]:.4f}"
            )
