# app/learning.py
# Within-epoch temporal credit assignment for GASPmini v1.
# Rewards/penalties propagate backward through recent history with decay.

from __future__ import annotations

import app.config as config
from app.models import Creature, HistoryEntry, SensorField, TransitionTuple
from app.feature_encoding import encode_sensor_for_learning
from app.logging_utils import debug_log


def _sensor_to_state_features(sensor: SensorField) -> tuple[int, ...]:
    """Convert sensor fields into numeric features for reward telemetry."""
    return encode_sensor_for_learning(sensor)


def record_history(
    creature: Creature,
    entry: HistoryEntry,
) -> None:
    """Append history and transition ring-buffer entries; trim to max length."""
    lt = creature.lifetime
    max_len = creature.genome.history_length or config.HISTORY_LENGTH
    lt.history.append(entry)
    if len(lt.history) > max_len:
        lt.history = lt.history[-max_len:]

    # Keep transition ring buffer size aligned with learning horizon.
    if lt.transition_buffer.length != max_len:
        lt.transition_buffer.length = max_len
        lt.transition_buffer.__post_init__()
    lt.transition_buffer.push(
        TransitionTuple(
            state_features=_sensor_to_state_features(entry.sensor),
            action=entry.action,
            reward=entry.reward,
            gene_id=entry.gene_id,
            tick_index=entry.tick_index,
        )
    )


def apply_reward_to_history(
    creature: Creature,
    reward: float,
    max_steps_back: int | None = None,
) -> None:
    """
    Reward-event learning using n-step returns over the transition ring buffer.

    Given recent transitions (newest-first) t0, t1, ...
    running_return starts from the reward event and folds trajectory rewards:

        G_0 = r(t0) + reward_event
        G_k = r(tk) + gamma * G_{k-1}     for k > 0

    Then each responsible <state, action>/gene receives:

        adjustment += learning_rate * G_k
    """
    genome = creature.genome
    lt = creature.lifetime

    lr = genome.learning_rate
    gamma = genome.reward_decay
    transitions = lt.transition_buffer.recent_first(max_steps_back)

    if not transitions:
        return

    running_return = reward

    for steps_back, transition in enumerate(transitions):
        running_return = transition.reward + running_return
        adjustment = lr * running_return
        gene_id = transition.gene_id
        state_gene_key = (transition.state_features, gene_id)
        current = lt.learned_gene_adjustments.get(gene_id, 0.0)
        lt.learned_gene_adjustments[gene_id] = current + adjustment
        state_current = lt.learned_state_gene_adjustments.get(state_gene_key, 0.0)
        lt.learned_state_gene_adjustments[state_gene_key] = state_current + adjustment

        if config.DEBUG_REWARDS:
            debug_log(
                "  CreditTrace: "
                f"tick={transition.tick_index}  gene={gene_id}  action={transition.action.name}  "
                f"steps_back={steps_back}  state={transition.state_features}  "
                f"transition_r={transition.reward:.2f}  event_r={reward:.2f}  "
                f"return={running_return:.4f}  adj+={adjustment:.4f}  "
                f"gene_total={lt.learned_gene_adjustments[gene_id]:.4f}  "
                f"state_total={lt.learned_state_gene_adjustments[state_gene_key]:.4f}"
            )
        running_return *= gamma
