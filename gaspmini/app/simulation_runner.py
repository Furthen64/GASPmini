# app/simulation_runner.py
# High-level simulation controller used by the GUI.
# Owns world state, drives tick/epoch transitions.

from __future__ import annotations

import random

from app.config import TICKS_PER_EPOCH, POPULATION_SIZE, DEFAULT_SEED
from app.models import WorldState, Creature
from app.world import generate_world, make_rng
from app.simulation import tick_world
from app.evolution import collect_epoch_results, evolve_next_generation
from app.logging_utils import log


class SimulationRunner:
    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self.world: WorldState | None = None
        self._rng: random.Random = make_rng(seed)
        self.epoch_history: list[dict] = []  # brief per-epoch summary

    # ── Initialisation ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Start fresh from epoch 0 with a new (or the same) seed."""
        if seed is not None:
            self.seed = seed
        self._rng = make_rng(self.seed)
        self.epoch_history = []
        self.world = generate_world(
            epoch_index=0,
            seed=self.seed,
        )
        log(f"Simulation reset. Seed={self.seed}  Creatures={len(self.world.creatures)}")

    # ── Tick ───────────────────────────────────────────────────────────────────

    def step_tick(self) -> None:
        """Advance the simulation by exactly one tick."""
        if self.world is None:
            self.reset()
        assert self.world is not None

        if self.world.tick_index >= TICKS_PER_EPOCH:
            self.step_epoch()
            return

        tick_world(self.world)

    # ── Epoch ──────────────────────────────────────────────────────────────────

    def step_epoch(self) -> None:
        """Finish the current epoch and start a new one."""
        if self.world is None:
            self.reset()
        assert self.world is not None

        # Drain remaining ticks if needed
        while self.world.tick_index < TICKS_PER_EPOCH:
            tick_world(self.world)

        # Collect results and log
        results = collect_epoch_results(self.world)
        self.epoch_history.append({
            'epoch': self.world.epoch_index,
            'results': results,
        })
        alive = sum(1 for c in self.world.creatures if c.lifetime.alive)
        top = results[0].fitness if results else 0.0
        log(
            f"Epoch {self.world.epoch_index} ended. "
            f"Alive: {alive}/{len(self.world.creatures)}  "
            f"Top fitness: {top:.2f}"
        )

        # Evolve
        next_genomes = evolve_next_generation(self.world, results, self._rng)
        next_epoch = self.world.epoch_index + 1

        self.world = generate_world(
            epoch_index=next_epoch,
            seed=self.seed,
            genomes=next_genomes,
        )
        log(f"Epoch {next_epoch} started. Population: {len(self.world.creatures)}")

    # ── Queries ────────────────────────────────────────────────────────────────

    def is_epoch_over(self) -> bool:
        if self.world is None:
            return False
        return self.world.tick_index >= TICKS_PER_EPOCH

    def alive_count(self) -> int:
        if self.world is None:
            return 0
        return sum(1 for c in self.world.creatures if c.lifetime.alive)

    def find_creature_at(self, x: int, y: int) -> Creature | None:
        if self.world is None:
            return None
        for c in self.world.creatures:
            if c.lifetime.alive and c.lifetime.x == x and c.lifetime.y == y:
                return c
        return None
