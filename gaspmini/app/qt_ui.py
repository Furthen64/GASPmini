# app/qt_ui.py
# Qt GUI for GASPmini v1.
# Uses PySide6 widgets with a custom painted grid widget.

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QColor, QPainter, QFont, QMouseEvent, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QGroupBox, QSplitter,
    QScrollArea,
)

from app.config import TICKS_PER_EPOCH
from app.models import CellType, Creature, WorldState
from app.simulation_runner import SimulationRunner
from app.sensors import build_sensor_data
from app.gene_logic import score_gene, score_gene_match
from app.evolution import compute_fitness
from app.logging_utils import set_log_callback


# ── Colours ────────────────────────────────────────────────────────────────────

CELL_COLORS = {
    CellType.EMPTY:    QColor(30,  30,  30),
    CellType.WALL:     QColor(100, 100, 100),
    CellType.FOOD:     QColor(60,  160, 60),
    CellType.CREATURE: QColor(60,  120, 220),
}
SELECTED_COLOR = QColor(255, 200, 0)
GRID_LINE_COLOR = QColor(50, 50, 50)
CELL_SIZE = 22   # pixels per cell


# ── Grid widget ────────────────────────────────────────────────────────────────

class GridWidget(QWidget):
    """Custom-painted grid widget."""

    def __init__(self, sim: SimulationRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sim = sim
        self._selected_creature_id: Optional[int] = None
        self.setMouseTracking(True)
        self._update_size()

    def _update_size(self) -> None:
        if self._sim.world is None:
            return
        w = self._sim.world.width * CELL_SIZE
        h = self._sim.world.height * CELL_SIZE
        self.setMinimumSize(w, h)
        self.setFixedSize(w, h)

    def set_selected(self, creature_id: Optional[int]) -> None:
        self._selected_creature_id = creature_id
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        world = self._sim.world
        if world is None:
            return

        from app.world import get_cell_type
        creature_positions: dict[tuple[int, int], Creature] = {
            (c.lifetime.x, c.lifetime.y): c
            for c in world.creatures if c.lifetime.alive
        }

        for y in range(world.height):
            for x in range(world.width):
                px = x * CELL_SIZE
                py = y * CELL_SIZE

                ct = get_cell_type(world, x, y)
                color = CELL_COLORS[ct]

                # Highlight selected creature
                pos = (x, y)
                if pos in creature_positions:
                    c = creature_positions[pos]
                    if c.creature_id == self._selected_creature_id:
                        color = SELECTED_COLOR

                painter.fillRect(px, py, CELL_SIZE, CELL_SIZE, color)

                # Grid lines
                painter.setPen(QPen(GRID_LINE_COLOR, 0))
                painter.drawRect(px, py, CELL_SIZE - 1, CELL_SIZE - 1)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._sim.world is None:
            return
        x = int(event.position().x() // CELL_SIZE)
        y = int(event.position().y() // CELL_SIZE)
        # Ignore clicks outside the valid grid area
        world = self._sim.world
        if x < 0 or x >= world.width or y < 0 or y >= world.height:
            return
        # Find the parent MainWindow and notify
        parent = self.parent()
        while parent and not isinstance(parent, MainWindow):
            parent = parent.parent()
        if isinstance(parent, MainWindow):
            parent.on_grid_click(x, y)


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GASPmini")

        self._sim = SimulationRunner()
        self._sim.reset()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._running = False
        self._timer_interval_ms = 100   # ms between ticks when running

        self._selected_creature: Optional[Creature] = None

        # Redirect log output to the log panel
        set_log_callback(self._append_log)

        self._build_ui()
        self._refresh()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # ── Left: grid ────────────────────────────────────────────────────────
        self._grid_widget = GridWidget(self._sim)

        scroll = QScrollArea()
        scroll.setWidget(self._grid_widget)
        scroll.setWidgetResizable(False)

        root_layout.addWidget(scroll, stretch=3)

        # ── Right: controls + info ─────────────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        root_layout.addWidget(right_panel, stretch=1)

        # Simulation status
        status_group = QGroupBox("Simulation")
        status_layout = QVBoxLayout(status_group)
        self._lbl_epoch = QLabel("Epoch: 0")
        self._lbl_tick  = QLabel("Tick: 0")
        self._lbl_alive = QLabel("Alive: 0")
        for lbl in (self._lbl_epoch, self._lbl_tick, self._lbl_alive):
            status_layout.addWidget(lbl)
        right_layout.addWidget(status_group)

        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self._btn_start     = QPushButton("▶ Start")
        self._btn_pause     = QPushButton("⏸ Pause")
        self._btn_step_tick = QPushButton("Step Tick")
        self._btn_step_epoch= QPushButton("Step Epoch")
        self._btn_reset     = QPushButton("Reset")

        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_step_tick.clicked.connect(self._on_step_tick)
        self._btn_step_epoch.clicked.connect(self._on_step_epoch)
        self._btn_reset.clicked.connect(self._on_reset)

        for btn in (self._btn_start, self._btn_pause, self._btn_step_tick,
                    self._btn_step_epoch, self._btn_reset):
            ctrl_layout.addWidget(btn)

        # Seed control
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self._seed_edit = QLineEdit(str(self._sim.seed))
        self._seed_edit.setFixedWidth(80)
        seed_layout.addWidget(self._seed_edit)
        btn_new_seed = QPushButton("New Seed")
        btn_new_seed.clicked.connect(self._on_new_seed)
        seed_layout.addWidget(btn_new_seed)
        ctrl_layout.addLayout(seed_layout)

        right_layout.addWidget(ctrl_group)

        # Selected creature info
        creature_group = QGroupBox("Selected Creature")
        creature_layout = QVBoxLayout(creature_group)
        self._creature_info = QTextEdit()
        self._creature_info.setReadOnly(True)
        self._creature_info.setFixedHeight(220)
        creature_layout.addWidget(self._creature_info)
        right_layout.addWidget(creature_group)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Courier", 8))
        log_layout.addWidget(self._log_text)
        right_layout.addWidget(log_group)

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        world = self._sim.world
        if world is None:
            return

        self._lbl_epoch.setText(f"Epoch: {world.epoch_index}")
        self._lbl_tick.setText(f"Tick: {world.tick_index} / {TICKS_PER_EPOCH}")
        self._lbl_alive.setText(f"Alive: {self._sim.alive_count()} / {len(world.creatures)}")

        # Keep selected creature reference fresh
        if self._selected_creature is not None:
            cid = self._selected_creature.creature_id
            found = next((c for c in world.creatures if c.creature_id == cid), None)
            self._selected_creature = found
            self._refresh_creature_info()

        self._grid_widget._update_size()
        self._grid_widget.update()

    def _refresh_creature_info(self) -> None:
        c = self._selected_creature
        if c is None:
            self._creature_info.setPlainText("(no creature selected – click a cell)")
            return

        lt = c.lifetime
        genome = c.genome
        world = self._sim.world

        lines: list[str] = [
            f"ID: {c.creature_id}",
            f"Position: ({lt.x}, {lt.y})  Dir: {lt.direction.name}",
            f"Energy: {lt.energy:.1f}",
            f"Food eaten: {lt.food_eaten}",
            f"Age ticks: {lt.age_ticks}",
            f"Alive: {lt.alive}",
            f"Fitness: {compute_fitness(c):.2f}",
            "",
            "── Genome params ──",
            f"Genes: {len(genome.genes)}",
            f"Learning rate: {genome.learning_rate:.3f}",
            f"Reward decay: {genome.reward_decay:.3f}",
            f"Exploration: {genome.exploration_rate:.3f}",
            "",
        ]

        if world is not None and lt.alive:
            sensor = build_sensor_data(c, world)
            lines += [
                "── Sensor ──",
                f"Front: {sensor.front_cell.name}",
                f"Left:  {sensor.left_cell.name}",
                f"Right: {sensor.right_cell.name}",
                f"Back:  {sensor.back_cell.name}",
                f"Hunger bucket: {sensor.hunger_bucket}",
                f"Last action: {sensor.last_action.name if sensor.last_action else 'None'}",
                f"Last success: {sensor.last_action_success}",
                "",
                "── Top genes ──",
            ]
            scored = sorted(
                genome.genes,
                key=lambda g: score_gene(g, sensor, lt.learned_gene_adjustments),
                reverse=True,
            )
            for g in scored[:5]:
                ms  = score_gene_match(sensor, g.pattern)
                adj = lt.learned_gene_adjustments.get(g.gene_id, 0.0)
                ts  = score_gene(g, sensor, lt.learned_gene_adjustments)
                lines.append(
                    f"  Gene {g.gene_id}: {g.action.name}  "
                    f"match={ms:.1f} base={g.base_priority:.2f} "
                    f"adj={adj:.3f} → {ts:.2f}"
                )

        if lt.learned_gene_adjustments:
            lines += ["", "── Learned adjustments ──"]
            for gid, adj in sorted(lt.learned_gene_adjustments.items()):
                lines.append(f"  Gene {gid}: {adj:+.4f}")

        self._creature_info.setPlainText('\n'.join(lines))

    # ── Event handlers ─────────────────────────────────────────────────────────

    def on_grid_click(self, x: int, y: int) -> None:
        c = self._sim.find_creature_at(x, y)
        self._selected_creature = c
        if c is not None:
            self._grid_widget.set_selected(c.creature_id)
        else:
            self._grid_widget.set_selected(None)
        self._refresh_creature_info()

    def _on_start(self) -> None:
        if not self._running:
            self._running = True
            self._timer.start(self._timer_interval_ms)

    def _on_pause(self) -> None:
        self._running = False
        self._timer.stop()

    def _on_step_tick(self) -> None:
        self._on_pause()
        self._sim.step_tick()
        self._refresh()

    def _on_step_epoch(self) -> None:
        self._on_pause()
        self._sim.step_epoch()
        self._refresh()

    def _on_reset(self) -> None:
        self._on_pause()
        seed_text = self._seed_edit.text().strip()
        try:
            seed = int(seed_text)
        except ValueError:
            seed = None
        self._selected_creature = None
        self._grid_widget.set_selected(None)
        self._sim.reset(seed=seed)
        self._refresh()

    def _on_new_seed(self) -> None:
        import random as _random
        new_seed = _random.randint(0, 999_999)
        self._seed_edit.setText(str(new_seed))
        self._on_reset()

    def _on_timer(self) -> None:
        if self._sim.is_epoch_over():
            self._sim.step_epoch()
        else:
            self._sim.step_tick()
        self._refresh()

    def _append_log(self, message: str) -> None:
        self._log_text.append(message)
        # Auto-scroll to bottom
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )
