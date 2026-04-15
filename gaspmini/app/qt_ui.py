# app/qt_ui.py
# Qt GUI for GASPmini v1.
# Uses PySide6 widgets with a custom painted grid widget.

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QRectF, QSize, QSettings
from PySide6.QtGui import QColor, QPainter, QFont, QMouseEvent, QPen, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QGroupBox, QComboBox,
    QSizePolicy, QCheckBox, QSplitter, QTabWidget, QDialog,
)

import app.config as config
from app.custom_maps import get_custom_map_items
from app.models import ActionType, CellType, Creature, RunHistorySample, WorldState
from app.simulation_runner import SimulationRunner
from app.sensors import build_sensor_data
from app.gene_logic import score_gene, score_gene_match
from app.evolution import compute_fitness
from app.logging_utils import set_log_callback
from app.ui_settings import (
    make_app_settings,
    load_main_window_settings,
    save_main_window_settings,
)


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
EPOCH_HISTORY_WINDOW = 6
GRAPH_BG_COLOR = QColor(24, 24, 24)
GRAPH_FRAME_COLOR = QColor(78, 78, 78)
GRAPH_TEXT_COLOR = QColor(220, 220, 220)
ENERGY_LINE_COLOR = QColor(88, 180, 255)
FOOD_LINE_COLOR = QColor(110, 220, 120)
FAIL_LINE_COLOR = QColor(255, 170, 70)
DARK_BLUE_FITNESS_COLOR = QColor(18, 40, 110)
BRIGHT_CYAN_FITNESS_COLOR = QColor(0, 255, 255)
ACTION_MODEL_TEXT = (
    "Actions: Move Forward, Turn Left, Turn Right, Idle. "
    "Food is consumed automatically when a creature enters a food tile."
)
GRID_LEGEND_TEXT = (
    "<b>Legend</b>  "
    "<span style='color: rgb(60,160,60);'>■</span> Food  "
    "<span style='color: rgb(100,100,100);'>■</span> Wall  "
    "<span style='color: rgb(18,40,110);'>■</span> Creature low  "
    "<span style='color: rgb(0,255,255);'>■</span> Creature high  "
    "<span style='color: rgb(255,200,0);'>■</span> Selected"
)


# ── Grid widget ────────────────────────────────────────────────────────────────

class GridWidget(QWidget):
    """Custom-painted grid widget."""

    def __init__(self, sim: SimulationRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sim = sim
        self._selected_creature_id: Optional[int] = None
        self._max_seen_fitness = 0.0
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._update_size()

    def _native_grid_size(self) -> QSize:
        world = self._sim.world
        if world is None:
            return QSize(CELL_SIZE, CELL_SIZE)
        return QSize(world.width * CELL_SIZE, world.height * CELL_SIZE)

    def _update_size(self) -> None:
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        return self._native_grid_size()

    def minimumSizeHint(self) -> QSize:
        native = self._native_grid_size()
        return native.scaled(240, 160, Qt.AspectRatioMode.KeepAspectRatio)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        native = self._native_grid_size()
        if native.width() <= 0:
            return width
        return max(1, int(width * native.height() / native.width()))

    def _grid_viewport(self) -> tuple[QRectF, float]:
        native = self._native_grid_size()
        if native.width() <= 0 or native.height() <= 0:
            return QRectF(0.0, 0.0, float(self.width()), float(self.height())), 1.0

        scale = min(self.width() / native.width(), self.height() / native.height())
        if scale <= 0:
            return QRectF(0.0, 0.0, float(self.width()), float(self.height())), 1.0

        scaled_w = native.width() * scale
        scaled_h = native.height() * scale
        offset_x = (self.width() - scaled_w) / 2
        offset_y = (self.height() - scaled_h) / 2
        return QRectF(offset_x, offset_y, scaled_w, scaled_h), scale

    def set_selected(self, creature_id: Optional[int]) -> None:
        self._selected_creature_id = creature_id
        self.update()

    def set_max_seen_fitness(self, fitness: float) -> None:
        self._max_seen_fitness = max(0.0, fitness)

    def _creature_color(self, creature: Creature) -> QColor:
        if self._max_seen_fitness <= 0:
            return DARK_BLUE_FITNESS_COLOR

        fitness_fraction = max(0.0, compute_fitness(creature)) / self._max_seen_fitness
        return _interpolate_color(
            DARK_BLUE_FITNESS_COLOR,
            BRIGHT_CYAN_FITNESS_COLOR,
            fitness_fraction,
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        world = self._sim.world
        painter.fillRect(self.rect(), CELL_COLORS[CellType.EMPTY])
        if world is None:
            return

        viewport, scale = self._grid_viewport()
        painter.save()
        painter.translate(viewport.left(), viewport.top())
        painter.scale(scale, scale)

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
                    else:
                        color = self._creature_color(c)

                painter.fillRect(px, py, CELL_SIZE, CELL_SIZE, color)

                # Grid lines
                painter.setPen(QPen(GRID_LINE_COLOR, 0))
                painter.drawRect(px, py, CELL_SIZE - 1, CELL_SIZE - 1)

        painter.restore()
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._sim.world is None:
            return

        world = self._sim.world
        viewport, scale = self._grid_viewport()
        if scale <= 0 or not viewport.contains(event.position()):
            return

        local_x = (event.position().x() - viewport.left()) / scale
        local_y = (event.position().y() - viewport.top()) / scale
        x = int(local_x // CELL_SIZE)
        y = int(local_y // CELL_SIZE)

        # Ignore clicks outside the valid grid area
        if x < 0 or x >= world.width or y < 0 or y >= world.height:
            return

        # Find the parent MainWindow and notify
        parent = self.parent()
        while parent and not isinstance(parent, MainWindow):
            parent = parent.parent()
        if isinstance(parent, MainWindow):
            parent.on_grid_click(x, y)


class RunHistoryGraphWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._samples: list[RunHistorySample] = []
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_samples(self, samples: list[RunHistorySample]) -> None:
        self._samples = list(samples)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), GRAPH_BG_COLOR)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if not self._samples:
            painter.setPen(GRAPH_TEXT_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a creature to view run history")
            painter.end()
            return

        outer = self.rect().adjusted(10, 10, -10, -10)
        panels = [
            ("Energy", [sample.energy for sample in self._samples], ENERGY_LINE_COLOR),
            ("Food", [sample.food_eaten for sample in self._samples], FOOD_LINE_COLOR),
            ("Failures", [sample.failed_actions for sample in self._samples], FAIL_LINE_COLOR),
        ]
        gap = 8
        panel_height = max(40, int((outer.height() - gap * (len(panels) - 1)) / len(panels)))

        for index, (label, values, color) in enumerate(panels):
            top = outer.top() + index * (panel_height + gap)
            panel_rect = QRectF(float(outer.left()), float(top), float(outer.width()), float(panel_height))
            self._draw_panel(painter, panel_rect, label, values, color)

        painter.end()

    def _draw_panel(
        self,
        painter: QPainter,
        panel_rect: QRectF,
        label: str,
        values: list[float],
        color: QColor,
    ) -> None:
        painter.setPen(QPen(GRAPH_FRAME_COLOR, 1))
        painter.drawRoundedRect(panel_rect, 4, 4)

        painter.setPen(GRAPH_TEXT_COLOR)
        current_value = values[-1]
        max_value = max(values) if values else 0
        painter.drawText(
            panel_rect.adjusted(8, 6, -8, -6),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{label}"
        )
        painter.drawText(
            panel_rect.adjusted(8, 6, -8, -6),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"now {current_value:.1f}  max {max_value:.1f}"
        )

        plot_rect = panel_rect.adjusted(10, 24, -10, -10)
        painter.setPen(QPen(GRAPH_FRAME_COLOR, 1))
        painter.drawRect(plot_rect)

        if len(values) == 1:
            painter.setPen(QPen(color, 2))
            center_y = plot_rect.center().y()
            painter.drawLine(plot_rect.left(), center_y, plot_rect.right(), center_y)
            return

        value_min = min(values)
        value_max = max(values)
        if value_max == value_min:
            value_max = value_min + 1.0

        mid_y = plot_rect.top() + plot_rect.height() / 2
        painter.setPen(QPen(GRAPH_FRAME_COLOR, 1, Qt.PenStyle.DashLine))
        painter.drawLine(plot_rect.left(), mid_y, plot_rect.right(), mid_y)

        painter.setPen(QPen(color, 2))
        point_count = len(values)
        for index in range(point_count - 1):
            x1 = plot_rect.left() + plot_rect.width() * index / max(1, point_count - 1)
            x2 = plot_rect.left() + plot_rect.width() * (index + 1) / max(1, point_count - 1)
            y1 = self._value_to_y(values[index], value_min, value_max, plot_rect)
            y2 = self._value_to_y(values[index + 1], value_min, value_max, plot_rect)
            painter.drawLine(x1, y1, x2, y2)

        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        last_x = plot_rect.right()
        last_y = self._value_to_y(values[-1], value_min, value_max, plot_rect)
        painter.drawEllipse(QRectF(last_x - 3, last_y - 3, 6, 6))

        painter.setPen(GRAPH_TEXT_COLOR)
        start_tick = self._samples[0].age_ticks
        end_tick = self._samples[-1].age_ticks
        painter.drawText(
            plot_rect.adjusted(2, 0, -2, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            f"t={start_tick}"
        )
        painter.drawText(
            plot_rect.adjusted(2, 0, -2, 0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"t={end_tick}"
        )

    @staticmethod
    def _value_to_y(value: float, value_min: float, value_max: float, rect: QRectF) -> float:
        fraction = (value - value_min) / (value_max - value_min)
        return rect.bottom() - fraction * rect.height()


def _set_text_preserving_scroll(text_edit: QTextEdit, text: str) -> None:
    if text_edit.toPlainText() == text:
        return

    scroll_bar = text_edit.verticalScrollBar()
    old_value = scroll_bar.value()
    old_maximum = scroll_bar.maximum()
    was_at_bottom = old_value >= max(0, old_maximum - 2)

    text_edit.setPlainText(text)

    new_scroll_bar = text_edit.verticalScrollBar()
    if was_at_bottom:
        new_scroll_bar.setValue(new_scroll_bar.maximum())
    else:
        new_scroll_bar.setValue(min(old_value, new_scroll_bar.maximum()))


def _format_action_name(action: ActionType | None) -> str:
    if action is None:
        return "None"
    return action.name.replace('_', ' ').title()


def _interpolate_color(start: QColor, end: QColor, fraction: float) -> QColor:
    clamped = max(0.0, min(1.0, fraction))
    red = round(start.red() + (end.red() - start.red()) * clamped)
    green = round(start.green() + (end.green() - start.green()) * clamped)
    blue = round(start.blue() + (end.blue() - start.blue()) * clamped)
    return QColor(red, green, blue)


class InspectorWindow(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GASPmini Inspector")
        self.resize(560, 760)

        layout = QVBoxLayout(self)

        self._summary_label = QLabel("No creature selected")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        creature_tab = QWidget()
        creature_layout = QVBoxLayout(creature_tab)
        creature_splitter = QSplitter(Qt.Orientation.Vertical)
        creature_layout.addWidget(creature_splitter)

        self._creature_info = QTextEdit()
        self._creature_info.setReadOnly(True)
        self._creature_info.setFont(QFont("Courier", 9))
        creature_splitter.addWidget(self._creature_info)

        self._creature_history_graph = RunHistoryGraphWidget()
        creature_splitter.addWidget(self._creature_history_graph)
        creature_splitter.setStretchFactor(0, 3)
        creature_splitter.setStretchFactor(1, 2)
        creature_splitter.setSizes([420, 300])

        tabs.addTab(creature_tab, "Creature")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Courier", 8))
        log_layout.addWidget(self._log_text)
        tabs.addTab(log_tab, "Log")

    def update_creature(self, summary: str, details: str, samples: list[RunHistorySample]) -> None:
        self._summary_label.setText(summary)
        _set_text_preserving_scroll(self._creature_info, details)
        self._creature_history_graph.set_samples(samples)

    def clear_creature(self, message: str) -> None:
        self._summary_label.setText(f"No creature selected. {ACTION_MODEL_TEXT}")
        _set_text_preserving_scroll(self._creature_info, message)
        self._creature_history_graph.set_samples([])

    def append_log(self, message: str) -> None:
        scroll_bar = self._log_text.verticalScrollBar()
        was_at_bottom = scroll_bar.value() >= max(0, scroll_bar.maximum() - 2)
        self._log_text.append(message)
        if was_at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GASPmini")

        self._sim = SimulationRunner(profile_id=config.DEFAULT_PROFILE_ID)
        self._sim.reset()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._running = False
        self._timer_interval_ms = 100   # ms between ticks when running

        self._selected_creature: Optional[Creature] = None
        self._settings: QSettings = make_app_settings()
        self._applying_ui_settings = False
        self._max_seen_fitness = 0.0

        # Redirect log output to the log panel
        set_log_callback(self._append_log)

        self._build_ui()
        self._load_ui_settings()
        self._refresh()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        self._inspector = InspectorWindow(self)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(self._main_splitter)

        # ── Left: headline + grid + legend ───────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self._lbl_current_max_fitness_title = QLabel("Curr Max Fitness:")
        title_font = self._lbl_current_max_fitness_title.font()
        title_font.setPointSize(max(8, title_font.pointSize() - 1))
        self._lbl_current_max_fitness_title.setFont(title_font)
        left_layout.addWidget(self._lbl_current_max_fitness_title)

        self._lbl_current_max_fitness_value = QLabel("0.00")
        fitness_font = QFont("Courier", 28)
        fitness_font.setBold(True)
        self._lbl_current_max_fitness_value.setFont(fitness_font)
        self._lbl_current_max_fitness_value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        left_layout.addWidget(self._lbl_current_max_fitness_value)

        self._grid_widget = GridWidget(self._sim)
        left_layout.addWidget(self._grid_widget, 1)

        self._lbl_grid_legend = QLabel(GRID_LEGEND_TEXT)
        self._lbl_grid_legend.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_grid_legend.setWordWrap(True)
        left_layout.addWidget(self._lbl_grid_legend)

        self._main_splitter.addWidget(left_panel)

        # ── Right: controls + compact status ──────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self._main_splitter.addWidget(right_panel)
        self._main_splitter.setStretchFactor(0, 4)
        self._main_splitter.setStretchFactor(1, 1)

        # Simulation status
        status_group = QGroupBox("Simulation")
        status_layout = QVBoxLayout(status_group)
        self._lbl_mode = QLabel("Mode: Evolution")
        self._lbl_epoch = QLabel("Epoch: 0")
        self._lbl_tick  = QLabel("Tick: 0")
        self._lbl_alive = QLabel("Alive: 0")
        self._lbl_selected = QLabel("Selected: none")
        for lbl in (self._lbl_mode, self._lbl_epoch, self._lbl_tick, self._lbl_alive, self._lbl_selected):
            status_layout.addWidget(lbl)
        right_layout.addWidget(status_group)

        action_group = QGroupBox("Action Model")
        action_layout = QVBoxLayout(action_group)
        self._lbl_action_model = QLabel(ACTION_MODEL_TEXT)
        self._lbl_action_model.setWordWrap(True)
        action_layout.addWidget(self._lbl_action_model)
        right_layout.addWidget(action_group)

        history_group = QGroupBox("Epoch Bests")
        history_layout = QVBoxLayout(history_group)
        self._epoch_history_text = QTextEdit()
        self._epoch_history_text.setReadOnly(True)
        self._epoch_history_text.setMinimumHeight(100)
        self._epoch_history_text.setMaximumHeight(180)
        self._epoch_history_text.setFont(QFont("Courier", 8))
        history_layout.addWidget(self._epoch_history_text)
        right_layout.addWidget(history_group)

        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self._btn_run       = QPushButton("▶ Start")
        self._btn_step_tick = QPushButton("Step Tick")
        self._btn_step_epoch= QPushButton("Step Epoch")
        self._btn_reset     = QPushButton("Reset")
        self._btn_select_best = QPushButton("Select the Best")
        self._btn_testing_ground = QPushButton("Enter Testing Ground")
        self._btn_open_inspector = QPushButton("Open Inspector")
        self._btn_testing_ground.setCheckable(True)

        self._btn_run.clicked.connect(self._on_toggle_running)
        self._btn_step_tick.clicked.connect(self._on_step_tick)
        self._btn_step_epoch.clicked.connect(self._on_step_epoch)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_select_best.clicked.connect(self._on_select_best)
        self._btn_testing_ground.toggled.connect(self._on_testing_ground_toggled)
        self._btn_open_inspector.clicked.connect(self._on_open_inspector)

        for btn in (self._btn_run, self._btn_step_tick,
            self._btn_step_epoch, self._btn_reset, self._btn_select_best,
            self._btn_testing_ground, self._btn_open_inspector):
            ctrl_layout.addWidget(btn)

        # Seed control
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self._seed_edit = QLineEdit(str(self._sim.seed))
        self._seed_edit.setFixedWidth(80)
        self._seed_edit.editingFinished.connect(self._save_ui_settings)
        seed_layout.addWidget(self._seed_edit)
        btn_new_seed = QPushButton("New Seed")
        btn_new_seed.clicked.connect(self._on_new_seed)
        seed_layout.addWidget(btn_new_seed)
        ctrl_layout.addLayout(seed_layout)

        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Profile:"))
        self._profile_combo = QComboBox()
        for profile_id, label in config.get_profile_items():
            self._profile_combo.addItem(label, profile_id)
        profile_index = self._profile_combo.findData(self._sim.profile_id)
        if profile_index >= 0:
            self._profile_combo.setCurrentIndex(profile_index)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        profile_layout.addWidget(self._profile_combo)
        ctrl_layout.addLayout(profile_layout)

        custom_map_layout = QHBoxLayout()
        custom_map_layout.addWidget(QLabel("Map:"))
        self._custom_map_combo = QComboBox()
        self._custom_map_combo.addItem("Seeded Random", "")
        for map_id, label in get_custom_map_items():
            self._custom_map_combo.addItem(label, map_id)
        custom_map_index = self._custom_map_combo.findData(self._sim.custom_map_id or "")
        if custom_map_index >= 0:
            self._custom_map_combo.setCurrentIndex(custom_map_index)
        self._custom_map_combo.currentIndexChanged.connect(self._on_custom_map_changed)
        custom_map_layout.addWidget(self._custom_map_combo)
        ctrl_layout.addLayout(custom_map_layout)

        # Ticks per epoch control
        tpe_layout = QHBoxLayout()
        tpe_layout.addWidget(QLabel("Ticks/Epoch:"))
        self._tpe_edit = QLineEdit(str(self._sim.ticks_per_epoch))
        self._tpe_edit.setFixedWidth(80)
        self._tpe_edit.editingFinished.connect(self._save_ui_settings)
        tpe_layout.addWidget(self._tpe_edit)
        ctrl_layout.addLayout(tpe_layout)

        persistence_group = QGroupBox("Best Creature Persistence")
        persistence_layout = QVBoxLayout(persistence_group)
        self._autosave_best_checkbox = QCheckBox("Autosave best creature")
        self._inject_saved_best_checkbox = QCheckBox("Inject saved creature on reset")
        self._autosave_best_checkbox.toggled.connect(self._save_ui_settings)
        self._inject_saved_best_checkbox.toggled.connect(self._save_ui_settings)
        persistence_layout.addWidget(self._autosave_best_checkbox)
        persistence_layout.addWidget(self._inject_saved_best_checkbox)

        autosave_path_layout = QHBoxLayout()
        autosave_path_layout.addWidget(QLabel("File:"))
        self._autosave_best_path_edit = QLineEdit(self._sim.autosave_best_path)
        self._autosave_best_path_edit.editingFinished.connect(self._save_ui_settings)
        autosave_path_layout.addWidget(self._autosave_best_path_edit)
        persistence_layout.addLayout(autosave_path_layout)

        self._btn_save_best_now = QPushButton("Save Best Now")
        self._btn_load_saved_best = QPushButton("Check Saved File")
        self._btn_save_best_now.clicked.connect(self._on_save_best_now)
        self._btn_load_saved_best.clicked.connect(self._on_check_saved_best)
        persistence_layout.addWidget(self._btn_save_best_now)
        persistence_layout.addWidget(self._btn_load_saved_best)
        ctrl_layout.addWidget(persistence_group)

        right_layout.addWidget(ctrl_group)
        right_layout.addStretch(1)

        native_grid = self._grid_widget.sizeHint()
        self.resize(native_grid.width() + 420, max(native_grid.height(), 720))
        self._main_splitter.setSizes([native_grid.width(), 360])

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        world = self._sim.world
        if world is None:
            return

        self._sync_selected_creature()

        mode_label = "Testing Ground" if self._sim.is_testing_ground() else "Evolution"
        self._lbl_mode.setText(f"Mode: {mode_label}")
        self._sync_persistence_settings_from_ui()
        self._lbl_epoch.setText(f"Epoch: {world.epoch_index}")
        if self._sim.is_testing_ground():
            self._lbl_tick.setText(f"Tick: {world.tick_index}")
        else:
            self._lbl_tick.setText(f"Tick: {world.tick_index} / {self._sim.ticks_per_epoch}")
        self._lbl_alive.setText(f"Alive: {self._sim.alive_count()} / {len(world.creatures)}")
        self._lbl_selected.setText(self._selected_creature_status_text())
        self._update_run_button_label()
        self._refresh_current_max_fitness_label()
        self._refresh_epoch_history()
        self._refresh_creature_info()

        self._grid_widget._update_size()
        self._grid_widget.update()

    def _refresh_current_max_fitness_label(self) -> None:
        best_creature = self._sim.current_best_creature()
        if best_creature is None:
            self._lbl_current_max_fitness_value.setText("0.00")
            self._lbl_current_max_fitness_value.setStyleSheet("")
            self._grid_widget.set_max_seen_fitness(self._max_seen_fitness)
            return

        current_fitness = compute_fitness(best_creature)
        reference_fitness = max(
            self._max_seen_fitness,
            current_fitness,
            self._sim.best_fitness_ever or 0.0,
        )
        self._max_seen_fitness = reference_fitness

        self._lbl_current_max_fitness_value.setText(f"{current_fitness:.2f}")
        self._lbl_current_max_fitness_value.setStyleSheet("")
        self._grid_widget.set_max_seen_fitness(reference_fitness)

    def _refresh_epoch_history(self) -> None:
        history = self._sim.epoch_history
        if not history:
            self._epoch_history_text.setPlainText("(no completed epochs yet)")
            return

        lines: list[str] = []
        if self._sim.best_fitness_ever is not None and self._sim.best_epoch_ever is not None:
            lines.append(
                f"All-time best: {self._sim.best_fitness_ever:.2f} "
                f"(epoch {self._sim.best_epoch_ever})"
            )
            lines.append("")

        if len(history) <= EPOCH_HISTORY_WINDOW:
            visible = history
        else:
            rotation_start = 0
            if self._sim.world is not None:
                rotation_start = self._sim.world.epoch_index % len(history)
            visible = history[rotation_start:] + history[:rotation_start]
            visible = visible[:EPOCH_HISTORY_WINDOW]

        lines.extend(
            f"Epoch {entry['epoch']}: best {entry['top_fitness']:.2f}"
            for entry in visible
        )
        self._epoch_history_text.setPlainText('\n'.join(lines))

    def _refresh_creature_info(self) -> None:
        c = self._selected_creature
        if c is None:
            self._inspector.clear_creature("(no creature selected – click a cell)")
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
            "── Action model ──",
            ACTION_MODEL_TEXT,
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
                f"Current: {sensor.current_cell.name}",
                f"Front: {sensor.front_cell.name}",
                f"Left:  {sensor.left_cell.name}",
                f"Right: {sensor.right_cell.name}",
                f"Back:  {sensor.back_cell.name}",
                f"Hunger bucket: {sensor.hunger_bucket}",
                f"Last action: {_format_action_name(sensor.last_action)}",
                f"Last success: {sensor.last_action_success}",
                "",
                "── Top genes ──",
            ]
            scored = sorted(
                genome.genes,
                key=lambda g: score_gene(g, sensor, lt.learned_gene_adjustments, c),
                reverse=True,
            )
            for g in scored[:5]:
                ms  = score_gene_match(sensor, g.pattern)
                adj = lt.learned_gene_adjustments.get(g.gene_id, 0.0)
                ts  = score_gene(g, sensor, lt.learned_gene_adjustments, c)
                lines.append(
                    f"  Gene {g.gene_id}: {_format_action_name(g.action)}  "
                    f"match={ms:.1f} base={g.base_priority:.2f} "
                    f"adj={adj:.3f} → {ts:.2f}"
                )

        if lt.learned_gene_adjustments:
            lines += ["", "── Learned adjustments ──"]
            for gid, adj in sorted(lt.learned_gene_adjustments.items()):
                lines.append(f"  Gene {gid}: {adj:+.4f}")

        detail_text = '\n'.join(lines)
        summary = (
            f"Creature {c.creature_id}  |  Pos ({lt.x}, {lt.y})  |  "
            f"Energy {lt.energy:.1f}  |  Fitness {compute_fitness(c):.2f}"
        )
        self._inspector.update_creature(summary, detail_text, lt.run_history)

    def _selected_creature_status_text(self) -> str:
        if self._selected_creature is None:
            return "Selected: none"
        lt = self._selected_creature.lifetime
        return (
            f"Selected: #{self._selected_creature.creature_id} at ({lt.x}, {lt.y}), "
            f"energy {lt.energy:.1f}"
        )

    def _sync_selected_creature(self) -> None:
        world = self._sim.world
        if world is None:
            self._selected_creature = None
            self._grid_widget.set_selected(None)
            return

        if self._selected_creature is not None:
            cid = self._selected_creature.creature_id
            self._selected_creature = next((c for c in world.creatures if c.creature_id == cid), None)

        if self._sim.is_testing_ground():
            if self._selected_creature is None:
                self._selected_creature = self._sim.current_best_creature()

        selected_id = self._selected_creature.creature_id if self._selected_creature is not None else None
        self._grid_widget.set_selected(selected_id)

    # ── Event handlers ─────────────────────────────────────────────────────────

    def on_grid_click(self, x: int, y: int) -> None:
        c = self._sim.find_creature_at(x, y)
        self._selected_creature = c
        if c is not None:
            self._grid_widget.set_selected(c.creature_id)
        else:
            self._grid_widget.set_selected(None)
        self._refresh_creature_info()

    def _update_run_button_label(self) -> None:
        if self._running:
            self._btn_run.setText("⏸ Pause")
        else:
            self._btn_run.setText("▶ Start")

    def _on_toggle_running(self) -> None:
        if self._running:
            self._on_pause()
        else:
            self._on_start()

    def _on_start(self) -> None:
        if not self._running:
            self._running = True
            self._timer.start(self._timer_interval_ms)
            self._update_run_button_label()

    def _on_pause(self) -> None:
        self._running = False
        self._timer.stop()
        self._update_run_button_label()

    def _on_select_best(self) -> None:
        self._on_pause()
        best_creature = self._sim.current_best_creature()
        self._selected_creature = best_creature
        if best_creature is None:
            self._grid_widget.set_selected(None)
            self._append_log("No current best creature is available to select.")
        else:
            self._grid_widget.set_selected(best_creature.creature_id)
        self._refresh()

    def _on_step_tick(self) -> None:
        self._on_pause()
        self._sim.step_tick()
        self._refresh()

    def _on_step_epoch(self) -> None:
        self._on_pause()
        self._sim.step_epoch()
        self._refresh()

    def _on_profile_changed(self) -> None:
        if self._applying_ui_settings:
            return

        profile_id = self._profile_combo.currentData()
        if profile_id is None or profile_id == self._sim.profile_id:
            return

        self._on_pause()
        self._sim.set_profile(profile_id)
        self._tpe_edit.setText(str(self._sim.ticks_per_epoch))
        self._selected_creature = None
        self._grid_widget.set_selected(None)
        self._sim.reset(preserve_hall_of_fame=self._sim.is_testing_ground())
        self._save_ui_settings()
        self._refresh()

    def _on_custom_map_changed(self) -> None:
        if self._applying_ui_settings:
            return

        custom_map_id = str(self._custom_map_combo.currentData() or '')
        next_custom_map_id = custom_map_id or None
        if next_custom_map_id == self._sim.custom_map_id:
            return

        self._on_pause()
        self._sim.set_custom_map(next_custom_map_id)
        self._selected_creature = None
        self._grid_widget.set_selected(None)
        self._sim.reset(preserve_hall_of_fame=self._sim.is_testing_ground())
        self._save_ui_settings()
        self._refresh()

    def _on_reset(self) -> None:
        self._on_pause()
        self._sync_persistence_settings_from_ui()
        seed_text = self._seed_edit.text().strip()
        try:
            seed = int(seed_text)
        except ValueError:
            seed = None
        tpe_text = self._tpe_edit.text().strip()
        try:
            tpe = int(tpe_text)
            if 1 <= tpe <= 10_000:
                self._sim.ticks_per_epoch = tpe
            else:
                self._tpe_edit.setText(str(self._sim.ticks_per_epoch))
        except ValueError:
            self._tpe_edit.setText(str(self._sim.ticks_per_epoch))
        self._selected_creature = None
        self._grid_widget.set_selected(None)
        self._sim.reset(seed=seed, preserve_hall_of_fame=self._sim.is_testing_ground())
        self._save_ui_settings()
        self._refresh()

    def _on_new_seed(self) -> None:
        import random as _random
        new_seed = _random.randint(0, 999_999)
        self._seed_edit.setText(str(new_seed))
        self._on_reset()

    def _on_timer(self) -> None:
        if self._sim.is_testing_ground() and self._sim.is_run_complete():
            self._on_pause()
            self._refresh()
            return

        self._sim.step_tick()
        if self._sim.is_testing_ground() and self._sim.is_run_complete():
            self._on_pause()
        self._refresh()

    def _on_testing_ground_toggled(self, checked: bool) -> None:
        self._on_pause()

        if checked:
            if not self._sim.enter_testing_ground():
                self._append_log("Testing Ground unavailable: no best or current creature to clone.")
                self._btn_testing_ground.blockSignals(True)
                self._btn_testing_ground.setChecked(False)
                self._btn_testing_ground.blockSignals(False)
                self._btn_testing_ground.setText("Enter Testing Ground")
                return
            self._btn_testing_ground.setText("Exit Testing Ground")
            self._selected_creature = self._sim.current_best_creature()
        else:
            if self._sim.is_testing_ground():
                self._sim.exit_testing_ground()
            self._btn_testing_ground.setText("Enter Testing Ground")
            self._selected_creature = None
            self._grid_widget.set_selected(None)

        self._save_ui_settings()
        self._refresh()

    def _on_open_inspector(self) -> None:
        self._inspector.show()
        self._inspector.raise_()
        self._inspector.activateWindow()

    def _sync_persistence_settings_from_ui(self) -> None:
        self._sim.configure_best_genome_persistence(
            autosave_enabled=self._autosave_best_checkbox.isChecked(),
            autosave_path=self._autosave_best_path_edit.text().strip(),
            inject_saved_best_enabled=self._inject_saved_best_checkbox.isChecked(),
        )

    def _load_ui_settings(self) -> None:
        values = load_main_window_settings(
            self._settings,
            default_profile_id=config.DEFAULT_PROFILE_ID,
            default_custom_map_id='',
            default_ticks_per_epoch=self._sim.ticks_per_epoch,
            default_seed=self._sim.seed,
        )

        self._applying_ui_settings = True
        self._autosave_best_checkbox.setChecked(bool(values['autosave_enabled']))
        self._inject_saved_best_checkbox.setChecked(bool(values['inject_saved_best_enabled']))
        self._autosave_best_path_edit.setText(str(values['autosave_path']))

        profile_index = self._profile_combo.findData(values['profile_id'])
        if profile_index >= 0:
            self._profile_combo.setCurrentIndex(profile_index)
        custom_map_index = self._custom_map_combo.findData(values['custom_map_id'])
        if custom_map_index >= 0:
            self._custom_map_combo.setCurrentIndex(custom_map_index)
        self._tpe_edit.setText(str(values['ticks_per_epoch']))
        self._seed_edit.setText(str(values['seed']))
        self._btn_testing_ground.blockSignals(True)
        self._btn_testing_ground.setChecked(False)
        self._btn_testing_ground.blockSignals(False)
        self._btn_testing_ground.setText("Enter Testing Ground")
        self._applying_ui_settings = False

        profile_id = self._profile_combo.currentData()
        if profile_id is not None and profile_id != self._sim.profile_id:
            self._sim.set_profile(profile_id)

        custom_map_id = str(self._custom_map_combo.currentData() or '')
        if (custom_map_id or None) != self._sim.custom_map_id:
            self._sim.set_custom_map(custom_map_id or None)

        ticks_per_epoch = self._parse_ticks_per_epoch_from_ui(default=self._sim.ticks_per_epoch)
        self._sim.ticks_per_epoch = ticks_per_epoch
        self._sync_persistence_settings_from_ui()

        seed = self._parse_seed_from_ui(default=self._sim.seed)
        self._sim.reset(seed=seed)

        if bool(values['testing_ground_enabled']):
            if self._sim.enter_testing_ground():
                self._btn_testing_ground.blockSignals(True)
                self._btn_testing_ground.setChecked(True)
                self._btn_testing_ground.blockSignals(False)
                self._btn_testing_ground.setText("Exit Testing Ground")
                self._selected_creature = self._sim.current_best_creature()
            else:
                self._btn_testing_ground.setText("Enter Testing Ground")

        main_geometry = values.get('main_window_geometry')
        if main_geometry:
            self.restoreGeometry(main_geometry)

        splitter_state = values.get('main_splitter_state')
        if splitter_state:
            self._main_splitter.restoreState(splitter_state)

        inspector_geometry = values.get('inspector_geometry')
        if inspector_geometry:
            self._inspector.restoreGeometry(inspector_geometry)

        if bool(values.get('inspector_visible')):
            self._inspector.show()

    def _save_ui_settings(self) -> None:
        if self._applying_ui_settings:
            return

        save_main_window_settings(
            self._settings,
            autosave_enabled=self._autosave_best_checkbox.isChecked(),
            inject_saved_best_enabled=self._inject_saved_best_checkbox.isChecked(),
            autosave_path=self._autosave_best_path_edit.text().strip(),
            profile_id=str(self._profile_combo.currentData() or self._sim.profile_id),
            custom_map_id=str(self._custom_map_combo.currentData() or ''),
            ticks_per_epoch=self._parse_ticks_per_epoch_from_ui(default=self._sim.ticks_per_epoch),
            seed=self._parse_seed_from_ui(default=self._sim.seed),
            testing_ground_enabled=self._sim.is_testing_ground(),
            main_window_geometry=self.saveGeometry(),
            main_splitter_state=self._main_splitter.saveState(),
            inspector_geometry=self._inspector.saveGeometry(),
            inspector_visible=self._inspector.isVisible(),
        )
        self._sync_persistence_settings_from_ui()

    def _parse_seed_from_ui(self, default: int) -> int:
        seed_text = self._seed_edit.text().strip()
        try:
            return int(seed_text)
        except ValueError:
            return default

    def _parse_ticks_per_epoch_from_ui(self, default: int) -> int:
        ticks_text = self._tpe_edit.text().strip()
        try:
            ticks = int(ticks_text)
        except ValueError:
            return default
        if 1 <= ticks <= 10_000:
            return ticks
        return default

    def _on_save_best_now(self) -> None:
        self._sync_persistence_settings_from_ui()
        if self._sim.best_genome_ever is None:
            self._append_log("No best creature is available to save yet.")
            return
        from app.genome_store import save_genome_to_file
        save_genome_to_file(self._sim.best_genome_ever, self._sim.autosave_best_path)
        self._append_log(f"Saved best creature to {self._sim.autosave_best_path}")

    def _on_check_saved_best(self) -> None:
        self._sync_persistence_settings_from_ui()
        if self._sim.has_saved_best_genome():
            self._append_log(f"Saved best creature found at {self._sim.autosave_best_path}")
        else:
            self._append_log(f"No saved best creature found at {self._sim.autosave_best_path}")

    def _append_log(self, message: str) -> None:
        self._inspector.append_log(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_ui_settings()
        super().closeEvent(event)
