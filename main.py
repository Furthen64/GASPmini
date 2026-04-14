# main.py
# Entry point for GASPmini.
# Run with: python main.py
# For console-only mode (no Qt): python main.py --console

from __future__ import annotations

import sys
import os

# Ensure the gaspmini package directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gaspmini'))


def run_console() -> None:
    """Quick console smoke-test: build a world and print it."""
    from app.world import generate_world, render_ascii
    from app.simulation import tick_world

    print("=== GASPmini console mode ===")
    world = generate_world(epoch_index=0, seed=42)
    print(f"Epoch 0, Tick 0  ({world.width}x{world.height})")
    print(render_ascii(world))
    print(f"Creatures: {len(world.creatures)}  Food: {len(world.food_positions)}")
    print()

    # Step 5 ticks
    for _ in range(5):
        tick_world(world)

    print(f"After 5 ticks (tick={world.tick_index}):")
    print(render_ascii(world))


def run_gui() -> None:
    from PySide6.QtWidgets import QApplication
    from app.qt_ui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    if '--console' in sys.argv:
        run_console()
    else:
        run_gui()
