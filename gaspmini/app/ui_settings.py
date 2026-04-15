from __future__ import annotations

from PySide6.QtCore import QSettings


SETTINGS_ORGANIZATION = 'GASPmini'
SETTINGS_APPLICATION = 'GASPmini'
AUTOSAVE_BEST_ENABLED_KEY = 'best_creature/autosave_enabled'
INJECT_SAVED_BEST_ENABLED_KEY = 'best_creature/inject_saved_best_enabled'
AUTOSAVE_BEST_PATH_KEY = 'best_creature/autosave_path'
PROFILE_ID_KEY = 'simulation/profile_id'
TICKS_PER_EPOCH_KEY = 'simulation/ticks_per_epoch'
SEED_KEY = 'simulation/seed'
TESTING_GROUND_ENABLED_KEY = 'simulation/testing_ground_enabled'


def make_app_settings() -> QSettings:
    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def load_best_creature_persistence_settings(settings: QSettings) -> dict[str, object]:
    return {
        'autosave_enabled': _to_bool(settings.value(AUTOSAVE_BEST_ENABLED_KEY, False)),
        'inject_saved_best_enabled': _to_bool(settings.value(INJECT_SAVED_BEST_ENABLED_KEY, False)),
        'autosave_path': str(settings.value(AUTOSAVE_BEST_PATH_KEY, 'saved_best_creature.json')),
    }


def save_best_creature_persistence_settings(
    settings: QSettings,
    *,
    autosave_enabled: bool,
    inject_saved_best_enabled: bool,
    autosave_path: str,
) -> None:
    settings.setValue(AUTOSAVE_BEST_ENABLED_KEY, autosave_enabled)
    settings.setValue(INJECT_SAVED_BEST_ENABLED_KEY, inject_saved_best_enabled)
    settings.setValue(AUTOSAVE_BEST_PATH_KEY, autosave_path)
    settings.sync()


def load_main_window_settings(
    settings: QSettings,
    *,
    default_profile_id: str,
    default_ticks_per_epoch: int,
    default_seed: int,
) -> dict[str, object]:
    values = load_best_creature_persistence_settings(settings)
    values.update({
        'profile_id': str(settings.value(PROFILE_ID_KEY, default_profile_id)),
        'ticks_per_epoch': _to_int(settings.value(TICKS_PER_EPOCH_KEY, default_ticks_per_epoch), default_ticks_per_epoch),
        'seed': _to_int(settings.value(SEED_KEY, default_seed), default_seed),
        'testing_ground_enabled': _to_bool(settings.value(TESTING_GROUND_ENABLED_KEY, False)),
    })
    return values


def save_main_window_settings(
    settings: QSettings,
    *,
    autosave_enabled: bool,
    inject_saved_best_enabled: bool,
    autosave_path: str,
    profile_id: str,
    ticks_per_epoch: int,
    seed: int,
    testing_ground_enabled: bool,
) -> None:
    save_best_creature_persistence_settings(
        settings,
        autosave_enabled=autosave_enabled,
        inject_saved_best_enabled=inject_saved_best_enabled,
        autosave_path=autosave_path,
    )
    settings.setValue(PROFILE_ID_KEY, profile_id)
    settings.setValue(TICKS_PER_EPOCH_KEY, ticks_per_epoch)
    settings.setValue(SEED_KEY, seed)
    settings.setValue(TESTING_GROUND_ENABLED_KEY, testing_ground_enabled)
    settings.sync()


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default