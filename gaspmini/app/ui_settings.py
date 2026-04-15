from __future__ import annotations

from PySide6.QtCore import QSettings


SETTINGS_ORGANIZATION = 'GASPmini'
SETTINGS_APPLICATION = 'GASPmini'
AUTOSAVE_BEST_ENABLED_KEY = 'best_creature/autosave_enabled'
INJECT_SAVED_BEST_ENABLED_KEY = 'best_creature/inject_saved_best_enabled'
AUTOSAVE_BEST_PATH_KEY = 'best_creature/autosave_path'
PROFILE_ID_KEY = 'simulation/profile_id'
CUSTOM_MAP_ID_KEY = 'simulation/custom_map_id'
TICKS_PER_EPOCH_KEY = 'simulation/ticks_per_epoch'
SEED_KEY = 'simulation/seed'
TESTING_GROUND_ENABLED_KEY = 'simulation/testing_ground_enabled'
MAIN_WINDOW_GEOMETRY_KEY = 'ui/main_window_geometry'
MAIN_SPLITTER_STATE_KEY = 'ui/main_splitter_state'
INSPECTOR_GEOMETRY_KEY = 'ui/inspector_geometry'
INSPECTOR_VISIBLE_KEY = 'ui/inspector_visible'


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
    default_custom_map_id: str,
    default_ticks_per_epoch: int,
    default_seed: int,
) -> dict[str, object]:
    values = load_best_creature_persistence_settings(settings)
    values.update({
        'profile_id': str(settings.value(PROFILE_ID_KEY, default_profile_id)),
        'custom_map_id': str(settings.value(CUSTOM_MAP_ID_KEY, default_custom_map_id)),
        'ticks_per_epoch': _to_int(settings.value(TICKS_PER_EPOCH_KEY, default_ticks_per_epoch), default_ticks_per_epoch),
        'seed': _to_int(settings.value(SEED_KEY, default_seed), default_seed),
        'testing_ground_enabled': _to_bool(settings.value(TESTING_GROUND_ENABLED_KEY, False)),
    })
    if settings.contains(MAIN_WINDOW_GEOMETRY_KEY):
        values['main_window_geometry'] = settings.value(MAIN_WINDOW_GEOMETRY_KEY)
    if settings.contains(MAIN_SPLITTER_STATE_KEY):
        values['main_splitter_state'] = settings.value(MAIN_SPLITTER_STATE_KEY)
    if settings.contains(INSPECTOR_GEOMETRY_KEY):
        values['inspector_geometry'] = settings.value(INSPECTOR_GEOMETRY_KEY)
    if settings.contains(INSPECTOR_VISIBLE_KEY):
        values['inspector_visible'] = _to_bool(settings.value(INSPECTOR_VISIBLE_KEY, False))
    return values


def save_main_window_settings(
    settings: QSettings,
    *,
    autosave_enabled: bool,
    inject_saved_best_enabled: bool,
    autosave_path: str,
    profile_id: str,
    custom_map_id: str,
    ticks_per_epoch: int,
    seed: int,
    testing_ground_enabled: bool,
    main_window_geometry: object = None,
    main_splitter_state: object = None,
    inspector_geometry: object = None,
    inspector_visible: bool = False,
) -> None:
    save_best_creature_persistence_settings(
        settings,
        autosave_enabled=autosave_enabled,
        inject_saved_best_enabled=inject_saved_best_enabled,
        autosave_path=autosave_path,
    )
    settings.setValue(PROFILE_ID_KEY, profile_id)
    settings.setValue(CUSTOM_MAP_ID_KEY, custom_map_id)
    settings.setValue(TICKS_PER_EPOCH_KEY, ticks_per_epoch)
    settings.setValue(SEED_KEY, seed)
    settings.setValue(TESTING_GROUND_ENABLED_KEY, testing_ground_enabled)
    if main_window_geometry is None:
        settings.remove(MAIN_WINDOW_GEOMETRY_KEY)
    else:
        settings.setValue(MAIN_WINDOW_GEOMETRY_KEY, main_window_geometry)
    if main_splitter_state is None:
        settings.remove(MAIN_SPLITTER_STATE_KEY)
    else:
        settings.setValue(MAIN_SPLITTER_STATE_KEY, main_splitter_state)
    if inspector_geometry is None:
        settings.remove(INSPECTOR_GEOMETRY_KEY)
    else:
        settings.setValue(INSPECTOR_GEOMETRY_KEY, inspector_geometry)
    if inspector_visible:
        settings.setValue(INSPECTOR_VISIBLE_KEY, inspector_visible)
    else:
        settings.remove(INSPECTOR_VISIBLE_KEY)
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