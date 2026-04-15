from __future__ import annotations

import json
from pathlib import Path

from app.models import ActionType, CellType, Gene, GenePattern, Genome


LEGACY_ACTION_RENAMES = {
    'EAT': 'IDLE',
}


def save_genome_to_file(genome: Genome, file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_genome_to_dict(genome), indent=2), encoding='utf-8')


def load_genome_from_file(file_path: str) -> Genome:
    path = Path(file_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    return _genome_from_dict(data)


def genome_file_exists(file_path: str) -> bool:
    return Path(file_path).is_file()


def _genome_to_dict(genome: Genome) -> dict:
    return {
        'learning_rate': genome.learning_rate,
        'reward_decay': genome.reward_decay,
        'exploration_rate': genome.exploration_rate,
        'history_length': genome.history_length,
        'genes': [_gene_to_dict(gene) for gene in genome.genes],
    }


def _gene_to_dict(gene: Gene) -> dict:
    return {
        'gene_id': gene.gene_id,
        'action': gene.action.name,
        'base_priority': gene.base_priority,
        'pattern': {
            'current_cell': _enum_name(gene.pattern.current_cell),
            'north_cell': _enum_name(gene.pattern.north_cell),
            'east_cell': _enum_name(gene.pattern.east_cell),
            'south_cell': _enum_name(gene.pattern.south_cell),
            'west_cell': _enum_name(gene.pattern.west_cell),
            'last_action': _enum_name(gene.pattern.last_action),
            'last_action_success': gene.pattern.last_action_success,
            'hunger_bucket': gene.pattern.hunger_bucket,
        },
    }


def _genome_from_dict(data: dict) -> Genome:
    return Genome(
        genes=[_gene_from_dict(gene_data) for gene_data in data['genes']],
        learning_rate=float(data['learning_rate']),
        reward_decay=float(data['reward_decay']),
        exploration_rate=float(data['exploration_rate']),
        history_length=int(data['history_length']),
    )


def _gene_from_dict(data: dict) -> Gene:
    pattern = data['pattern']
    legacy_direction_fields = {'front_cell', 'left_cell', 'right_cell', 'back_cell'}
    if legacy_direction_fields & set(pattern):
        raise ValueError('Saved genomes using facing-relative sensor fields are no longer supported.')
    return Gene(
        gene_id=int(data['gene_id']),
        action=_action_type(data['action']),
        base_priority=float(data['base_priority']),
        pattern=GenePattern(
            current_cell=_cell_type_or_none(pattern['current_cell']),
            north_cell=_cell_type_or_none(pattern['north_cell']),
            east_cell=_cell_type_or_none(pattern['east_cell']),
            south_cell=_cell_type_or_none(pattern['south_cell']),
            west_cell=_cell_type_or_none(pattern['west_cell']),
            last_action=_action_type_or_none(pattern['last_action']),
            last_action_success=pattern['last_action_success'],
            hunger_bucket=pattern['hunger_bucket'],
        ),
    )


def _enum_name(value: CellType | ActionType | None) -> str | None:
    if value is None:
        return None
    return value.name


def _cell_type_or_none(value: str | None) -> CellType | None:
    if value is None:
        return None
    return CellType[value]


def _action_type_or_none(value: str | None) -> ActionType | None:
    if value is None:
        return None
    return _action_type(value)


def _action_type(value: str) -> ActionType:
    mapped_value = LEGACY_ACTION_RENAMES.get(value, value)
    return ActionType[mapped_value]