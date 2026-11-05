"""Block Jacobian tag parsing and enumeration.

Tag form: `{inputs}-{outputs}` with level names in canonical order.
Example: `Z1-Z1,Z2` differentiates mid only and keeps mid+low outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
from typing import Iterable, Sequence


LEVELS_L3: tuple[str, ...] = ("U", "Z1", "Z2")


def levels_plus(num_coarse: int) -> tuple[str, ...]:
    if num_coarse < 1:
        raise ValueError(f"num_coarse must be >= 1, got {num_coarse}")
    return ("U",) + tuple(f"Z{i + 1}" for i in range(num_coarse))


def _nonempty_subsets(levels: Sequence[str]) -> list[tuple[str, ...]]:
    items = list(levels)
    return list(
        chain.from_iterable(
            combinations(items, r) for r in range(1, len(items) + 1)
        )
    )


@dataclass(frozen=True)
class BlockSpec:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def tag(self) -> str:
        return f"{','.join(self.inputs)}-{','.join(self.outputs)}"

    @property
    def is_square(self) -> bool:
        return set(self.inputs) == set(self.outputs)

    @property
    def spectrum_kind(self) -> str:
        return "eigenvalue" if self.is_square else "singular_value"


def canonicalize_names(
    names: Iterable[str], allowed: Sequence[str]
) -> tuple[str, ...]:
    allowed_set = set(allowed)
    order = {name: i for i, name in enumerate(allowed)}
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name not in allowed_set:
            raise ValueError(
                f"Unknown level {name!r}; allowed={list(allowed)}"
            )
        if name in seen:
            raise ValueError(f"Duplicate level {name!r} in block side")
        seen.add(name)
        out.append(name)
    if not out:
        raise ValueError("Block side must list at least one level")
    out.sort(key=lambda n: order[n])
    return tuple(out)


def parse_block_tag(tag: str, allowed: Sequence[str]) -> BlockSpec:
    text = tag.strip()
    if text.count("-") != 1:
        raise ValueError(
            f"Block tag must be inputs-outputs with one dash, got {tag!r}"
        )
    left, right = text.split("-", 1)
    inputs = canonicalize_names(left.split(","), allowed)
    outputs = canonicalize_names(right.split(","), allowed)
    return BlockSpec(inputs=inputs, outputs=outputs)


def all_blocks(allowed: Sequence[str]) -> list[BlockSpec]:
    sides = _nonempty_subsets(allowed)
    return [
        BlockSpec(inputs=ins, outputs=outs)
        for ins in sides
        for outs in sides
    ]


def resolve_blocks(
    requested: Sequence[str] | None, allowed: Sequence[str]
) -> list[BlockSpec]:
    if not requested:
        return all_blocks(allowed)
    seen: set[str] = set()
    out: list[BlockSpec] = []
    for raw in requested:
        spec = parse_block_tag(raw, allowed)
        if spec.tag in seen:
            continue
        seen.add(spec.tag)
        out.append(spec)
    return out


def tag_to_dirname(tag: str) -> str:
    """Filesystem-safe directory name for a block tag."""
    return tag.replace(",", "_")
