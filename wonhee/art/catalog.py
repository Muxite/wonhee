"""Generate a browsable Markdown catalog directly from registered effect metadata."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from .registry import REGISTRY, discover


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def render_catalog() -> str:
    """Return a stable index of all effects, grouped by family and sorted by name."""
    discover()
    families = sorted({effect.family for effect in REGISTRY.values()})
    lines = [
        "# Art effect index",
        "",
        f"Total: {len(REGISTRY)} effects.",
        "",
        "Generate: `PYTHONPATH=. /home/muk/projects/wonhee/.venv/bin/python -m wonhee.art.catalog "
        "--out docs/art/INDEX.md`",
        "",
        "| Family | Count |",
        "| --- | ---: |",
    ]
    for family in families:
        count = sum(effect.family == family for effect in REGISTRY.values())
        lines.append(f"| {_cell(family)} | {count} |")
    for family in families:
        lines.extend([
            "",
            f"## {family}",
            "",
            "| Effect name | Tags | Description | Params | Inputs |",
            "| --- | --- | --- | --- | --- |",
        ])
        effects = sorted((effect for effect in REGISTRY.values() if effect.family == family), key=lambda e: e.name)
        for effect in effects:
            source = inspect.getsource(effect.fn)
            inputs = ", ".join(port for port in ("source", "data") if f"ctx.{port}" in source) or "none"
            doc = inspect.getdoc(effect.fn) or inspect.getdoc(inspect.getmodule(effect.fn)) or ""
            description = doc.splitlines()[0] if doc else ""
            params = "; ".join(
                f"{name}={default!r} — {description}"
                for name, (default, description) in effect.params.items()
            ) or "none"
            cells = (effect.name, ", ".join(effect.tags) or "none", description, params, inputs)
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/art/INDEX.md"), help="Markdown output path")
    args = parser.parse_args(argv)
    text = render_catalog()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
