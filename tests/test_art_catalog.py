"""Keep the generated catalog complete as effects and families are registered."""

from wonhee.art.catalog import main, render_catalog
from wonhee.art.registry import REGISTRY, discover


def test_catalog_contains_every_effect_and_family():
    discover()
    text = render_catalog()
    assert f"Total: {len(REGISTRY)} effects." in text
    for effect in REGISTRY.values():
        assert f"| {effect.name} |" in text
    for family in {effect.family for effect in REGISTRY.values()}:
        assert f"\n## {family}\n" in text
        count = sum(effect.family == family for effect in REGISTRY.values())
        assert f"| {family} | {count} |" in text


def test_catalog_cli_writes_markdown(tmp_path):
    out = tmp_path / "art" / "INDEX.md"
    main(["--out", str(out)])
    assert out.read_text(encoding="utf-8") == render_catalog()
