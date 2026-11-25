
import json
from pathlib import Path

from pokemon_stats.poke_colors import (
    color_types,
    color_stats,
    color_pca,
    color_default,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "website" / "data" / "colors.json"

# Export the original Python palettes without changing their values.
palettes = {
    "types": color_types,
    "stats": color_stats,
    "pca": color_pca,
    "default": color_default,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(palettes, file, indent=2)

print(f"Exported color palettes to {OUTPUT_FILE}")
print("Palettes:", ", ".join(palettes.keys()))
