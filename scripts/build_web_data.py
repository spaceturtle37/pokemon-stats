
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "pokedex_complete.csv"
OUTPUT_FILE = ROOT / "website" / "data" / "pokemon.json"

STAT_COLUMNS = [
    "Total",
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed",
]


def convert_pokemon(row):
    """Convert one consolidated CSV record into a website record."""

    name = row["Name"].strip()
    types = row["Type"].split()
    
    # Canonical image path, relative to the project root.
    image_path = row["Image File Path"].strip()

    if not image_path:
        raise ValueError(f"Missing image path for {name}")

    image_file = ROOT / image_path

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Image not found for {name}: {image_path}"
        )

    # Browser path, relative to website/index.html.
    browser_image_path = f"../{Path(image_path).as_posix()}"

    pokedex_number = int(row["Pokedex Number"])

    image_number = (
        int(float(row["Image File Number"]))
        if row["Image File Number"].strip()
        else None
    )

    pokemon = {
        # Explicit identifiers
        "Pokemon ID": row["Pokemon ID"],
        "Pokedex Number": pokedex_number,
        "Image File Number": image_number,
        "Image File Path": image_path,

        # Display information
        "Name": name,
        "Type 1": types[0] if types else "",
        "Type 2": types[1] if len(types) > 1 else "",
        "Number of Types": 2 if len(types) > 1 and types[1] else 1,

        # Browser-ready image path
        "Image": browser_image_path,

        # Backward compatibility with existing frontend code
        "Number": pokedex_number,
    }

    for stat in STAT_COLUMNS:
        pokemon[stat] = int(row[stat])

    pokemon["isMega"] = "mega" in name.casefold().split()

    pokemon["isAlternate"] = any(
        marker in name
        for marker in [
            "Alolan",
            "Galarian",
            "Hisuian",
            "Paldean",
            "Partner",
            "Breed",
        ]
    )

    return pokemon


def main():
    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        pokemon_data = [
            convert_pokemon(row)
            for row in reader
        ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pokemon_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Exported {len(pokemon_data)} Pokémon entries.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
