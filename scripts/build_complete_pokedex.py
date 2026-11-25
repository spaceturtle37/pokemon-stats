
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATS_FILE = ROOT / "data" / "pokedex_stats.csv"
PICTURES_FILE = ROOT / "data" / "pokedex_with_pictures.csv"
OUTPUT_FILE = ROOT / "data" / "pokedex_complete.csv"

# Explicit image overrides for forms with dedicated artwork.
FORM_IMAGES = {
    "victreebel mega victreebel": "images/forms/mega-victreebel.jpg",
    "dragonite mega dragonite": "images/forms/mega-dragonite.jpg",
    "malamar mega malamar": "images/forms/mega-malamar.jpg",
    "hawlucha mega hawlucha": "images/forms/mega-hawlucha.jpg",
}


def normalize_name(name):
    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def create_pokemon_id(name):
    """Create a stable ID for a Pokémon species or form."""

    name = normalize_name(name)

    # Preserve Nidoran's gender distinction.
    name = name.replace("♀", " female ")
    name = name.replace("♂", " male ")

    if " mega " in name:
        base_name, mega_name = name.split(" mega ", 1)

        # Remove the repeated species name.
        mega_name = mega_name.replace(base_name, "", 1).strip()

        pokemon_id = f"{base_name}-mega"

        if mega_name:
            pokemon_id += f"-{mega_name}"

    else:
        pokemon_id = name

    return re.sub(
        r"[^a-z0-9]+",
        "-",
        pokemon_id,
    ).strip("-")


def main():
    stats = pd.read_csv(STATS_FILE)
    pictures = pd.read_csv(PICTURES_FILE)

    stats = stats.rename(columns={"#": "Pokedex Number"})
    pictures = pictures.rename(
        columns={
            "Index": "Pokedex Number",
            "Image": "Image File Path",
        }
    )

    stats["Pokedex Number"] = stats["Pokedex Number"].astype(int)
    pictures["Pokedex Number"] = pictures["Pokedex Number"].astype(int)

    stats["_name_key"] = stats["Name"].apply(normalize_name)
    pictures["_name_key"] = pictures["Name"].apply(normalize_name)

    keys = ["Pokedex Number", "_name_key"]

    pictures["Image File Number"] = (
        pictures["Image File Path"]
        .str.extract(r"/(\d+)\.png$")[0]
        .astype("Int64")
    )

    image_lookup = pictures[
        keys + ["Image File Number", "Image File Path"]
    ]

    complete = stats.merge(
        image_lookup,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Apply dedicated artwork to Pokémon forms.
    complete["_name_key"] = complete["Name"].apply(normalize_name)

    for name, image_path in FORM_IMAGES.items():
        complete.loc[
            complete["_name_key"] == name,
            "Image File Path",
        ] = image_path

    # Create stable IDs for each individual Pokémon form.
    complete["Pokemon ID"] = complete["Name"].apply(create_pokemon_id)

    if complete["Pokemon ID"].duplicated().any():
        duplicates = complete[
            complete["Pokemon ID"].duplicated(keep=False)
        ]
        raise ValueError(
            "Duplicate Pokemon IDs found:\n"
            + duplicates[["Pokedex Number", "Name", "Pokemon ID"]]
            .to_string(index=False)
        )

        # Identify Mega Evolution forms.
    complete["isMega"] = complete["Name"].str.contains(
        " Mega ",
        regex=False,
        case=False,
    )

    # Check that every image path points to an existing file.
    complete["_image_exists"] = complete["Image File Path"].apply(
        lambda path: (ROOT / path).is_file()
        if pd.notna(path)
        else False
    )

    missing = complete[~complete["_image_exists"]]

    print("Pokémon records:", len(complete))
    print("Unique Pokemon IDs:", complete["Pokemon ID"].nunique())
    print("Missing image paths:", complete["Image File Path"].isna().sum())
    print("Missing physical image files:", len(missing))

    if not missing.empty:
        print("\nRecords with missing images:")
        print(
            missing[["Pokedex Number", "Name", "Image File Path"]]
            .to_string(index=False)
        )

    complete = complete.drop(
        columns=["_name_key", "_image_exists"]
    )

    complete.to_csv(OUTPUT_FILE, index=False)

    print(f"\nExported: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
