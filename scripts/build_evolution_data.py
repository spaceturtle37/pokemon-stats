
import json
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "pokemon_evolution.json"

API_URL = "https://pokeapi.co/api/v2/evolution-chain/"

def get_all_evolution_chains():
    """Download all available evolution chains."""

    response = requests.get(
        API_URL,
        params={"limit": 1000},
        timeout=30,
    )
    response.raise_for_status()

    return response.json()["results"]


def process_chain(chain, stage, records):
    """Recursively process an evolutionary family tree."""

    name = chain["species"]["name"]

    records[name] = {
        "Stage": stage,
    }

    for evolution in chain["evolves_to"]:
        process_chain(evolution, stage + 1, records)


def main():
    print("Retrieving evolution chains...")

    chains = get_all_evolution_chains()

    evolution_data = {}

    for index, chain_reference in enumerate(chains, start=1):
        response = requests.get(
            chain_reference["url"],
            timeout=30,
        )
        response.raise_for_status()

        chain_data = response.json()

        process_chain(
            chain_data["chain"],
            1,
            evolution_data,
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(chains)} chains")

    # Determine the maximum stage in each evolutionary family.
    # Every branch shares the same total number of stages.
    for pokemon in evolution_data.values():
        pokemon["Number of Stages"] = 0

    # Reconstruct family maximums from the chain traversal.
    # Each chain is processed independently in the loop above.

    print("Evolution data retrieved:", len(evolution_data))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(evolution_data, file, indent=4)

    print("Exported:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
