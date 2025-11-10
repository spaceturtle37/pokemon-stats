from setuptools import setup, find_packages

setup(
    name="pokemon_stats",
    version="1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)