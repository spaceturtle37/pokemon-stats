# save color palettes for consistency accross all the jupyternotebooks

# Color HEX values for pokemon types
# https://gist.github.com/apaleslimghost/0d25ec801ca4fc43317bcff298af43c3
# more in depth future possible variations to tinker with
# https://www.pokemonaaah.net/art/colordex/
# https://www.epidemicjohto.com/t882-type-colors-hex-colors

color_types = {
	'Normal': '#A8A77A',
	'Fire': '#EE8130',
	'Water': '#6390F0',
	'Electric': '#F7D02C',
	'Grass': '#7AC74C',
	'Ice': '#96D9D6',
	'Fighting': '#C22E28',
	'Poison': '#A33EA1',
	'Ground': '#E2BF65',
	'Flying': '#A98FF3',
	'Psychic': '#F95587',
	'Bug': '#A6B91A',
	'Rock': '#B6A136',
	'Ghost': '#735797',
	'Dragon': '#6F35FC',
	'Dark': '#705746',
	'Steel': '#B7B7CE',
	'Fairy': '#D685AD',
    'normal': '#A8A77A',
    'fire': '#EE8130',
    'water': '#6390F0',
    'electric': '#F7D02C',
    'grass': '#7AC74C',
    'ice': '#96D9D6',
    'fighting': '#C22E28',
    'poison': '#A33EA1',
    'ground': '#E2BF65',
    'flying': '#A98FF3',
    'psychic': '#F95587',
    'bug': '#A6B91A',
    'rock': '#B6A136',
    'ghost': '#735797',
    'dragon': '#6F35FC',
    'dark': '#705746',
    'steel': '#B7B7CE',
    'fairy': '#D685AD',
};


# HEX codes for pokemon stats colors
# https://bulbapedia.bulbagarden.net/wiki/Help:Color_templates

color_stats = {
    'HP': '#9EE865',
    'Attack': '#F5DE69',
    'Defense': '#F09A65',
    'Special Attack': '#66D8F6',
    'Sp. Atk': '#66D8F6',
    'Special Defense': '#899EEA',
    'Sp. Def': '#899EEA',
    'Speed': '#E46CCA',
};


# set up a color scheme for principal components
color_keys = [f'PC{i+1}' for i in range(6)]
color_hexes = [
    "#1E5631",  # deep forest green
    "#F4A261",  # warm muted orange (earthy)
    "#264653",  # dark blue-gray
    "#E63946",  # strong crimson red
    "#6A4C93",  # deep violet
    "#2A9D8F"   # teal green
]
color_pca = dict(zip(color_keys, color_hexes))


# set -up default color palette when none is given
color_keys = [i for i in range(6)]
colors_hexes = [
    '#008000',  # green
    '#800000',  # maroon
    '#006400',  # dark green
    '#00008b',  # dark blue
    '#000080',  # navy
    '#000000'   # black
]
color_default = dict(zip(color_keys, color_hexes))


# put all dictionaries into a dictionary
color_dics = {
    "types": color_types,
    "stats": color_stats,
    "pca": color_pca,
} 

