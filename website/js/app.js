
// =========================================
// POKEMON STATS — STAT LAB
// =========================================

let pokemonData = [];
let palettes = {};

const STAT_COLUMNS = [
    "Total",
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed"
];

const DEFAULT_FILTERS = {
    type: "all",
    minTotal: 0,
    search: ""
};


// =========================================
// LOAD DATA
// =========================================

async function loadDashboardData() {
    try {
        const [pokemonResponse, colorsResponse] = await Promise.all([
            fetch("data/pokemon.json"),
            fetch("data/colors.json")
        ]);

        if (!pokemonResponse.ok || !colorsResponse.ok) {
            throw new Error("Could not load JSON files.");
        }

        [pokemonData, palettes] = await Promise.all([
            pokemonResponse.json(),
            colorsResponse.json()
        ]);
        document.getElementById("chart").textContent = "";

        console.log(`Loaded ${pokemonData.length} Pokémon.`);
        console.log("Loaded color palettes:", palettes);

        initializeDashboard();

    } catch (error) {
        console.error("Dashboard loading failed:", error);

        document.getElementById("chart").textContent =
            "Could not load Pokémon data. Check the browser console.";
    }
}


// =========================================
// INITIALIZATION
// =========================================

function initializeDashboard() {

    populateSelect("x-axis", STAT_COLUMNS);
    populateSelect("y-axis", STAT_COLUMNS);

    document.getElementById("x-axis").value = "Attack";
    document.getElementById("y-axis").value = "Speed";

    const types = Object.keys(palettes.types)
        .filter(type => type[0] === type[0].toUpperCase())
        .sort();

    // Primary type
    populateSelect("type-filter", types, true);
    document.getElementById("type-filter").value = "all";

    // Secondary type
    populateSelect("secondary-type-filter", types);
    document.getElementById("secondary-type-filter").add(
        new Option("Any Secondary Type", "all"),
        0
    );

    document.getElementById("type-filter").value = "all";
    document.getElementById("secondary-type-filter").value = "all";
    document.getElementById("type-count-filter").value = "all";

    document.getElementById("type-filter").add(
        new Option("All Types", "all"),
        0
    );

    document.getElementById("type-filter").value = "all";
    document.getElementById("secondary-type-filter")
        .addEventListener("change", updateChart);

    document.getElementById("type-count-filter")
        .addEventListener("change", updateChart);

    document.getElementById("min-total").addEventListener(
        "input",
        event => {
            document.getElementById("min-total-value").textContent =
                event.target.value;

            updateChart();
        }
    );

    document.getElementById("search").addEventListener(
        "input",
        updateChart
    );

    document.getElementById("type-filter").addEventListener(
        "change",
        updateChart
    );

    document.getElementById("x-axis").addEventListener(
        "change",
        updateChart
    );

    document.getElementById("y-axis").addEventListener(
        "change",
        updateChart
    );

    document.getElementById("chart-mode").addEventListener(
        "change",
        updateChart
    );

    document.getElementById("reset-button").addEventListener(
        "click",
        resetFilters
    );

    updateChart();
}


// =========================================
// SELECT HELPER
// =========================================

function populateSelect(elementId, options, preserveDefault = false) {

    const select = document.getElementById(elementId);

    if (!preserveDefault) {
        select.innerHTML = "";
    }

    options.forEach(option => {
        select.add(new Option(option, option));
    });
}


// =========================================
// FILTER DATA
// =========================================

function getFilteredPokemon() {

    const selectedType =
        document.getElementById("type-filter").value;

    const selectedSecondaryType =
        document.getElementById("secondary-type-filter").value;

    const selectedTypeCount =
        document.getElementById("type-count-filter").value;

    const minTotal = Number(
        document.getElementById("min-total").value
    );

    const search = document.getElementById("search")
        .value.trim().toLowerCase();

    return pokemonData.filter(pokemon => {

        const matchesPrimaryType =
            selectedType === "all" ||
            pokemon["Type 1"] === selectedType;

        const matchesSecondaryType =
            selectedSecondaryType === "all" ||
            pokemon["Type 2"] === selectedSecondaryType;

        const matchesTypeCount =
            selectedTypeCount === "all" ||
            String(pokemon["Number of Types"]) === selectedTypeCount;

        const matchesTotal =
            pokemon.Total >= minTotal;

        const matchesSearch =
            pokemon.Name.toLowerCase().includes(search);

        return (
            matchesPrimaryType &&
            matchesSecondaryType &&
            matchesTypeCount &&
            matchesTotal &&
            matchesSearch
        );
    });
}


// =========================================
// UPDATE DASHBOARD
// =========================================

function updateChart() {

    const data = getFilteredPokemon();

    updateMetrics(data);

    const mode = document.getElementById("chart-mode").value;

    if (mode === "histogram") {
        drawHistogram(data);
    } else {
        drawScatter(data);
    }
}


// =========================================
// SUMMARY METRICS
// =========================================

function updateMetrics(data) {

    document.getElementById("count-visible").textContent =
        data.length.toLocaleString();

    document.getElementById("dataset-count").textContent =
        pokemonData.length.toLocaleString();

    if (data.length === 0) {
        document.getElementById("average-total").textContent = "—";
        document.getElementById("top-stat").textContent = "—";
        return;
    }

    const average = data.reduce(
        (sum, pokemon) => sum + pokemon.Total,
        0
    ) / data.length;

    document.getElementById("average-total").textContent =
        average.toFixed(1);

    const averages = STAT_COLUMNS.slice(1).map(stat => ({
        stat,
        value: data.reduce(
            (sum, pokemon) => sum + pokemon[stat],
            0
        ) / data.length
    }));

    averages.sort((a, b) => b.value - a.value);

    document.getElementById("top-stat").textContent =
        averages[0].stat;
}


// =========================================
// SCATTER PLOT
// =========================================

function drawScatter(data) {

    const xStat = document.getElementById("x-axis").value;
    const yStat = document.getElementById("y-axis").value;

    const typeGroups = {};

    data.forEach(pokemon => {

        const type = pokemon["Type 1"];

        if (!typeGroups[type]) {
            typeGroups[type] = [];
        }

        typeGroups[type].push(pokemon);
    });

    const traces = Object.entries(typeGroups).map(
        ([type, pokemon]) => ({

            x: pokemon.map(p => p[xStat]),
            y: pokemon.map(p => p[yStat]),

            text: pokemon.map(p => p.Name),

            customdata: pokemon.map(p => [
                p["Type 1"],
                p["Type 2"],
                p.Total,
                p.HP,
                p.Attack,
                p.Defense,
                p["Sp. Atk"],
                p["Sp. Def"],
                p.Speed
            ]),

            mode: "markers",
            type: "scatter",
            name: type,

            marker: {
                color: palettes.types[type] || "#999999",
                size: 9,
                opacity: 0.8,
                line: {
                    color: "#ffffff55",
                    width: 0.5
                }
            },

            hovertemplate:
                "<b>%{text}</b><br>" +
                "Type: %{customdata[0]} %{customdata[1]}<br>" +
                "Total: %{customdata[2]}<br>" +
                "HP: %{customdata[3]}<br>" +
                "Attack: %{customdata[4]}<br>" +
                "Defense: %{customdata[5]}<br>" +
                "Sp. Atk: %{customdata[6]}<br>" +
                "Sp. Def: %{customdata[7]}<br>" +
                "Speed: %{customdata[8]}" +
                "<extra></extra>"
        })
    );

    const layout = {

        autosize: true,

        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",

        font: {
            color: "#c5d0de",
            family: "Manrope, sans-serif"
        },

        margin: {
            l: 55,
            r: 25,
            t: 20,
            b: 55
        },

        xaxis: {
            title: xStat,
            gridcolor: "#29394f",
            zerolinecolor: "#29394f"
        },

        yaxis: {
            title: yStat,
            gridcolor: "#29394f",
            zerolinecolor: "#29394f"
        },

        legend: {
            orientation: "h",
            y: -0.2
        },

        hoverlabel: {
            bgcolor: "#152338",
            bordercolor: "#65d6d1",
            font: {
                color: "#ffffff"
            }
        }
    };

    Plotly.react("chart", traces, layout, {
        responsive: true,
        displaylogo: false
    });
}


// =========================================
// HISTOGRAM
// =========================================

function drawHistogram(data) {

    const stat = document.getElementById("x-axis").value;

    const trace = {
        x: data.map(p => p[stat]),
        type: "histogram",

        marker: {
            color: "#65d6d1",
            line: {
                color: "#152338",
                width: 1
            }
        },

        hovertemplate:
            stat + ": %{x}<br>" +
            "Count: %{y}<extra></extra>"
    };

    const layout = {

        autosize: true,

        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",

        font: {
            color: "#c5d0de"
        },

        xaxis: {
            title: stat,
            gridcolor: "#29394f"
        },

        yaxis: {
            title: "Number of Pokémon",
            gridcolor: "#29394f"
        },

        margin: {
            l: 55,
            r: 25,
            t: 20,
            b: 55
        },

        bargap: 0.08
    };

    Plotly.react("chart", [trace], layout, {
        responsive: true,
        displaylogo: false
    });
}


// =========================================
// RESET FILTERS
// =========================================

function resetFilters() {

    document.getElementById("type-filter").value = "all";
    document.getElementById("secondary-type-filter").value = "all";
    document.getElementById("type-count-filter").value = "all"; 
    document.getElementById("min-total").value = 0;
    document.getElementById("min-total-value").textContent = "0";
    document.getElementById("search").value = "";

    document.getElementById("x-axis").value = "Attack";
    document.getElementById("y-axis").value = "Speed";
    document.getElementById("chart-mode").value = "scatter";

    updateChart();
}


// =========================================
// START
// =========================================

loadDashboardData();
