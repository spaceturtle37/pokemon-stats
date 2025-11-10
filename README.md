<p align="center">
  <img src="banner.png" alt="Pokémon Type Analysis Banner" width="800">
</p>

# Pokemon Type Analysis — A Data Science Exploration 🧠🎮  

> Statistical modeling and visualization of Pokemon type matchups using Python — eigenvectors, graph theory, and PCA meet Pokemon battles.

---

## 🔍 Overview  

Inspired by analytical Pokemon YouTube channels, this project rebuilds and expands their statistical models from scratch in Python. It transforms the familiar 18x18 type effectiveness table into a **stochastic matrix**, applies **eigenvector centrality** to simulate “type dominance,” and scales up to a **324×324 dual-type model** for richer interactions.  

Using **NetworkX**, the project visualizes the Pokemon type network — highlighting key connections, strongest edges, and ties using graph-theoretic measures. It then incorporates **population weighting** to rescale outcomes based on how often each type appears in the actual Pokemon dataset.  

Finally, **PCA (Principal Component Analysis)** is applied to Pokemon base stats to uncover patterns in multivariate data, showing that a small number of components explain most performance variation.  

---

## 🧩 Questions Explored  

- 🧮 *Which Pokemon type is truly the strongest?*  
- ⚔️ *How do dual typings shift the landscape of type matchups?*  
- 🌍 *What happens when you account for how common each typing actually is?*  
- 🧬 *Do Pokemon really need all six stats to define their strength?*  

---

## 🧰 Tools & Libraries  

- **Python**, **NumPy**, **pandas** — data processing & matrix modeling  
- **Matplotlib**, **Seaborn**, **NetworkX** — data visualization & network graphs  
- **scikit-learn** — PCA and dimensionality reduction  
- **JupyterLab** — experimentation & reproducibility  

---

## 📈 Example Outputs  

- Type advantage network visualizations  
- Eigenvector-based rankings for single and dual types   
- PCA scatter plots of Pokemon stat distributions  

---

## 🎥 Youtube Inspirations

Pokemon's 19th Type According to Simple Math
https://www.youtube.com/watch?v=f4OY4qhCI04&t=7s

Pokemon's (Actual) 19th Type According to Simple Math
https://www.youtube.com/watch?v=Ov85T9xO3Wk

The Pokemon Type Advantage Network #SoME2
https://www.youtube.com/watch?v=4TevYag6P-0

Do Pokemon Really Need All 6 Stats?
https://www.youtube.com/watch?v=UhHSX5CahkU&t=513s

---

## Pokemon Databases


Pokemon Stats All Gens
https://pokemondb.net/pokedex/all

Pokemon Single Type Advantage Chart
https://pokemondb.net/type

Pokemon Dual Type Advantage Chart
https://pokemondb.net/type/dual

---

## 📚 Motivation  

Pokemon may be a game, but it’s also a great sandbox for data science — combining **combinatorics, probability, and linear algebra** into one rich dataset. This project captures that blend of **curiosity, rigor, and playfulness** that drives my broader interest in statistical modeling and computational reasoning.  

---

### 📦 Repository Structure  

```
├── Input_data/           # Raw and processed type matchup tables  
├── Notebooks/       	  # Jupyter notebooks for each analysis stage  
├── Output_data/       	  # Exported plots and network graphs  
├── src/                  # Helper functions for matrix operations and plotting  
└── README.md             # You’re here!
```

---

### 🧑‍💻 Author  

**Juan Velasco**  
[GitHub](https://github.com/spaceturtle37) | [LinkedIn](#) | [Email](#)
