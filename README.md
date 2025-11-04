# Pokémon Type Analysis — A Data Science Exploration 🧠🎮  

> Statistical modeling and visualization of Pokémon type matchups using Python — eigenvectors, graph theory, and PCA meet Pokémon battles.

---

## 🔍 Overview  

Inspired by analytical Pokémon YouTube channels, this project rebuilds and expands their statistical models from scratch in Python. It transforms the familiar type effectiveness table into a **stochastic matrix**, applies **eigenvector centrality** to simulate “type dominance,” and scales up to a **324×324 dual-type model** for richer interactions.  

Using **NetworkX**, the project visualizes the Pokémon type network — highlighting key connections, strongest edges, and ties using graph-theoretic measures. It then incorporates **population weighting** to rescale outcomes based on how often each type appears in the actual Pokémon dataset.  

Finally, **PCA (Principal Component Analysis)** is applied to Pokémon base stats to uncover patterns in multivariate data, showing that a small number of components explain most performance variation.  

---

## 🧩 Questions Explored  

- 🧮 *Which Pokémon type is truly the strongest?*  
- ⚔️ *How do dual typings shift the landscape of type matchups?*  
- 🌍 *What happens when you account for how common each typing actually is?*  
- 🧬 *Do Pokémon really need all six stats to define their strength?*  

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
- Population-adjusted balance metrics  
- PCA scatter plots of Pokémon stat distributions  

---

## 🎥 Youtube Inspirations

Pokémon's 19th Type According to Simple Math
https://www.youtube.com/watch?v=f4OY4qhCI04&t=7s

Pokémon's (Actual) 19th Type According to Simple Math
https://www.youtube.com/watch?v=Ov85T9xO3Wk

The Pokémon Type Advantage Network #SoME2
https://www.youtube.com/watch?v=4TevYag6P-0

Do Pokémon Really Need All 6 Stats?
https://www.youtube.com/watch?v=UhHSX5CahkU&t=513s

---

## 📚 Motivation  

Pokémon may be a game, but it’s also a great sandbox for data science — combining **combinatorics, probability, and linear algebra** into one rich dataset. This project captures that blend of **curiosity, rigor, and playfulness** that drives my broader interest in statistical modeling and computational reasoning.  

---

### 📦 Repository Structure  

```
├── data/                 # Raw and processed type matchup tables  
├── notebooks/            # Jupyter notebooks for each analysis stage  
├── visualizations/       # Exported plots and network graphs  
├── src/                  # Helper functions for matrix operations and plotting  
└── README.md             # You’re here!
```

---

### 🧑‍💻 Author  

**Juan Velasco**  
[GitHub](#) | [YouTube](#) | [LinkedIn](#) | [Email](#)
