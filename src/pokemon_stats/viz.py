# gaussian function
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

import pokemon_stats.poke_colors as pcol
import importlib
importlib.reload(pcol)

# import plotly.express as px
# import plotly.graph_objects as go
# from ipywidgets import interact
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output
# import socket


def gaussian(x: float | np.ndarray, mu: float, sigma: float, A:float ) -> float | np.ndarray:
    """
    Calculates the value of a Gaussian function at a given point or points x.

    The formula used is: A * exp(-(x - mu)**2 / (2 * sigma**2))

    Args:
        x: The input value(s) (float or numpy array) where the Gaussian is evaluated.
        mu: The mean (center) of the Gaussian distribution.
        sigma: The standard deviation (width) of the Gaussian distribution.
        A: The amplitude (height) of the Gaussian function.

    Returns:
        The calculated value(s) of the Gaussian function (float or numpy array).
    """
    return A * norm.pdf(x, loc=mu, scale=sigma)

def hist_stats(df: pd.DataFrame, color_by=None) -> None:
    """
    generate a subtplot for histograms with mean and std dev plus gaussian fit for each column of df

    Args:
        df: data frame to plot

    Returns:
        None
    """
    
    # print(df.head())
    nbins = 20
    bin_width = 5
    tick_spacing = 50
    scale = 1.1
    nfit = 100

    # calculate min and max
    stat_min = df.min()
    stat_max = df.max()
    all_min = stat_min.min()
    all_max = stat_max.max()
    xrange = np.linspace(all_min, all_max, nfit)
    
    ticks = np.arange(0, all_max*scale, tick_spacing)
    bin_borders = np.arange(0, all_max + bin_width, bin_width) 
    bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2
    
    fig, axes = plt.subplots(3,2,figsize=(10, 8), sharex=True, sharey=True, tight_layout=True)
    
    for j,(col, x) in enumerate(df.items()):
        # print(j, col, x)
        # set the ax
        ax = axes.flat[j]
        # print(x)
        x_mean = x.mean()
        x_std = x.std()
        # print(x_mean, x_std)
        
        # do the histogram counts
        hist_counts, bin_borders = np.histogram(x, bins=nbins)# bins=bin_borders)
        bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2
        # print('counts: ', hist_counts)
        # print('bin borders: ', bin_borders)
        # print('bin centers: ', bin_centers)
        amplitude = max(hist_counts)
        # print('amplitude:', amplitude)
        
        # initial guess for parameters
        p0 = [x_mean, x_std, amplitude]
        
        # get the parameter values and errors 
        popt, pcov = curve_fit(gaussian, bin_centers, hist_counts, p0 = p0)
        fit_mean, fit_std, fit_amplitude = popt
        fit_counts = gaussian(bin_centers, *popt)
        # print(popt)
        # print(pcov)
    
        #plot the fit results for bins
        # ax.scatter(bin_centers, fit_counts, alpha=0.5)
        ax.plot(xrange, gaussian(xrange, *popt) , alpha=0.25, color='black', linewidth = 2, linestyle='--', label='Binned Gaussian Fit')
        
        # x_min = stat_min[col]
        # x_max = stat_max[col]
        # print(x_min, x_max)
        # bins = np.arange(x_min, x_max + bin_width, bin_width) 

        # print(color_by)
        if color_by == 'stats':
            color = pcol.color_stats[col]   
        elif color_by == 'pca':
            color = pcol.color_pca[col]
        else:
            color = 'grey'
        # print('color')
        ax.hist(x, bins=bin_borders, alpha=0.5, color=color)# , density=True , edgecolor='black')
    
        # Add a vertical line for the mean
        ax.axvline(fit_mean, color='red', linestyle='dashed', linewidth=2, label=f' Binned Mean: {fit_mean:.2f}') # + "\n" + f' Mean: {x_mean:.0f}' )
    
        # Add vertical lines for +/- 1 standard deviation
        ax.axvline(fit_mean - fit_std, color='green', linestyle='dotted', linewidth=2, label=f' Binned Std: {fit_std:.2f}')
        ax.axvline(fit_mean + fit_std, color='green', linestyle='dotted', linewidth=2)
        
        ax.set_title(col)
        # ax.set_xlim(0, all_max*scale)
        # ax.set_xticks(ticks)
    
        plt.sca(ax)
        plt.legend()
    
    fig.supxlabel('Stat')
    fig.supylabel('Count')
    plt.show()

    
def scatter_stats(df: pd.DataFrame, color_by='stats') -> None:
    """
    create a single plot showing scatters with mean and std dev for each column of df

    Args:
        df: data frame to plot

    Returns:
        None
    """
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(df.columns))
    
    for i, (col, values) in enumerate(df.items()):
    
        # Mean and std for this column
        mean = values.mean()
        std = values.std()
        ax.errorbar(i, mean, yerr=std, fmt='o', color='black', capsize=4, lw=1.5, alpha=0.5)
        
        # Add jitter in both directions
        x = np.full(len(values), i) + .5 * np.random.uniform(-1, 1, size=len(values))
        y = values # + np.random.uniform(-0.05, 0.05, size=len(values))  
        
        # print(color_by)
        if color_by == 'stats':
            color = pcol.color_stats[col]   
        elif color_by == 'pca':
            color = pcol.color_pca[col]
        else:
            color = 'grey'
        # print(color)
        ax.scatter(x, y, alpha=0.1 , color=color)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df.columns) # , rotation=90, ha='right')  # rotate labels here
    ax.set_title("Stats with mean ± std (with jitter)")
    ax.set_ylabel("Stat Value")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def heatmap(df: pd.DataFrame) -> None:
    """
    create a heatmap 

    Args:
        df: data frame to plot

    Returns:
        None
    """

    plt.figure(figsize=(10,6))
    sns.heatmap(df, annot=True, cmap='RdBu_r', center=0)
    plt.title("PCA Expressed in Pokemon Stats Weights Heatmap")
    plt.xlabel("Pokemon Stats Weights")
    plt.ylabel("PCA Vectors")
    
    # plt.xticks(rotation=45)  # optional: rotate for readability
    plt.gca().xaxis.set_label_position('top')  # move xlabel to top
    plt.gca().xaxis.tick_top()  # move ticks to top

    # plt.show()


def barcharts(df, sorted=False) -> None:
    fig, axes = plt.subplots(3,2,figsize=(10, 8), sharey=True, tight_layout=True)
    
    for i,(pca_component, row) in enumerate(df.iterrows()):
        ax = axes.flat[i]
        plt.sca(ax)
        if sorted:
            row = row.sort_values()
        plt.bar(row.index, row.values, color=[pcol.color_stats[feature] for feature in row.index])
        plt.title(f'PC{i+1} Expansion in terms of Original Stats')
        # ax.tick_params(axis='x', labelbottom=True)
    
    fig.supxlabel('Pokemon Stat')
    fig.supylabel('Weight')
    plt.show()
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Chart 1", "Chart 2"))
    """


    