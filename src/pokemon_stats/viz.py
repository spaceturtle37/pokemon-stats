# gaussian function
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import pokemon_stats.poke_colors as pcol
import importlib
importlib.reload(pcol)

# import plotly.express as px
# import plotly.graph_objects as go
# from ipywidgets import interact
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output
# import socket


def plot_dataframe(df, plot_func, figsize=(12, 4)):
    cols = df.columns
    n = len(cols)
    nrows = math.ceil(n / 2)
    
    fig, axes = plt.subplots(nrows, 2, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten()  # flatten for easy indexing
    
    for i, col in enumerate(cols):
        ax = axes[i]
        plot_func(ax, df[col], col)  # call your provided function
    
    # Turn off leftover empty subplots if odd number of columns
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


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


def hist_stats(df: pd.DataFrame, color_by='types') -> None:
    """
    generate a subtplot for histograms with mean and std dev plus gaussian fit for each column of df

    Args:
        df: data frame to plot

    Returns:
        None
    """
    color_dic = pcol.color_dics[color_by]
    
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
    
    for j,col in enumerate(df.columns):
        values = df[col]
        # set the ax
        ax = axes.flat[j]
        # print(x)
        v_mean = values.mean()
        v_std = values.std()
        # print(x_mean, x_std)
        
        # do the histogram counts
        hist_counts, bin_borders = np.histogram(values, bins=nbins)# bins=bin_borders)
        bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2
        # print('counts: ', hist_counts)
        # print('bin borders: ', bin_borders)
        # print('bin centers: ', bin_centers)
        amplitude = max(hist_counts)
        # print('amplitude:', amplitude)
        
        # initial guess for parameters
        p0 = [v_mean, v_std, amplitude]
        
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

        ax.hist(x, bins=bin_borders, alpha=0.5, color=color_dic[col])# , density=True , edgecolor='black')
    
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

    
def scatter_stats(df: pd.DataFrame, title, ylabel, alpha=1, color_by='types', sort=False, top=50) -> None:
    """
    create a single plot showing scatters with mean and std dev for each column of df

    Args:
        df: data frame to plot

    Returns:
        None
    """
    offset = 0.25
    xnoise = 0.4
    ynoise = 0.1
    
    color_dic = pcol.color_dics[color_by]

    cols = df.columns
    if sort:
        # Compute column means and get the sorted order
        col_means = df.mean(axis=0)
        cols = col_means.sort_values().index  # ascending order
        # print(cols)
        n = len(cols)
        # slice top cols
        if n<top:
            top = n
        cols = cols[-top:]
        
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(df.columns))
    
    for j, col in enumerate(cols):
        values = df[col]
        nvals = len(values)
    
        # Mean and std for this column
        mean = values.mean()
        std = values.std()
        ax.errorbar(j+offset, mean, yerr=std, fmt='o', color='black', capsize=4, lw=1.5, alpha=0.5)
        
        # Add jitter in both directions
        x = np.full(nvals, j) + xnoise * np.random.uniform(-1, 1, size=nvals)
        y = values + ynoise * np.random.uniform(-1, 1, size=nvals)  

        features = ensure_tuple(col)
        # print(features)
        nfeatures = len(features)
        # print(nfeatures)
        if nfeatures == 1:
            new_features = features[0]
        else:
            randints = np.random.randint(0, nfeatures-1)
            new_features = [features[i] for i in randints]
            
        ax.scatter(x, y, alpha=alpha , color=color_dic[new_features])
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(cols, rotation=90, ha='right')  # rotate labels here
    ax.set_title(f"{title} (with jitter) and mean ± std")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def heatmap(df: pd.DataFrame ,xlabel, ylabel) -> None:
    """
    create a heatmap 

    Args:
        df: data frame to plot

    Returns:
        None
    """

    plt.figure(figsize=(10,8))
    sns.heatmap(df, annot=True, cmap='viridis', cbar=True, linewidths=.5, center=0)
    # plt.title("PCA Expressed in Pokemon Stats Weights Heatmap")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.gca().xaxis.set_label_position('top')  # move xlabel to top
    plt.gca().xaxis.tick_top()  # move ticks to top
    plt.xticks(rotation=90)  # optional: rotate for readability
        
    plt.show()

def find_min_difference_factors(a):
    """
    Finds two factors b and c of a such that abs(b-c) is minimized.

    Args:
        a: An integer.

    Returns:
        A tuple (b, c) of the factors.
        b <= c
    """
    if a <= 0:
        return None  # Or handle as appropriate for your use case

    b = 1
    c = a
    # Loop up to the square root of a
    for i in range(1, int(np.sqrt(a)) + 1):
        if a % i == 0:
            # If i is a factor, then a // i is the other factor
            # The last pair found will have the minimum difference
            b = i
            c = a // i
    return (b, c)


"""
def subplots(plot_func, df, color_by='types', title='', top=50, nstripes=1):

    if isinstance(df, pd.Series):
    # print("The variable is a Pandas Series.")
        df = df.to_frame()
    # print(df.shape)
    
    # n, m = find_min_difference_factors(len(df.columns))
   
    cols = df.columns
    ncols = len(cols)
    # print(ncols)
    fig, axes = plt.subplots(ncols,1, figsize=(10,6*ncols), sharey=True, tight_layout=True)
    print(axes)
    if ncols==1:
        flax = [axes]
    else:
        flax = axes.flatten()
    for j, col in enumerate(cols):
        plt.sca(flax[j])
        s = df[col]
    
    # fig.supxlabel('Pokemon Stat')
    fig.supylabel('Weight')
    plt.show()
"""
    
    

def barcharts(s, sort=False, color_by='types', title='', top=50, nstripes=1) -> None:
    """
    if isinstance(s, pd.DataFrame):
        print("this is a dataframe not a single dataseries")
        print(s.columns)
        for col in s.columns: 
            print(col)
            print(s[col])
            barcharts(s[col], sort, color_by, title, top, nstripes)
    """
    color_dic = pcol.color_dics[color_by]
    nvals = len(s)
    
    if sort:
        s = s.sort_values()
        # slice top cols
        if nvals < top:
            top = nvals
        s = s[-top:]
    
    # print(features)
    # ensure tuple for single items 
    features = [ensure_tuple(f) for f in s.index]
    # print(features)
    lenfeatures = [len(f) for f in features]
    # print(lenfeatures)
    nfeatures = max(lenfeatures)
    # print(nfeatures)
    # no need to do loop if only single type pokemon
    if nfeatures==1:
        nstripes = 1
    step = s/nstripes
    for i in range(nstripes):
        plt.bar(s.index, step, bottom = i*step, color=[color_dic[f[i%nfeatures]] for f in features])
        
    plt.title(f'{s.name} ' + title)
   
    # ax.tick_params(axis='x', labelbottom=True)
    # plt.xticks(x, features, rotation=90)
    plt.xticks(rotation=90)
    # plt.legend()
  

def barcharts_re_im(df, sort=False, color_by='types', title='') -> None:

    color_dic = pcol.color_dics[color_by]
    
    # n, m = find_min_difference_factors(len(df.columns))
    m = len(df.index)
    n = 1
    fig, axes = plt.subplots(m,n, figsize=(10,6*m), sharey=True, tight_layout=True)

    for i,(pca_component, row) in enumerate(df.iterrows()):
        ax = axes.flat[i]
        plt.sca(ax)
        if sort:
            row = row.sort_values()
        # Positions for bars
        features = row.index
        x = np.arange(len(features))
        bar_width = 0.4
        
        # Plot real and imaginary parts
        plt.bar(x - bar_width/2, np.abs(np.real(row)), width=bar_width,
                color=[color_dic[f] for f in features], label='Real (positive)')
        plt.bar(x + bar_width/2, -np.abs(np.imag(row)), width=bar_width,
                color=[color_dic[f] for f in features], alpha=0.5, label='Imag (negative)')
        # plt.bar(row.index, row.values, color=[color_dic[feature] for feature in row.index])
        
        # ax.tick_params(axis='x', labelbottom=True)
        plt.xticks(x, features, rotation=90)
        plt.axhline(0, color='black', linewidth=1)
        plt.ylabel("<-- Imaginary magnitude   |   Real magnitude -->")
        plt.legend()
        plt.title(f"Real and Imaginary Components of {pca_component} " + title)
        # plt.title(f'{pca_component} Expansion in terms of Original Stats')
        # plt.legend()
        
    fig.supxlabel('Pokemon Stat')
    fig.supylabel('Weight')
    plt.show()


def hover_scatter(df, color_by='types'):
    fig = go.Figure()
    offset = 0.25  # shift centroids
    xnoise = 0.2
    ynoise = 0.1
    
    # Compute column means and get the sorted order
    col_means = df.mean(axis=0)
    sorted_cols = col_means.sort_values().index  # ascending order
    color_dic = pcol.color_dics[color_by]
    color = [color_dic[ind] for ind in df.index]
    
    for i, col in enumerate(sorted_cols):
        values = df[col]
        nvals = len(values)

        # Mean ± std markers (shifted)
        mean = values.mean()
        std = values.std()
        fig.add_trace(go.Scatter(
            x=[i + offset],
            y=[mean],
            mode='markers',
            marker=dict(color='black', size=10),
            error_y=dict(type='data', array=[std], visible=True),
            hoverinfo='skip',
            showlegend=False,
            opacity=0.5,
        ))
        
        # Jittered scatter positions
        x = np.full(nvals, i) + xnoise * np.random.uniform(-1, 1, size=nvals)
        y = values + ynoise * np.random.uniform(-1, 1, size=nvals) 

        z = (values - mean)/std
        # print(z)
        
        """
        # Color points by sign with transparency
        colors = [
        'rgba(255,128,114,0.5)' if val < 0 else
        'rgba(128,0,128,0.5)' if val == 0 else
        'rgba(135,206,235,0.5)' 
        for val in values
    ]
        """
        # Scatter points (hover shows only row label)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color=color, size=8),
            text=df.index,
            customdata=np.stack((values,z),axis=-1),
            hovertemplate=(
                '%{text}<br>'
                'Damage: %{customdata[0]:.1f}<br>'
                'z-score: %{customdata[1]:.1f}<br>'
                '<extra></extra>'# removes the default trace name box
            ),
            showlegend=False,
            opacity=0.5,
        ))
        

    # Ensure x-axis, ticks, and labels all match the sorted order
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(sorted_cols)),
            ticktext=sorted_cols,
            tickangle=-90
        ),
        yaxis_title="Relative Damage",
        xaxis_title='Attacking Type',
        title="Net Damage with mean ± std (sorted by mean)",
        width=900,
        height=400,
        showlegend=False
    )
    
    fig.show()
    # print(len(fig.data)) 


def ensure_tuple(item):
    if not isinstance(item, tuple):
        # If it's not a tuple, convert it.
        # For single items, wrap them in a list first to ensure tuple() works as expected.
        if isinstance(item, (list, dict, set)): # Check for common iterables
            return tuple(item)
        else: # For non-iterable items like integers, floats, booleans AND ALSO STRING
            return (item,) # Create a single-element tuple
    else:
        return item


    