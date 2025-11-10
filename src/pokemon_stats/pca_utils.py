import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from sklearn.decomposition import PCA

import pokemon_stats.poke_colors as pcol
import importlib
importlib.reload(pcol)

def standardize(df: pd.DataFrame)-> pd.DataFrame:
    """
    drop non-numeric columns and standarize the remaining columns to be mean=0 and var=1

    Args:
        df: original data frame

    Returns:
        df: standarized data frame
    """

    # check all df columns are numeric
    all_numeric = len(df.columns) == len(df.select_dtypes(include='number').columns)
    if not all_numeric:
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
        print('not all the columns fed were numeric, so we will drop them:')
        print(non_numeric_columns)
        # select only the numeric columns
        df = df.select_dtypes(include='number')

    # print(df.columns)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
    # print(df_scaled.columns)
    return df_scaled


def pca(df: pd.DataFrame, n_components: int = 0)-> tuple[pd.DataFrame, "PCA"]:
    """
    standarize the data and then return the pca transformed data

    Args:
        df: original data frame

    Returns:
        tuple
        df: pca fitted data in new basis
        PCA: pca fit object with data about transform
    """
    # print(df.columns)
    # standardize the data  
    df_scaled = standardize(df)
    # print(df_scaled.columns)
    
    # default value resorts to all components
    if n_components == 0:
        n_components = df.shape[1] 
    
    # create pca object and fit the data
    pca = PCA(n_components=n_components).fit(df_scaled)
    # print(pca.feature_names_in_)

    # do the fit
    df_pca = pca.transform(df_scaled)

    # turn np array into df with proper names
    df_pca = pd.DataFrame(df_pca)
    df_pca.columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca.index = df.index

    return df_pca, pca


def pca_components(pca: "PCA") -> pd.DataFrame:
    """
    retrieve the pca vector with weights in terms of the original features basis 
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        DataFrame with PC vectors as rows with weights in original basis as columns
    """
    
    df = pd.DataFrame(pca.components_)   # shape (rows: n PCA components, cols: original features)
    df.index = [f'PC{i+1}' for i in range(df.shape[0])]
    df.columns = pca.feature_names_in_
    # df.columns = pca.get_feature_names_out()
    # print(df.columns)
    return df 


def heatmap_pca_components(pca: "PCA") -> None:
    """
    heatmap the decomposition of each PCA components in terms of old features
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """

def bar_abs_pca_components(pca: "PCA") -> None:
    """
    bar chart showing the decomposition of PCA components in terms of absolute value of weights of old features 
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    # print(pca)
    df = pca_components(pca)
    # Take absolute value to show magnitude
    df = df.abs()
    # Colors for each feature
    # colors = plt.cm.tab20.colors[:len(df)]
    
    plt.figure(figsize=(8,6))
    
    bottom = np.zeros(len(df.columns))
    for i, (feature, col) in enumerate(df.items()):
        plt.bar(df.index, col, bottom=bottom, color=pcol.color_stats[feature], label=feature)
        
        # Add text labels
        for j in range(len(col)):
            val = col.iloc[j]
            plt.text(
                x=j,
                y=bottom[j] + val/2,
                s=f"{val:.3f}",
                ha='center',
                va='center',
                color='white' if val > 0.05 else 'black',
                fontsize=9
            )
        
        bottom += col.values  # update bottom for next slice
    
    plt.ylabel('PCA Component Magnitude')
    plt.title('Pokemon Stats PCA Component Breakdown (Absolute Values)')
    plt.legend(title='Features', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()


def bar_sign_pca_components(pca: "PCA") -> None:
    """
    bar chart showing the decomposition of PCA components in terms of positive and negative weights of old features  
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    df = pca_components(pca)
    plt.figure(figsize=(8,6))
    
    # Initialize cumulative bottoms for positive and negative stacks
    pos_bottom = np.zeros(len(df.columns))
    neg_bottom = np.zeros(len(df.columns))
    
    # Plot each feature
    for i, (feature, col) in enumerate(df.items()):
        for j in range(len(col)):
            val = col.iloc[j]
            if val >= 0:
                plt.bar(df.index[j], val, bottom=pos_bottom[j], color=pcol.color_stats[feature])
                # label
                plt.text(df.index[j], pos_bottom[j] + val/2, f"{val:.3f}", ha='center', va='center', color='white', fontsize=9)
                pos_bottom[j] += val
            else:
                plt.bar(df.index[j], val, bottom=neg_bottom[j], color=pcol.color_stats[feature])
                # label
                plt.text(df.index[j], neg_bottom[j] + val/2, f"{val:.3f}", ha='center', va='center', color='white', fontsize=9)
                neg_bottom[j] += val
    
    plt.ylabel('PCA Components with Sign')
    plt.title('Diverging Stacked Bar Chart of PCA Components')
    # plt.legend(df.index, title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create legend handles manually
    handles = [mpatches.Patch(color=pcol.color_stats[feature], label=feature) for i, feature in enumerate(df.columns)]
    plt.legend(handles=handles, title='Features', bbox_to_anchor=(1.05,1), loc='upper left')
    
    plt.axhline(0, color='black', linewidth=0.8)  # zero line
    plt.tight_layout()
    plt.show()
    
    
def pca_var_bar2(pca: "PCA") -> None:
    """
    two aligned bar charts of individual and cummulative variance of PCA components
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    var = pd.Series(pca.explained_variance_ratio_)
    n_components = len(var)
    var.index = [i+1 for i in range(n_components)]
    cumsum = var.cumsum()
    
    fig, axes = plt.subplots(2,1,tight_layout=True, sharex =True, sharey=True)
    
    plt.sca(axes.flat[0])
    plt.bar(var.index, var.values, label='individual', alpha=0.5, color=list(pcol.color_pca.values()))
    plt.title('Individual')
    
    plt.sca(axes.flat[1])
    plt.bar(cumsum.index, cumsum.values, label='cummulative', alpha=0.5,  color=list(pcol.color_pca.values()))
    plt.title('Cummulative')
    
    fig.supxlabel('PCA Component')
    fig.supylabel('variance fraction')
    plt.show()


def pca_var_bar(pca: "PCA") -> None:
    """
    single bar chart overlappinfg individual and cummulative variance of PCA components
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    var = pd.Series(pca.explained_variance_ratio_)
    n_components = len(var)
    var.index = [i+1 for i in range(n_components)]
    cumsum = var.cumsum()

    # Plot
    plt.figure(figsize=(8,5))
    
    # Cumulative sum (overlay line or bar with transparency)
    plt.bar(var.index, cumsum.values, color='orange', alpha=1, label='Cumulative')

    # Individual contributions (base bars)
    plt.bar(cumsum.index, var.values, color='skyblue', alpha=1, label='Individual')
    
    # Add legend
    plt.legend()

    plt.xlabel('PCA Component')
    plt.ylabel('Variance Fractrion')
    plt.title('Cumulative Bar Chart with Individual Contributions')
    plt.show()


def pca_var_stackedbar(pca: "PCA") -> None:
    """
    single stacked bar chart of individual and cummulative variance of PCA components
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    var = pd.Series(pca.explained_variance_ratio_)
    n_components = len(var)
    var.index = [i+1 for i in range(n_components)]
    cumsum = var.cumsum()
    
    # Initialize bottom values
    bottom = 0
    plt.figure(figsize=(6,6))
    
    # Plot each slice
    for i, val in enumerate(var):
        plt.bar('Total', val, bottom=bottom, label=var.index[i], color=list(pcol.color_pca.values())[i])
        # optional: label in middle
        plt.text(x='Total',
                 y=bottom + val/2,
                 s=f"{val:.2f}",
                 ha='center', 
                 va='center', 
                 color='white', 
                 fontweight='bold'
                )
        bottom += val
    
    # Get current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Reverse order of both
    handles = handles[::-1]
    labels = labels[::-1]
    # Add legend
    plt.legend(handles, labels, title='PCA Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.ylabel('Variance Fraction')
    plt.title('Cumulative Variance Fraction by PCA Component')
    plt.xticks([])  # remove x-axis label since it's just one bar
    plt.yticks(np.arange(0,1.1,0.1))
    plt.tight_layout()
    plt.show()


def bar_bar(pca: "PCA") -> None:
    """
    consecutive stacked bar charts of individual and cummulative variance of PCA components
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    s = pd.Series(pca.explained_variance_ratio_)
    n = len(s)
    # colors = plt.cm.tab20.colors[:n]  # distinct colors

    # Compute cumulative contributions for each column
    cumsum_matrix = np.zeros((n, n))  # rows = components, cols = cumulative steps
    for col in range(n):
        cumsum_matrix[:col+1, col] = s.iloc[:col+1].values

    plt.figure(figsize=(max(6, n), 5))

    # Plot stacked bars column by column
    for col in range(n):
        bottom = 0
        for row in range(n):
            val = cumsum_matrix[row, col]
            if val > 0:
                plt.bar(
                    x=col,
                    height=val,
                    bottom=bottom,
                    color=list(pcol.color_pca.values())[row],
                    edgecolor='black',
                    # label = row+1,
                )
                # Add label in middle
                plt.text(
                    x=col,
                    y=bottom + val/2,
                    s=f"{val:.2f}",
                    ha='center',
                    va='center',
                    color='white' if val > 0 else 'black',
                    fontsize=9
                )
                bottom += val

    legend_handles = [mpatches.Patch(color=list(pcol.color_pca.values())[i], label=s.index[i]+1) for i in range(n)]
    legend_handles = legend_handles[::-1]
    plt.legend(handles=legend_handles, title='PCA Component', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(range(n), [f'{i+1}' for i in range(n)])
    plt.xlabel('Cummulative Sum Step')
    plt.ylabel('Variance Fraction')
    plt.title('Cummulative Variance Fraction Stacked Barchart')
    plt.tight_layout()
    plt.show()


def bar_bar2(pca: "PCA") -> None:
    """
    consecutive stacked bar charts of individual and cummulative variance of PCA components
    
    Args:
        pca: pca fit object with data about transform
    
    Returns:
        None
    """
    
    s = pd.Series(pca.explained_variance_ratio_)
    n = len(s)
    colors = plt.cm.tab20.colors[:n]  # distinct colors

    # Compute cumulative contributions for each column
    cumsum_matrix = np.zeros((n, n))  # rows = components, cols = cumulative steps
    for col in range(n):
        cumsum_matrix[:col+1, col] = s.iloc[:col+1].values

    plt.figure(figsize=(max(6, n), 5))

    # Plot stacked bars column by column
    for col in range(n):
        bottom = 0
        for row in range(n):
            val = cumsum_matrix[row, col]
            if val > 0:
                plt.bar(
                    x=col,
                    height=val,
                    bottom=bottom,
                    color=colors[row],
                    edgecolor='black',
                    # label = row+1,
                )
                # Add label in middle
                plt.text(
                    x=col,
                    y=bottom + val/2,
                    s=row+1,
                    ha='center',
                    va='center',
                    color='white' if val > 0 else 'black',
                    fontsize=9
                )
                bottom += val

    legend_handles = [mpatches.Patch(color=colors[i], label=f"{s.values[i]:.2f}") for i in range(n)]
    legend_handles = legend_handles[::-1]
    plt.legend(handles=legend_handles, title='PCA Component', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(range(n), [f'{i+1}' for i in range(n)])
    plt.xlabel('Cummulative Sum Step')
    plt.ylabel('Variance Fraction')
    plt.title('Cummulative Variance Fraction Stacked Barchart')
    plt.tight_layout()
    plt.show()
