import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig


def importdf(name='', index_col_length=1):
    if name=='':
        print("no name given")
    name = '../Input_data/' + name + ".csv"
    labels = list(range(index_col_length))
    return pd.read_csv(name, index_col=labels, header=labels)


def exportdf(df, name=""):
    if name=='':
        print("no name given")
    name = '../Input_data/' + name + ".csv"
    print(f"Saved as {name}")
    df.to_csv(name)
    
    
def slice_df(df , pokemon_types):
    return df.loc[pokemon_types, pokemon_types]
    

def eigen(df, side='col'):
    A = np.asarray(df, dtype=float)
    m, n = A.shape
    if A.ndim != 2 or m != n:
        raise ValueError("A must be a square matrix")
    A = np.asarray(df, dtype=float)

    right = True
    left = False
    if side!='col':
        right = False
        left = True
        
    eigenvalues, eigenvectors = eig(A, right=right, left=left)
    eigenvectors = pd.DataFrame(eigenvectors, index=df.index, columns=[f'EV{i+1}' for i in range(n)])
    # transpose for plotting purposes and for consistency of having original basis on columns
    return eigenvalues, eigenvectors.T


def col_eigen(df):
    return eigen(df, side='col')


def row_eigen(df):
    return eigen(df, side='row')
    

# column vectors eigenvectors tally the total damage taken in across all types
def damage_in_eigenvectors(df):
    eigenvalues, eigenvectors = row_eigen(df)
    eigenvectors.index.name = 'Damage In'
    return eigenvalues, eigenvectors


# ROW vectors eigenvectors tally the total damage given out across all types
def damage_out_eigenvectors(df):
    # can reuse the function by transposing
    eigenvalues, eigenvectors = col_eigen(df)
    eigenvectors.index.name = 'Damage Out'
    return eigenvalues, eigenvectors


def perron_dense(df, side='col'):
    eigenvalues, eigenvectors = eigen(df, side=side)
    # find the index where eigenvalue norm is maximum
    idx = np.argmax(np.abs(eigenvalues))
    # print(idx)
    perron_val = np.real_if_close(eigenvalues[idx])
    perron_vec = np.real_if_close(eigenvectors.iloc[idx,:])
    # ensure positive 
    perron_vec = np.abs(perron_vec)
    # scale to unit L1 norm
    # perron_vec /= perron_vec.sum()
    # turn into row of df
    perron_vec = pd.DataFrame(perron_vec, index=df.index, columns=['Perron Vector']).T
    # print(perron_vec)
    return perron_val, perron_vec
        

def perron_power(df, tol=1e-12, max_iter=20000, normalize='l1', random_state=None):
    A = np.asarray(df, dtype=float)
    m, n = A.shape
    if A.ndim != 2 or m != n:
        raise ValueError("A must be a square matrix")

    # generate random guess of vector 
    rng = np.random.default_rng(random_state)
    v = rng.random(n)
    v = np.abs(v)
    if normalize == 'l1':
        v /= v.sum()
    elif normalize == 'l2':
        v /= np.linalg.norm(v)
    else:
        v /= v.max()

    # iterate over repeated applications of the matrix
    for it in range(1, max_iter+1):
        # apply the matrix
        w = A @ v
        # guard tiny negatives
        w[w < 0] = 0.0  
        # calculate denominator to normalize by
        denom = w.sum() if normalize == 'l1' else np.linalg.norm(w) if normalize == 'l2' else w.max()
        # interrupt if zero vector
        if denom == 0:
            break
        # else normalize
        v_next = w / denom
        # check for tolerance
        if np.linalg.norm(v_next - v, ord=1) < tol:
            v = v_next
            break
        # update trial vector
        v = v_next

    # turn into row of df
    perron_vec = pd.DataFrame(v, index=df.index, columns=['Perron Vector']).T
    print(perron_vec)
    # Rayleigh estimate for lambda
    if v@v == 0:
        perron_val = 0
    else:
        perron_val = (v @ (A @ v)) / (v @ v)
    
    return perron_val, perron_val


def pair_probability_matrix(df):
    df_tot = df + df.T
    div = np.where(df_tot == 0, 0.5, df / df_tot)
    df_pair = pd.DataFrame(div, index = df.index, columns=df.columns)
    return df_pair


def stochastic_matrix(df, by='col'):
    # rescale each column by their respective sum
    if by=='col':
        return df/df.sum(axis=0)
    # rescale each row by their respective sum
    elif by=='row':
        return df.div(df.sum(axis=1), axis=0)
        

def real_eigenvectors(df):
    # Compute eigenvalues and eigenvectors
    eigvecs = column_eigenvectors(df)
    # print(eigvecs)
    real_index = []
    for i, (name, vec) in enumerate(eigvecs.iterrows()):
        # print(vec)
        skip = False
        for vi in vec:
            # print(vi)
            re = np.real(vi)
            im = np.imag(vi)
            if im != 0:
                skip = True
        if not skip:
            real_index.append(i)
            # print(f'eigenvector: {i+1}')
            # print(name)
            # print(vec)  
            # plt.bar(vec.index, vec.values)
            # plt.xticks(rotation=90)
            # plt.show()
    # print(real_index)
    real_eigvecs = eigvecs.iloc[real_index] 
    # print(real_eigvecs)
    return real_eigvecs
    

def scatter_eigenvalues(df):
    # plot origin
    # plt.scatter(0,0,label='origin', color='black', marker='x')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    eigenvalues, eigenvectors = col_eigen(df)

    # plot perron eigenvalue separately
    pval = eigenvalues[0]
    plt.scatter(np.real(pval), np.imag(pval), color='purple', label='principal')

    # plot remaining eigenvalues
    for val in eigenvalues[1:]:
        plt.scatter(np.real(val), np.imag(val), color='skyblue', alpha=1)

    # plot rings
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.plot(x, y, label='r=1', linestyle='--', color='yellow')
    plt.plot(2*x, 2*y, label='r=2', linestyle='--', color='orange')
    plt.plot(3*x, 3*y, label='r=3', linestyle='--', color='red')
    
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Eigenvalues Scatter Plot')
    plt.legend()
    # equal aspect ratio
    plt.axis('equal')
    plt.show()


def calculate_statistics(df, name='', by='row'):
    df2 = pd.DataFrame(index=df.index)
    if name=='':
        name = by
    if by=='row':
        axis=1
    else:
        axis=0
    df2.loc[:, name + '_tot'] = df.mean(axis=axis)
    df2.loc[:, name + '_avg'] = df.mean(axis=axis)
    df2.loc[:, name + '_std'] = df.std(axis=axis)
    df2.loc[:, name + '_var'] = df.std(axis=axis)
    return df2


def calculate_net_statistics(df, name=''):
    if name=='':
        name = 'damage_net'
    # calculate col statistics
    col_name = 'damage_in'
    dfc = calculate_statistics(df, col_name, by='col')
    # calculate row statistics
    row_name = 'damage_out'
    dfr = calculate_statistics(df, row_name, by='row')
    # after doing row and column sums we can add them to the dataframe
    df2 = dfc.join(dfr, how='outer')
    
    # --- Compute damage given - taken
    df2.loc[:, name + '_tot'] = df2.loc[:, row_name + '_tot'] - df2.loc[:, col_name + '_tot'] 
    df2.loc[:, name + '_avg'] = df2.loc[:, row_name + '_avg'] - df2.loc[:, col_name + '_avg']
    df2.loc[:, name + '_var'] = df2.loc[:, row_name + '_var'] + df2.loc[:, col_name + '_var']
    df2.loc[:, name + '_std'] = np.sqrt(df2.loc[:, name + '_var'])

    return df2