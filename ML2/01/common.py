# By Lyhin Illia
# This file contains common functions and variables used in the project
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

def show_images(samples, labels=None, n_rows=2, n_cols=5):
    samples = np.array(samples)
    labels = np.array(labels)
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for idx in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, idx + 1)
        image = np.array(samples[idx]).reshape(28, 28)
        plt.imshow(image, cmap='gray')
        if labels is not None:
            plt.title(f"Label: {str(labels[idx])}")
        plt.axis('off')
    plt.show()

def show_image(sample, title=None):
    sample = np.array(sample)
    plt.figure(figsize=(2, 2))
    image = np.array(sample).reshape(28, 28)
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()



def expand_param_grid(grid):
    expanded = []
    for group in grid:
        keys = group.keys()
        values = [group[k] if isinstance(group[k], list) else [group[k]] for k in keys]
        for combination in product(*values):
            expanded.append(dict(zip(keys, combination)))
    return expanded



def tune_model(model_class, param_grid, Xtrain, ytrain, Xval, yval, verbose=False):
    """
    Trains and evaluates models using a list of hyperparameter dictionaries.
    
    Args:
        model_class: class of the model to be trained
        param_grid: unexpanded parameter grid
        Xtrain, ytrain: training data
        Xval, yval: validation data
        verbose: if True, print intermediate results

    Returns:
        best_model: the fitted model with the best score
        best_score: the best score achieved
        all_results: list of (params, score) tuples
    """
    param_grid = expand_param_grid(param_grid)
    
    best_score = -1
    best_model = None
    all_results = []

    for params in param_grid:
        model = model_class(**params)
        model.fit(Xtrain, ytrain)
        score = model.score(Xval, yval)
        all_results.append((params, score))
        
        if verbose:
            print(f"Params: {params} => Validation Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score, all_results


def apply_pca(Xtrain, ytrain, Xval, yval, n_components_range, model, param_grid, verbose=False):
    """
    Apply PCA with different numbers of components and evaluate using hyperparameter tuning for the specified model.
    
    Args:
        Xtrain, ytrain: training data
        Xval, yval: validation data
        n_components_range: range of PCA components to evaluate
        model: model to be tuned and evaluated
        param_grid: parameter grid for the model
    
    Returns:
        best_model: the fitted model with the best score
        best_score: the best score achieved
        num_components: the number of PCA components that resulted in the best score
    """
    best_score = -1
    best_model = None
    num_components = None

    pca = PCA()
    pca.fit(Xtrain)
    Xtrain_pca_all = pca.transform(Xtrain)
    Xval_pca_all = pca.transform(Xval)

    for n_components in n_components_range:
        if verbose: print(f"\nEvaluating for {n_components} components...")

        Xtrain_pca = Xtrain_pca_all[:, :n_components]
        Xval_pca = Xval_pca_all[:, :n_components]

        if verbose: print(f"Hyperparameter tuning for the {type(model()).__name__}...")
        best_model_temp, best_score_temp, _ = tune_model(model, param_grid, Xtrain_pca, ytrain, Xval_pca, yval, verbose=verbose)
        if verbose: print(f"Best score for {n_components} components: {best_score_temp:.4f}")
        # If this model has a better score, update the best model and best score
        if best_score_temp > best_score:
            best_score = best_score_temp
            best_model = best_model_temp
            num_components = n_components

    return best_model, best_score, num_components


def apply_lle(Xtrain, ytrain, Xval, yval, n_components_range, n_neighbors_list, model, param_grid, verbose=False):
    """
    Apply LLE with different numbers of components and neighbors,
    then evaluate using hyperparameter tuning for the specified model.

    Args:
        Xtrain, ytrain: training data
        Xval, yval: validation data
        n_components_range: range of LLE components to evaluate
        n_neighbors_list: list of neighbor counts to evaluate
        model: model to be tuned and evaluated
        param_grid: parameter grid for the model

    Returns:
        best_model: the fitted model with the best score
        best_score: the best score achieved
        num_components: the number of LLE components that resulted in the best score
        best_n_neighbors: the number of neighbors that resulted in the best score
    """
    best_score = -1
    best_model = None
    best_n_components = None
    best_n_neighbors = None

    for n_components in n_components_range:
        for n_neighbors in n_neighbors_list:
            if verbose:
                print(f"\nEvaluating for n_components={n_components}, n_neighbors={n_neighbors}...")

            lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='standard')
            Xtrain_lle = lle.fit_transform(Xtrain)
            Xval_lle = lle.transform(Xval)

            if verbose:
                print(f"Hyperparameter tuning for the {model.__name__}...")
            best_model_temp, best_score_temp, _ = tune_model(model, param_grid, Xtrain_lle, ytrain, Xval_lle, yval, verbose=verbose)
            if verbose:
                print(f"Score: {best_score_temp:.4f}")

            if best_score_temp > best_score:
                best_score = best_score_temp
                best_model = best_model_temp
                best_n_components = n_components
                best_n_neighbors = n_neighbors

    return best_model, best_score, best_n_components, best_n_neighbors



###
### TESTS
###

#Test expand_param_grid
def test_expand_param_grid():
    grid = [
        {'C': [0.1, 1, 10], 'kernel': 'linear'},
        {'C': [0.1, 1, 10], 'kernel': 'rbf', 'gamma': [0.01, 0.1]},
        {'C': [0.1, 1], 'kernel': 'blabla', 'gig': [2, 3]}
    ]
    expanded = expand_param_grid(grid)
    assert len(expanded) == 13
    assert expanded[0] == {'C': 0.1, 'kernel': 'linear'}
    assert expanded[5] == {'C': 1, 'kernel': 'rbf', 'gamma': 0.01}
    assert expanded[12] == {'C': 1, 'kernel': 'blabla', 'gig': 3}

