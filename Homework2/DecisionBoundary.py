import MachineLearningModel as mlm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plotDecisionBoundary(X1, X2, y, model):
    """
    Plots the decision boundary for a binary classification model along with the training data points.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y (array-like): Target labels.
        model (object): Trained binary classification model with a `predict` method.

    Returns:
        None
    """
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)
    
    h = 0.01
    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1, X2.max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    classes = model.predict(grid_points)
    clz_mesh = classes.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
    plt.scatter(X1, X2, c=y, cmap=cmap_bold, edgecolors='k', s=20, alpha=0.7)