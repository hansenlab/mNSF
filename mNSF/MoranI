import numpy as np
from scipy.spatial.distance import cdist

def calculate_morans_i(coordinates, y, p=2, threshold=None, num_permutations=999):
    """
    Calculate Moran's I statistic and its significance
    
    Parameters:
    coordinates : array-like
        Array of point coordinates (x, y)
    y : array-like
        The variable of interest
    p : float, optional (default=2)
        Power parameter for inverse distance weighting
    threshold : float, optional (default=None)
        Distance beyond which spatial weights are zero
    num_permutations : int, optional (default=999)
        Number of permutations for calculating significance
    
    Returns:
    I : float
        Moran's I statistic
    p_value : float
        The pseudo p-value for Moran's I
    """
    
    def calculate_weights(coords):
        dist = cdist(coords, coords)
        np.fill_diagonal(dist, np.finfo(float).max)
        
        if threshold is not None:
            dist[dist > threshold] = np.finfo(float).max
        
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.power(dist, -p, where=dist!=0)
        
        w[~np.isfinite(w)] = 0
        row_sums = w.sum(axis=1)
        w[row_sums > 0] = w[row_sums > 0] / row_sums[row_sums > 0, np.newaxis]
        
        return w
    
    def calculate_i(values, weights):
        values = np.asarray(values)
        weights = np.asarray(weights)
        
        value_mean = np.mean(values)
        value_dev = values - value_mean
        
        numerator = np.sum(weights * np.outer(value_dev, value_dev))
        denominator = np.sum(value_dev**2)
        
        return (len(values) / np.sum(weights)) * (numerator / denominator)
    
    # Calculate weights
    w = calculate_weights(coordinates)
    
    # Calculate Moran's I
    I = calculate_i(y, w)
    
    # Calculate significance
    I_simulations = np.array([calculate_i(np.random.permutation(y), w) for _ in range(num_permutations)])
    p_value = (np.sum(np.abs(I_simulations) >= np.abs(I)) + 1) / (num_permutations + 1)
    
    return I, p_value

