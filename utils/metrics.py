import numpy as np 


# Adding type hints for better code clarity and numpy style comments for documentation  
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    Returns
    -------
    float
        Mean Absolute Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    Returns
    -------
    float
        Bias
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_pred - y_true)
