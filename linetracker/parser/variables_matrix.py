"""Get the variable matrix from the parsed variables of each line. Main function is get_variable_matrix"""

import numpy as np
import sklearn.preprocessing as skPrepro
import sklearn.metrics as skMetrics
import scipy

from typing import *  # type: ignore

class EmptyLog(Exception):
    """Exception when no logs is found"""

    def __init__(self, msg, logs: List[Dict[str, Any]]):
        super().__init__(msg)
        self.logs = logs


class NoVariable(Exception):
    """Exception when no variables are inside log lines"""

    def __init__(self, msg, logs: List[Dict[str, Any]]):
        super().__init__(msg)
        self.logs = logs

def pretty_print_matrix_variables(matrix_variables: np.ndarray, binarizer: skPrepro.MultiLabelBinarizer):
    """Allow to print the variable matrix with each line variable shown
    
    # Arguments
    - matrix_variables: np.ndarray, the variable matrix
    - binarizer: skPrepro.MultiLabelBinarizer, the binarizer used to get the matrix from the variables (see get_variable_matrix for example)
    """
    import pandas as pd
    df = pd.DataFrame(matrix_variables, index=range(len(matrix_variables)), columns=[e if len(e) < 10 else e[:10]+"..." for e in binarizer.classes_])
    print(df)
    

def jaccard_distance(mat: scipy.sparse.csr_matrix):
    """Compute the jaccard distance for a provided matrix
    
    # Arguments
    - mat: np.ndarray, sparse matrix generated from parsed variables
    # return - dense matrix with Jaccard distantes
    """
    # Intersection Samples A and B
    rows_sum = mat.getnnz(axis=1)
    ab = mat * mat.T
    # Sample Set A
    aa = np.repeat(rows_sum, ab.getnnz(axis=1))
    # Sample Set B
    bb = rows_sum[ab.indices]

    # Calculates Jaccard similarity
    similarities = ab.copy()
    similarities.data = similarities.data/(aa + bb - ab.data)

    # Calculates Jaccard distance
    distance = 1 - similarities.todense()

    return distance

def exact_jaccard_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(int)
    intersection = np.dot(matrix, matrix.T)
    # union(A,B) = A + B - A inter B
    union = np.sum(matrix, axis=1, keepdims=True) + np.sum(matrix, axis=1, keepdims=True).T - intersection
    distances = 1.0 - intersection / union
    return distances
    
def get_variable_matrix(
    parsed_events: List[List[str]],
    enable_optional_exception: bool = False
) -> np.ndarray:
    """Build the variable matrix from parsed logs

    # Arguments
    - parsed_events: List[List[str]], for each log line the variables inside this line. !! warning !! can be empty if there are no variable for a line

    # Returns
    - np.ndarray a one hot encoding indicating for each line if any of the variables inside the full log file is seen in this line
    """
    binarizer = skPrepro.MultiLabelBinarizer(sparse_output=False)
    matrix_variables = binarizer.fit_transform(parsed_events)
    if matrix_variables.shape[0] == 0:
        raise EmptyLog("No logs in the logs provided", logs=parsed_events)  # type: ignore
    if enable_optional_exception:
        if matrix_variables.shape[1] == 0:
            raise NoVariable("No variables in the logs provided", logs=parsed_events)  # type: ignore
    elif matrix_variables.shape[1] == 0:
        # if not managing the exception we assume that we have no match between the variables
        return np.zeros((matrix_variables.shape[0],matrix_variables.shape[0]))
    matrix_variables = matrix_variables.astype(bool)  # type: ignore
    # pretty_print_matrix_variables(matrix_variables,binarizer)
    # matrix_variables_distance = jaccard_distance(
    #     scipy.sparse.csr_matrix(matrix_variables),
    # )
    matrix_variables_distance = skMetrics.pairwise_distances(
        matrix_variables, metric="jaccard"
    )
    return matrix_variables_distance
