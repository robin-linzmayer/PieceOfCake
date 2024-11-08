import numpy as np
from scipy.optimize import linear_sum_assignment

def optimal_assignment(R, V):
    num_requests = len(R)
    num_values = len(V)
    cost_matrix = np.zeros((num_requests, num_values))
    epsilon = 1e-6  # Small constant to prevent division by zero

    # Fill the cost matrix with adjusted relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            denominator = abs(r) + abs(v) + epsilon
            cost_matrix[i][j] = abs(r - v) / denominator

    # Solving the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Assignment array where assignment[i] is the index of V matched to R[i]
    assignment = [int(col_indices[i]) for i in range(num_requests)]
    return assignment



# Test cases
def test_optimal_assignment():
    # Case 1: Basic functionality test with simple inputs
    R = [1, 2, 3]
    V = [2, 1, 3]
    result = optimal_assignment(R, V)
    print("Test Case 1:", result)  # Expected: [1, 0, 2]

    # Case 2: R has a zero, should handle division by zero gracefully
    R = [0, 2, 3]
    V = [1, 2, 3]
    try:
        result = optimal_assignment(R, V)
        print("Test Case 2:", result)
    except ZeroDivisionError:
        print("Test Case 2 Failed: Division by zero error")

    # Case 6: Duplicate values in R and V
    R = [1, 1, 2]
    V = [2, 2, 1]
    result = optimal_assignment(R, V)
    print("Test Case 6:", result)  # Expected: depends on cost optimization

    # Case 7: Negative values in R and/or V
    R = [-1, -2, -3]
    V = [-2, -1, -3]
    result = optimal_assignment(R, V)
    print("Test Case 7:", result)  # Expected: cost should be correctly computed

    # Case 8: Large input values to test floating point precision
    R = [1e10, 1e11, 1e12]
    V = [1e10 + 1, 1e11 + 1, 1e12 + 1]
    result = optimal_assignment(R, V)
    print("Test Case 8:", result)  # Expected: closest matches with slight difference

# Run tests
test_optimal_assignment()
