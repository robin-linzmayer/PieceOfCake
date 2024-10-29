from shapely.geometry import Polygon
import math
from scipy.optimize import linear_sum_assignment

def sorted_assign(polygons: list[Polygon], requests: list[float], d: float) -> list[int]:
    # Get sorted indices of polygons and requests in decreasing order of area
    sorted_polygon_indices = sorted(
        range(len(polygons)), key=lambda i: polygons[i].area, reverse=True
    )
    sorted_request_indices = sorted(
        range(len(requests)), key=lambda i: requests[i], reverse=True
    )

    # Assign each sorted polygon to each sorted request by index
    assignment = [-1] * len(requests)

    for i in range(min(len(sorted_polygon_indices), len(sorted_request_indices))):
        polygon_idx = sorted_polygon_indices[i]
        request_idx = sorted_request_indices[i]
        assignment[request_idx] = polygon_idx  # Match request index to polygon index

    return assignment

def greedy_best_fit_assignment(polygons: list[Polygon], requests: list[float], d: float) -> list[int]:
    # sort requests and polygons in descending order by size
    sorted_polygons = sorted(enumerate(polygons), key=lambda x: x[1].area, reverse=True)
    sorted_requests = sorted(enumerate(requests), key=lambda x: x[1], reverse=True)

    assignment = [-1] * len(requests)
    used_polygons = set()

    # iterate over each request, from largest to smallest
    for request_idx, request_size in sorted_requests:
        best_fit_polygon_idx = None
        min_penalty = float('inf')
        closest_area_diff = float('inf')

        # search for the best fitting polygon for this request
        for poly_idx, polygon in sorted_polygons:
            if poly_idx in used_polygons:
                continue  # Skip if this polygon is already used

            # calculate the percentage difference from the requested size
            area_diff_percentage = abs(polygon.area - request_size) / request_size * 100
            penalty = area_diff_percentage if area_diff_percentage > d else 0

            # find the polygon with minimum penalty and closest area to the request
            if (penalty < min_penalty) or (penalty == min_penalty and abs(polygon.area - request_size) < closest_area_diff):
                best_fit_polygon_idx = poly_idx
                min_penalty = penalty
                closest_area_diff = abs(polygon.area - request_size)

        # assigns the best fit polygon if found
        if best_fit_polygon_idx is not None:
            assignment[request_idx] = best_fit_polygon_idx
            used_polygons.add(best_fit_polygon_idx)

    return assignment


def hungarian_min_penalty(polygons: list[Polygon], requests: list[float], d: float) -> list[int]:
    n = len(requests)
    m = len(polygons)
    cost_matrix = []

    # this builds the cost matrix (penalties in our case)
    for request_size in requests:
        row = []
        for polygon in polygons:
            area_diff = abs(polygon.area - request_size) / request_size * 100
            penalty = area_diff if area_diff > d else 0
            row.append(penalty)
        cost_matrix.append(row)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assignment = [-1] * len(requests)
    for i in range(len(row_ind)):
        assignment[row_ind[i]] = col_ind[i]

    return assignment



def dp_min_penalty(polygons: list[Polygon], requests: list[float], d: float) -> list[int]:
    n = len(polygons)
    m = len(requests)
    dp = [[math.inf] * (m + 1) for _ in range(n + 1)]
    assignment = [-1] * m

    dp[0][0] = 0

    # this fills the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            area_diff = abs(polygons[i-1].area - requests[j-1]) / requests[j-1] * 100
            penalty = area_diff if area_diff > d else 0

            dp[i][j] = min(dp[i-1][j], dp[i-1][j-1] + penalty)

    # backtrack for assignment
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (abs(polygons[i-1].area - requests[j-1]) / requests[j-1] * 100 if abs(polygons[i-1].area - requests[j-1]) / requests[j-1] * 100 > d else 0):
            assignment[j-1] = i-1
            i -= 1
            j -= 1
        else:
            i -= 1

    return assignment

from itertools import permutations




# use just any assignment for now,
# ideally, we want to find the assignment with the smallest penalty
# instead of this random one
def index_assign(polygons: list[Polygon], requests: list[float], d: float) -> list[int]:
    if len(requests) > len(polygons):
        # specify amount of -1 padding needed
        padding = len(requests) - len(polygons)
        return padding * [-1] + list(range(len(polygons)))

    # return an amount of polygon indexes
    # without exceeding the amount of requests
    return list(range(len(polygons)))[: len(requests)]
