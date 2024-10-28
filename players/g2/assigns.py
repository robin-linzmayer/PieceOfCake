from shapely.geometry import Polygon


def sorted_assign(polygons: list[Polygon], requests: list[float]) -> list[int]:
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


# use just any assignment for now,
# ideally, we want to find the assignment with the smallest penalty
# instead of this random one
def index_assign(polygons: list[Polygon], requests: list[float]) -> list[int]:
    if len(requests) > len(polygons):
        # specify amount of -1 padding needed
        padding = len(requests) - len(polygons)
        return padding * [-1] + list(range(len(polygons)))

    # return an amount of polygon indexes
    # without exceeding the amount of requests
    return list(range(len(polygons)))[: len(requests)]
