import math

def sneak(start_pos, end_pos, cake_width, cake_len):
        """
        Given a start position & goal position, uses the 1-pixel shaving technique
        to create a list of necessary steps to get from the start to the goal without
        cutting across the cake.
        """
        nearest_x, x_dist = nearest_edge_x(start_pos, cake_width)
        nearest_y, y_dist = nearest_edge_y(start_pos, cake_len)

        end_x, end_x_dist = nearest_edge_x(end_pos, cake_width)
        end_y, end_y_dist = nearest_edge_y(end_pos, cake_len)

        moves = []

        # if we are on the top or bottom or the board and require more than 1 move
        if y_dist == 0 and (x_dist > 0.1 or nearest_x != end_x):
            bounce_y = bounce(nearest_y)
            moves.append([end_x, bounce_y])

            # if the end is not on the same line so we must ricochet off of corner
            if end_y_dist > 0 or nearest_y != end_y:
                bounce_x = bounce(end_x)
                moves.append([bounce_x, nearest_y])

                # if the end position is on the opposite side
                if end_y_dist == 0:
                    bounce_y = bounce(end_y)
                    moves.append([bounce_x, end_y])
                    moves.append([end_x, bounce_y])

        # if we are on the left or right side of the board and require more than 1 move
        elif x_dist == 0 and (y_dist > 0.1 or nearest_y != end_y):
            bounce_x = bounce(nearest_x)
            moves.append([bounce_x, end_y])

            # if the end is not on the same line so we must ricochet off of corner
            if end_x_dist > 0 or nearest_x != end_x:
                bounce_y = bounce(end_y)
                moves.append([nearest_x, bounce_y])

                # if the end position is on the opposite side
                if end_x_dist == 0:
                    bounce_x = bounce(end_x)
                    moves.append([end_x, bounce_y])
                    moves.append([bounce_x, end_y])

        moves.append(end_pos)
        return moves

def nearest_edge_x(pos, cake_width):
    """
    Returns the nearest X-edge and the distance to said edge
    X-edge is 0 if the position is closer to the left side, or cake_width
        if it is closer to the right side.
    """
    min_x = pos[0]
    x_edge = 0
    if cake_width - pos[0] < min_x:
        min_x = cake_width - pos[0]
        x_edge = cake_width
    return x_edge, min_x


def nearest_edge_y(pos, cake_len):
    """
    Returns the nearest Y-edge and the distance to said edge
    Y-edge is 0 if the position is closer to the top, or cake_len
        if it is closer to the bottom.
    """
    min_y = pos[1]
    y_edge = 0
    if cake_len - pos[1] < min_y:
        min_y = cake_len - pos[1]
        y_edge = cake_len
    return y_edge, min_y


def bounce(margin):
    """
    Returns a value 0.01 away from the provided margin
    """
    if margin == 0:
        return 0.01
    return round(margin - 0.01, 2)

def is_uniform(requests, tolerance=0) -> bool:
    """
    Returns whether or not the requests can be considered uniform
    """
    if len(requests) < 1: return True
    return (max(requests) - min(requests)) <= (2 * tolerance)

def divide_requests(requests):
    """
    If we were to divide the requests into a nearly-square array,
    we return the total sum of every request in the list, a list of 
    the sums of all the requests that would be in each row, and a 
    list of the sums of all the requests that would be in each column
    """
    n = len(requests)
    s = int(math.sqrt(n))
    requests_copy = requests[:]
    median = (max(requests) + min(requests)) / 2
    if n%s != 0: n = s * math.ceil(n/s)
    while len(requests_copy) < n:
        requests_copy.append(median)
    total_sum = 0
    h_sums = []
    v_sums = []
    for i in range(0,len(requests_copy)):
        val = requests_copy[i]
        total_sum += val

        if i<s: v_sums.append(val)
        else: v_sums[int(i%s)] += val

        if int(i/s) >= len(h_sums): h_sums.append(val)
        else: h_sums[int(i/s)] += val
    return total_sum, h_sums, v_sums

