class PieceOfCakeState:
    def __init__(self, polygons, cur_pos, turn_number, requests, cake_len, cake_width):
        """
            Args:
                polygons (List[Polygon]): List of polygons
                cur_pos (Tuple[int, int]): Current position of the knife
        """
        self.polygons = polygons
        self.cur_pos = cur_pos
        self.turn_number = turn_number
        self.requests = requests
        self.cake_len = cake_len
        self.cake_width = cake_width
