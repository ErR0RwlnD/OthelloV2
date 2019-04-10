from hyper import Hyper


class Board():
    _directions = [(1, 1), (1, 0), (1, -1), (0, -1),
                   (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self):
        self.pieces = list(list(0 for _ in range(8)) for _ in range(8))
        self.pieces[3][3] = Hyper.white
        self.pieces[4][4] = Hyper.white
        self.pieces[3][4] = Hyper.black
        self.pieces[4][3] = Hyper.black

    def __getitem__(self, index):
        return self.pieces[index]

    def countDiff(self, color):
        count = 0
        for y in range(8):
            for x in range(8):
                if self[x][y] == color:
                    count += 1
                if self[x][y] == -color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        moves = set()
        for y in range(8):
            for x in range(8):
                if self[x][y] == color:
                    newMoves = self.get_moves_for_square((x, y))
                    moves.update(newMoves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(8):
            for x in range(8):
                if self[x][y] == color:
                    newMoves = self.get_moves_for_square((x, y))
                    if len(newMoves) > 0:
                        return True
        return False

    def get_moves_for_square(self, square):
        (x, y) = square
        color = self[x][y]

        if color == 0:
            return None

        moves = []
        for direction in self._directions:
            moves.append(self._discover_move(square, direction))
        return list(filter(None, moves))

    def execute_move(self, move, color):
        flips = [flip for direction in self._directions
                 for flip in self._get_flips(move, direction, color)]
        assert len(list(flips)) > 0
        for x, y in flips:
            self[x][y] = color

    def _discover_move(self, origin, direction):
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction):
            if self[x][y] == 0:
                if flips:
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        flips = [origin]
        for x, y in Board._increment_move(origin, direction):
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                return flips
        return []

    @staticmethod
    def _increment_move(move, direction):
        move = list(map(sum, zip(move, direction)))
        while all(map(lambda x: 0 <= x < 8, move)):
            yield move
            move = list(map(sum, zip(move, direction)))
