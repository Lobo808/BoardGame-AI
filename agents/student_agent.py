# Student agent: Add your own agent here
import math

from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        self.counter = 0

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # List of all possible next states in terms of position
        reachable = self.reachable(chess_board, my_pos, adv_pos, max_step, 0)
        set(reachable)

        # Finding the best move among all reachable squares
        # The goal is to get as close as possible to the opponent and try to block it
        best_dist = 2*chess_board.shape[0]
        wall_side = 0
        final_pos = my_pos
        for pos in reachable:
            dist, side = self.dist_dir(chess_board, pos, adv_pos)
            if dist < best_dist:
                final_pos = pos
                best_dist = dist
                wall_side = side
        final_x, final_y = final_pos
        if chess_board[final_x, final_y, wall_side]:    # Wall already present at the intended side, choose another wall
            for value in self.dir_map.values():
                if not chess_board[final_x, final_y, value]:
                    wall_side = value
                    break

        return final_pos, wall_side

    def reachable(self, board, my_pos, adv_pos, max_step, cur_step):
        """
        Find all reachable squares from current position.
        """
        squares = []
        x, y = my_pos
        wall_count = 0

        # Optimization for avoiding entering a square that has 2 walls or more
        for value in self.dir_map.values():
            if board[x, y, value]:
                wall_count += 1
        if not wall_count >= 2:
            squares.append(my_pos)

        # Depth-first search
        if cur_step == max_step:
            return squares
        cur_x, cur_y = my_pos
        for value in self.dir_map.values():
            if not board[cur_x, cur_y, value]:
                if value == 0:      # Move up
                    new_my_pos = (cur_x - 1, cur_y)
                    if not new_my_pos == adv_pos:
                        child_squares = self.reachable(board, new_my_pos, adv_pos, max_step, cur_step + 1)
                        squares = squares + child_squares
                elif value == 1:    # Move right
                    new_my_pos = (cur_x, cur_y + 1)
                    if not new_my_pos == adv_pos:
                        child_squares = self.reachable(board, new_my_pos, adv_pos, max_step, cur_step + 1)
                        squares = squares + child_squares
                elif value == 2:    # Move down
                    new_my_pos = (cur_x + 1, cur_y)
                    if not new_my_pos == adv_pos:
                        child_squares = self.reachable(board, new_my_pos, adv_pos, max_step, cur_step + 1)
                        squares = squares + child_squares
                else:               # Move left
                    new_my_pos = (cur_x, cur_y - 1)
                    if not new_my_pos == adv_pos:
                        child_squares = self.reachable(board, new_my_pos, adv_pos, max_step, cur_step + 1)
                        squares = squares + child_squares
        return squares

    def dist_dir(self, board, my_pos, adv_pos):
        """
        Get the distance between own position and opponent position.
        Get the direction of the opponent (to choose which side to build a wall).
        """
        x1, y1 = adv_pos
        x2, y2 = my_pos
        x_dif = x2-x1
        y_dif = y2-y1
        side = -1
        direc = math.degrees(math.atan2(x_dif, y_dif))
        if -45 < direc <= 45:
            side = 3
        elif 45 < direc <= 135:
            side = 0
        elif 135 < direc <= 180 or direc <= -135:
            side = 1
        elif -135 < direc <= -45:
            side = 2
        dist = abs(x_dif) + abs(y_dif)

        # Since finding the true cost is expensive, use it only when near the opponent
        if dist <= 3:
            dist = self.find_true_dist(board, my_pos, adv_pos)

        return dist, side

    def find_true_dist(self, board, my_pos, adv_pos):
        """
        Find the true distance from own position to opponent position.
        Walls are considered when finding path.
        """
        states = [(my_pos, 0)]
        visited = [my_pos]
        min_dist = 2*board.shape[0]
        adv_x, adv_y = adv_pos

        # Running best-first search
        while states:
            cur_pos, cur_step = states.pop(0)
            if cur_pos == adv_pos:
                min_dist = cur_step
                break
            cur_x, cur_y = cur_pos
            for value in self.dir_map.values():
                if not board[cur_x, cur_y, value]:
                    if value == 0:
                        new_my_pos = (cur_x - 1, cur_y)
                        if visited.count(new_my_pos) != 0:
                            continue
                        visited.append(new_my_pos)
                        if adv_x < cur_x:
                            states.insert(0, (new_my_pos, cur_step + 1))
                        else:
                            states.append((new_my_pos, cur_step + 1))
                    elif value == 1:
                        new_my_pos = (cur_x, cur_y + 1)
                        if visited.count(new_my_pos) != 0:
                            continue
                        visited.append(new_my_pos)
                        if adv_y > cur_y:
                            states.insert(0, (new_my_pos, cur_step + 1))
                        else:
                            states.append((new_my_pos, cur_step + 1))
                    elif value == 2:
                        new_my_pos = (cur_x + 1, cur_y)
                        if visited.count(new_my_pos) != 0:
                            continue
                        visited.append(new_my_pos)
                        if adv_x > cur_x:
                            states.insert(0, (new_my_pos, cur_step + 1))
                        else:
                            states.append((new_my_pos, cur_step + 1))
                    else:
                        new_my_pos = (cur_x, cur_y - 1)
                        if visited.count(new_my_pos) != 0:
                            continue
                        visited.append(new_my_pos)
                        if adv_y < cur_y:
                            states.insert(0, (new_my_pos, cur_step + 1))
                        else:
                            states.append((new_my_pos, cur_step + 1))
        return min_dist
