from ggbot.core import PythonBot
import ggbot.utils
from ggbot.algorithms import aStar
import random


class PathFinderBot(PythonBot):
    """This bot demonstates how to use the multi-target aStar algorithm to obtain a sequence of moves along a path"""

    WAIT_TICK_INTERVAL: int = 12

    def do_turn(self) -> None:

        if self.queued_moves > 0:
            return # wait for queued moves to execute

        if self.game.tick % self.WAIT_TICK_INTERVAL != 0:
            return  # wait for armies to build up

        # pick unit with largest army
        strongest_owned_tile = max(self.game.own_tiles, key=lambda x: x.strength)
        if strongest_owned_tile.strength <= 1:
            return  # not big enough to move

        # pick a random position on the board
        target = -1 # tiles outside the game board are never passable
        while not self.game.is_passable(target):
            target = random.randint(0, self.game.size-1)

        # get a tile path from the start to the target 
        path = aStar(self.game, strongest_owned_tile.tile, target)

        # send the moves to the movement queue
        self.queue_moves(path)
     

if __name__ == "__main__":
    
    #config = ggbot.utils.get_config_from_file("../config.json")
    config = ggbot.utils.get_config_from_cmdline_args()
    
    PathFinderBot().with_config(config).run()

