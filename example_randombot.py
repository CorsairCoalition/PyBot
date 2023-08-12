from ggbot.core import PythonBot
import ggbot.utils
import random

class RandomBot(PythonBot):
    """A simple bot that only takes random moves."""

    def do_turn(self):

        # Get a list of our tiles with armies that can move
        moveable_units = self.game.get_moveable_army_tiles()

        # If there are no movable units, skip this turn
        if len(moveable_units) == 0:
            return

        # pick random unit
        moveable_unit = random.choice(moveable_units)

        # move it to a random adjacent tile
        adj_tiles = self.game.get_adjacent_tiles(moveable_unit)
        self.move(moveable_unit, random.choice(adj_tiles))

if __name__ == "__main__":
    
    config = ggbot.utils.get_config_from_cmdline_args()    
    # config = ggbot.utils.get_config_from_file('config.pybot1.json')
    
    RandomBot().with_config(config).run()
