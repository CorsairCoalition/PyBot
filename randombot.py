from pythonbot import PythonBot
import random

class RandomBot(PythonBot):

    def __init__(self, bot_id:str) -> None:
        super().__init__(bot_id)
        pass

    def do_turn(self) -> None:

        # Get a list of all the territories that we own
        moveable_units = self.game.map.get_moveable_army_tiles()
        if len(moveable_units) == 0:
            return
        
        #pick random unit
        moveable_unit = random.choice(moveable_units)
        
        #move it to a random adjacent tile
        adj_tiles = self.game.map.get_adjacent_tiles(moveable_unit)
        self.move(moveable_unit, random.choice(adj_tiles))
