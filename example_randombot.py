from ggbot.core import PythonBot
import random


class RandomBot(PythonBot):
    """A simple bot that only takes random moves."""

    def __init__(self, bot_id: str) -> None:
        super().__init__(bot_id)
        pass

    def do_turn(self) -> None:

        # Get a list of all the territories that we own
        moveable_units = self.game.get_moveable_army_tiles()
        if len(moveable_units) == 0:
            return

        # pick random unit
        moveable_unit = random.choice(moveable_units)

        # move it to a random adjacent tile
        adj_tiles = self.game.get_adjacent_tiles(moveable_unit)
        self.move(moveable_unit, random.choice(adj_tiles))




if __name__ == "__main__":

    from ggbot.core import RedisConnectionManager
    import os
    import json
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

    # Using the same config as other CorsairCoalition components
    config_file = uppath(__file__, 2) + os.sep + 'config.json'
    config: dict = json.load(open(config_file))
    
    # This is the main entry point for the bot.
    with RedisConnectionManager(config['redisConfig']) as rcm:

        # Instantiate bot and register it with Redis
        rcm.register(RandomBot(config['gameConfig']))

        # Start listening for Redis messages
        rcm.run()
