from pythonbot import PythonBot
import random


class RandomBot(PythonBot):
    """A simple bot that only takes random moves."""

    def __init__(self, bot_id: str) -> None:
        super().__init__(bot_id)
        pass

    def do_turn(self) -> None:

        # Get a list of all the territories that we own
        moveable_units = self.game.map.get_moveable_army_tiles()
        if len(moveable_units) == 0:
            return

        # pick random unit
        moveable_unit = random.choice(moveable_units)

        # move it to a random adjacent tile
        adj_tiles = self.game.map.get_adjacent_tiles(moveable_unit)
        self.move(moveable_unit, random.choice(adj_tiles))


if __name__ == "__main__":

    from pythonbot import RedisConnectionManager
    import os
    import json

    # Using the same config as other CorsairCoalition components
    redis_config_file = os.path.dirname(__file__) + './config.json'
    config: dict = json.load(open(redis_config_file))['redisConfig']
    redis_config = dict(host=config['HOST'], port=config['PORT'], username=config['USERNAME'],
                        password=config['PASSWORD'], ssl=False)

    # This is the main entry point for the bot.
    with RedisConnectionManager(redis_config) as rcm:

        # Multiple bots can be registered
        # rcm.register(randombot.RandomBot("cortex-7LQqyM8"))
        rcm.register(RandomBot("cortex-7LQqyM8"))

        # Start listening for Redis messages
        rcm.run()
