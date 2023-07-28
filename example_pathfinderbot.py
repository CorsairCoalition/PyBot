from pythonbot import PythonBot, Algorithms, Move
import random


class PathFinderBot(PythonBot):
    """This bot demonstates how to use the aStar pathfinding algorithm to obtain a sequence of moves along a path"""

    WAIT_TICK_INTERVAL: int = 12

    def __init__(self, game_config: dict) -> None:
        super().__init__(game_config)
        pass

    def do_turn(self) -> None:

        if self.queued_moves > 0:
            return # wait for queued moves to execute

        if self.game.tick % self.WAIT_TICK_INTERVAL != 0:
            return  # wait for armies to build up

        # pick unit with largest army
        strongest_owned_tile = max(
            self.game.own_tiles, key=lambda x: x.strength)
        if strongest_owned_tile.strength <= 1:
            return  # not big enough to move

        # pick a random position on the board
        target = -1
        while not self.game.is_passable(target):
            target = random.randint(0, self.game.size-1)

        # get the list of tiles along the path
        path = Algorithms.aStar(self.game, strongest_owned_tile.tile, target)

        # convert the path to a sequence of moves
        path = Move.ints_as_moves(path)

        # send the moves to the movement queue
        for move in path:
            self.move(move.start, move.end)



if __name__ == "__main__":

    from pythonbot import RedisConnectionManager
    import os
    import json
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    
    # Using the same config as other CorsairCoalition components
    config_file = uppath(__file__, 2) + os.sep + 'config.json'
    config: dict = json.load(open(config_file))
    
    # This is the main entry point for the bot.
    with RedisConnectionManager(config['redisConfig']) as rcm:

        # Instantiate bot and register it with Redis
        rcm.register(PathFinderBot(config['gameConfig']))

        # Start listening for Redis messages
        rcm.run()