from pythonbot import PythonBot, Algorithms,Move
import random

class PathFinderBot(PythonBot):
    """This bot demonstates how to use the aStar pathfinding algorithm to obtain a sequence of moves along a path"""

    WAIT_TURN_INTERVAL:int = 12

    def __init__(self, bot_id:str) -> None:
        super().__init__(bot_id)
        pass

    def do_turn(self) -> None:
        
        if self.game.turn % self.WAIT_TURN_INTERVAL != 0:
            return # wait for armies to build up

        # pick unit with largest army
        strongest_owned_tile = max(self.game.map.own_tiles, key=lambda x:x.strength)
        if strongest_owned_tile.strength <= 1:
            return # not big enough to move

        # pick a random position on the board
        target = -1 
        while not self.game.map.is_passable(target):
            target = random.randint(0,self.game.map.size-1)

        # get the list of Moves along the path
        path = Algorithms.aStar(self.game.map,strongest_owned_tile.tile,target)

        # send the moves to the server's movement queue
        for move in path:
            self.move(move.start,move.end)


if __name__ == "__main__":

    from pythonbot import RedisConnectionManager
    import os,json

    # Using the same config as other CorsairCoalition components
    redis_config_file = os.path.dirname(__file__) + './config.json' # get file path
    config:dict = json.load(open(redis_config_file))['redisConfig'] # read file into dict
    redis_config = dict(
        host=config['HOST'],
        port=config['PORT'],
        username=config['USERNAME'],
        password=config['PASSWORD'],
        ssl=False)

    # This is the main entry point for the bot.
    with RedisConnectionManager(redis_config) as rcm:
        
        # Multiple bots can be registered
        # rcm.register(randombot.RandomBot("cortex-7LQqyM8"))
        rcm.register(PathFinderBot("cortex-7LQqyM8"))

        # Start listening for Redis messages
        rcm.run()