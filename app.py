import randombot
from pythonbot import RedisConnectionManager
import os,json

if __name__ == "__main__":

    # Using the same config as other CorsairCoalition components
    redis_config_file = os.path.dirname(__file__) + './config.json'
    
    config:dict = json.load(open(redis_config_file))['redisConfig']
    
    redis_config = dict(
        host=config['HOST'],
        port=config['PORT'],
        username=config['USERNAME'],
        password=config['PASSWORD'],
        ssl=False)

    # This is the main entry point for the bot.
    with RedisConnectionManager(redis_config) as rcm:
        
        # Multiple bots can be registered
        rcm.register(randombot.RandomBot("cortex-7LQqyM8"))
        
        # Start listening for Redis messages
        rcm.run()