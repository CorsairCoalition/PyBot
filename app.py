import redis
import json, os, sys
import pythonbot

class RedisConnectionManager:

    channel_map:dict[str, callable] = {}

    # Path to the Redis config file
    redis_config_file:str = os.path.dirname(__file__) + '/../config.json'
    redis_connection:redis.Redis
    pubsub:redis.client.PubSub

    def __init__(self):
        config_file = json.load(open(self.redis_config_file))['redisConfig']
        
        self.redis_config = dict(
            host=config_file['HOST'],
            port=config_file['PORT'],
            username=config_file['USERNAME'],
            password=config_file['PASSWORD'],
            ssl=False)
        
        # Connect to Redis
        self.redis_connection = redis.Redis(**(self.redis_config), client_name="Python_RedisConnectionManager")
        
        # Subscribe to redis channel(s)
        self.pubsub = self.redis_connection.pubsub()

    def run(self):

        # Start listening for Redis messages
        for message in self.pubsub.listen():
            try:
                channel:str = message['channel'].decode()
                self.channel_map[channel](message)
            except KeyError:
                print(f"Received unknown message type: {message}", file=sys.stderr)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.redis_connection.close()

    def register(self, bot:pythonbot.PythonBot):
        bot.__register_redis__(self.redis_connection, self.pubsub)
        self.channel_map[bot.channel_list.TURN] = bot.__handle_turn_channel_message__
        self.channel_map[bot.channel_list.STATE] = bot.__handle_state_channel_message__

import randombot
if __name__ == "__main__":
    with RedisConnectionManager() as rcm:
        rcm.register(randombot.RandomBot("cortex-7LQqyM8"))
        rcm.run()