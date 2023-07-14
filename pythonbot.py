import json
from ast import literal_eval
import redis
import sys

from typing import final

@final
class Map:

    size:int
    height:int
    width:int

    enemyGeneral:int
    ownGeneral:int

    armies:list[int]
    cities:list[int]
    discoveredTiles:list[bool]
    enemyTiles:list[list[int]] #format is bracketed tuples 
    ownTiles:list[list[int]]
    terrain:list[int]
    # swamps:list[any] # unused
    # lights:list[any] # unused

    def __init__(self, game_state_data:dict) -> None:
        self.height = game_state_data['height']
        self.width = game_state_data['width']
        self.size = game_state_data['size']
        
        self.armies = game_state_data['armies']
        self.cities = game_state_data['cities']
        self.discoveredTiles = game_state_data['discoveredTiles']
        self.enemyTiles = game_state_data['enemyTiles']
        self.ownTiles = game_state_data['ownTiles'] # TODO format is '[[175, 1]]'???
        self.terrain = game_state_data['terrain']
        self.ownGeneral = game_state_data['ownGeneral']
        print(f'ownTiles: {game_state_data["ownTiles"]}')


    @classmethod
    class __TILE__:
            EMPTY:int = -1
            MOUNTAIN:int = -2
            FOG:int = -3
            FOG_OBSTACLE:int = -4
            OFF_LIMITS:int = -5

    def is_passable(self, tile:int) -> bool:
        """Returns True if the given tile is passable."""
        return tile not in [Map.__TILE__.MOUNTAIN, 
                            Map.__TILE__.FOG_OBSTACLE, 
                            Map.__TILE__.OFF_LIMITS] and not self.isCity(tile)
    
    def is_city(self, tile:int) -> bool:
        """Returns True if the given tile is a city."""
        return tile in self.cities
    
    def is_enemy(self, tile:int) -> bool:
        """Returns True if the given tile is an enemy tile."""
        return self.terrain[tile] >= 0 and self.terrain[tile] != self.playerIndex

    def get_adjacent_tiles(self,tile:int) -> list[int]:
        """Returns a list of the tiles adjacent to the given tile."""
        adjacent_tiles:list[int] = []
        if tile % self.width != 0:
            adjacent_tiles.append(tile - 1) # right
        if tile % self.width != self.width - 1:
            adjacent_tiles.append(tile + 1) # left
        if tile >= self.width:
            adjacent_tiles.append(tile - self.width) # up
        if tile < self.size - self.width:
            adjacent_tiles.append(tile + self.width) # down
        return adjacent_tiles

    def get_moveable_army_tiles(self) -> list[int]:
        """Returns a list of the tiles that contain armies that can move."""
        return [tile for (tile,strength) in self.ownTiles if strength > 1]
    
    def as_coordinates(self,tile) -> tuple[int,int]:
        """Returns the coordinates of the given tile."""
        return (tile % self.width, tile // self.width)
    
    def as_tile(self,coordinates:tuple[int,int]) -> int:
        """Returns the tile at the given coordinates."""
        return coordinates[0] + coordinates[1] * self.width

@final
class GameInstance:

    playerIndex:int
    playerColors:list[int]
    replay_id:str
    chat_room:str
    usernames:list[str]
    teams:list[int]
    game_type:str
    options:dict
    turn:int
    map:Map
    
    def __init__(self, game_start_message:dict) -> None:
        game_start_message = literal_eval(game_start_message.decode())['game_start']
        print(f"MADE HERE {game_start_message}")
        self.playerIndex= game_start_message['playerIndex']
        self.playerColors = game_start_message['playerColors']
        self.replay_id= game_start_message['replay_id']
        self.chat_room = game_start_message['chat_room']
        self.usernames = game_start_message['usernames']
        self.teams = game_start_message['teams']
        self.game_type = game_start_message['game_type']
        self.options=game_start_message['options']


    def update(self,game_state_byte_data:dict) -> None:
        game_state_data_dict = dict({k.decode():json.loads(v.decode()) for k,v in game_state_byte_data.items()})
        self.turn = game_state_data_dict['turn']
        self.map = Map(game_state_data_dict)

@final
class Action:
    @classmethod
    def serialize(cls,start:int,end:int,is50:bool=False,interrupt:bool=False):
        return json.dumps({'interrupt':interrupt,'actions':[{'start':start, 'end':end, 'is50?':is50}]})

@final
class RedisChannelList:
    TURN:str
    ACTION:str
    RECOMMENDATION:str
    STATE:str

    def __init__(self, bot_id:str):
        self.TURN = f"{bot_id}-turn"
        self.ACTION = f"{bot_id}-action"
        self.RECOMMENDATION = f"{bot_id}-recommendation"
        self.STATE = f"{bot_id}-state"

class PythonBot:
    game:GameInstance
    redis_connetion:redis.Redis
    pubsub:redis.client.PubSub
    bot_id:str
    channel_list:RedisChannelList

    def __init__(self, bot_id:str) -> None:
        self.bot_id = bot_id
        self.channel_list = RedisChannelList(self.bot_id)

    def do_turn():
        raise NotImplementedError("do_turn() must be implemented by the bot subclass.")

    @final
    def move(self, start:int, end:int, is50:bool=False,interrupt=False) -> None:
        self.redis_connetion.publish(self.channel_list.ACTION, Action.serialize(start,end,is50,interrupt))

    @final
    def __handle_turn_channel_message__(self, gameState:GameInstance) -> None:
        self.__update_state__()
        self.do_turn()
    
    @final
    def __update_state__(self):
        update_data = self.redis_connetion.hgetall(f'{self.bot_id}-{self.game.replay_id}')
        self.game.update(update_data)

    @final
    def __is_game_start_message__(self, message: dict) -> bool:
        if message['channel'].decode() == self.channel_list.STATE:
            data = json.loads(message['data'].decode())
            return 'game_start' in data.keys()
        return False

    @final
    def __register_redis__(self,r_conn:redis.Redis, pubsub:redis.Redis.pubsub):
        self.redis_connetion = r_conn

        for channel in [self.channel_list.TURN, self.channel_list.STATE]:
            pubsub.subscribe(channel)

            #First message should always be a subscription confirmation   
            if pubsub.get_message(timeout=None) is None:    
                print(f"Failed to subscribe to {channel} channel. Terminating.", file=sys.stderr)
                exit()
            print(f"Successfully subscribed to {channel} for: {self.bot_id}")

    @final
    def __handle_state_channel_message__(self, message:dict) -> None:
        if self.__is_game_start_message__(message):
            self.game = GameInstance(message['data'])

import os
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

    def register(self, bot:PythonBot):
        bot.__register_redis__(self.redis_connection, self.pubsub)
        self.channel_map[bot.channel_list.TURN] = bot.__handle_turn_channel_message__
        self.channel_map[bot.channel_list.STATE] = bot.__handle_state_channel_message__
