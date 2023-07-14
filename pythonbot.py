import json
from ast import literal_eval
import redis
import sys

from typing import final

@final
class Map:

    """Manages all map-related data for a bot including terrain, armies, cities, etc.
    Map objects are updated for each bot before each turn automatically by the framework
    whenever a turn message is received from the server. The map object is accessible
    through the game object, e.g. `game.map`.

    Multiple helper methods are provided to make it easier to work with the map data.

    Provides access to the following data:
    - size: The total number of tiles on the map.
    - height: The height of the map.
    - width: The width of the map.
    - enemyGeneral: The tile index of the enemy general.
    - ownGeneral: The tile index of the bot's general.
    - armies: A list of the armies on the map. Each entry is a tuple of the form (tile, strength).
    - cities: A list of the cities on the map. Each entry is a tile index.
    - discoveredTiles: A list of booleans indicating whether each tile has been discovered.
    - enemyTiles: A list of lists of the enemy tiles. Each entry is a tuple of the form [tile, strength].
    - ownTiles: A list of lists of the bot's tiles. Each entry is a list of tile indices.
    - terrain: A list of the terrain on the map. Each entry is a tile index of the type __TILE__. 
    """

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
    """Created whenever a game starts. Contains all the information about the game that is available to the bot.
    Access the map with game.map, and the current turn with game.turn.

    Args:
        game_start_message (dict): The game_start message from the redis channel.
    """

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
        self.playerIndex= game_start_message['playerIndex']
        self.playerColors = game_start_message['playerColors']
        self.replay_id= game_start_message['replay_id']
        self.chat_room = game_start_message['chat_room']
        self.usernames = game_start_message['usernames']
        self.teams = game_start_message['teams']
        self.game_type = game_start_message['game_type']
        self.options=game_start_message['options']

    def update(self,game_state_data:dict) -> None:
        """
        Updates the game instance with the given game state data.
        """
        self.turn = game_state_data['turn']
        self.map = Map(game_state_data)

@final
class Action:
    @classmethod
    def serialize(cls,start:int,end:int,is50:bool=False,interrupt:bool=False):
        """Serializes a move for the game engine.

        Args:
            start (int): the tile to move from
            end (int): the tile to move to
            is50 (bool, optional): If true, only half of the army will be moved. Defaults to False.
            interrupt (bool, optional): If True, the server-side movement queue will be cleared. Defaults to False.

        Returns:
            str: json representation of the movement action suitable to be passed to Redis
        """
        return json.dumps({'interrupt':interrupt,'actions':[{'start':start, 'end':end, 'is50?':is50}]})

@final
class RedisChannelList:
    """A data class to hold the redis channel names for a bot
    """
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
    """A base class for python bots. Subclass this to create a bot. 

    Subclasses must implement the do_turn() method.

    Raises:
        NotImplementedError: if the do_turn() method is not implemented

    Args:
        bot_id: a unique identifier for the bot. Must be unique across all bots.
    """

    game:GameInstance
    redis_connetion:redis.Redis
    pubsub:redis.client.PubSub
    bot_id:str
    channel_list:RedisChannelList

    def __init__(self, bot_id:str) -> None:
        self.bot_id = bot_id
        self.channel_list = RedisChannelList(self.bot_id)

    def do_turn():
        """Called every turn. Must be implemented by the bot subclass."""
        raise NotImplementedError("do_turn() must be implemented by the bot subclass.")

    @final
    def move(self, start:int, end:int, is50:bool=False,interrupt=False) -> None:
        """
        Moves armies from one tile to another.

        Args:
            start (int): the tile to move from
            end (int): the tile to move to
            is50 (bool, optional): If true, only half of the army will be moved. Defaults to False.
            interrupt (bool, optional): If True, the server-side movement queue will be cleared. Defaults to False.
        """
        self.redis_connetion.publish(self.channel_list.ACTION, Action.serialize(start,end,is50,interrupt))

    @final
    def __handle_turn_channel_message__(self, gameState:GameInstance) -> None:
        self.__update_state__()
        self.do_turn()
    
    @final
    def __update_state__(self):
        update_data = self.redis_connetion.hgetall(f'{self.bot_id}-{self.game.replay_id}')
        game_state_data_dict = dict({k.decode():json.loads(v.decode()) for k,v in update_data.items()})
        self.game.update(game_state_data_dict)

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

class RedisConnectionManager:
    """This class will handle all Redis connections and subscriptions for Python bots. 
    Register bots with the register() method. Individual bots will be notified when a 
    game starts or when a turn is taken.

    Args:
        redis_config_file (str, optional): The path to the Redis config file. Defaults to None.
    """
    __channel_map__:dict[str, callable] = {}

    __redis_connection__:redis.Redis
    __pubsub__:redis.client.PubSub

    def __init__(self,redis_config:dict) -> None:
        # Connect to Redis
        self.__redis_connection__ = redis.Redis(**(redis_config), client_name="Python_RedisConnectionManager")
        
        # Subscribe to redis channel(s)
        self.__pubsub__ = self.__redis_connection__.pubsub()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.__redis_connection__.close()

    def run(self):
        """
        This method will start listening for Redis messages. When a message is received,
        the appropriate bot will be notified, and the message will be passed to the bot's
        handler method. If it is a turn, the bot's do_turn() method will be called. This 
        method will block until the connection is closed.
        """

        # Start listening for Redis messages
        for message in self.__pubsub__.listen():
            try:
                channel:str = message['channel'].decode()
                self.__channel_map__[channel](message)
            except KeyError:
                print(f"Received unknown message type: {message}", file=sys.stderr)

    def register(self, bot:PythonBot):
        """Registers the bot with this connection manager. All messages received on the bot's
        channels will be passed to the bot's handler methods. 
        This method also sets the bot's redis connection and pubsub objects to the ones used
        by this connection manager; any messages sent by the bot (e.g., move commands) will 
        be sent over this connection.

        Args:
            bot (PythonBot): the bot to be registered with this connection manager
        """
        bot.__register_redis__(self.__redis_connection__, self.__pubsub__)
        self.__channel_map__[bot.channel_list.TURN] = bot.__handle_turn_channel_message__
        self.__channel_map__[bot.channel_list.STATE] = bot.__handle_state_channel_message__
