import json
from ast import literal_eval
import redis, sys

from abc import ABCMeta, abstractmethod
from typing import NamedTuple, final, Callable

class TERRAIN_TYPES:
        EMPTY:int = -1
        MOUNTAIN:int = -2
        FOG:int = -3
        FOG_OBSTACLE:int = -4
        OFF_LIMITS:int = -5

class OwnedTile(NamedTuple):
    tile:int
    """index of the tile"""
    strength:int
    """size of the army on the tile"""

@final
class Map:

    """Manages all map-related data for a bot including terrain, armies, cities, etc.
    
    Map objects are updated for each bot before each turn automatically by the framework
    whenever a turn message is received from the server. The map object is accessible
    through the game object, e.g. `game.map`.

    Multiple helper methods are provided to make it easier to work with the map data.

    Provides access to the following data:
    - size (int) : The total number of tiles on the map.
    - height (int) : The height of the map.
    - width (int) : The width of the map.
    - player_index (int) : The bot's player index.
    - enemy_general (int) : The tile index of the enemy general.
    - own_general (int) : The tile index of the bot's general.
    - armies (list[int]) : A list of the all armie strengths on the map. 
    - cities (list[int]) : Each entry is a tile index for a neutral city on the map.
    - discovered_tiles (list[bool]): A list of booleans indicating whether each tile has been discovered.
    - enemy_tiles (list[list[int]]) : A list of lists of the enemy tiles. Each entry is a tuple of the form [tile, strength].
    - own_tiles (list[list[int]]) : A list of lists of the bot's tiles. Each entry is a tuple of the form [tile, strength].
    - terrain (list[int]): represents EITHER the TERRAIN_TYPE or a player index. 

    Usage:
        For a given tile T, if the tile is owned by player  with playerIndex P, then terrain[T]==P, and armies[T] is the army strength S on tile T. If P is your own bot, then there's an item [tile,S] in own_tiles; otherwise, [tile,S] is an item in enemy_tiles.
    """

    size:int
    """The total number of tiles on the map."""
    height:int
    """The height of the map."""
    width:int
    """The width of the map."""
    player_index:int
    """The bot's player index."""

    enemyGeneral:int
    """The tile index of the enemy general."""
    own_general:int
    """The tile index of the bot's general."""
    armies:list[int]
    """A list of the all armie strengths on the map."""
    cities:list[int]
    """Each entry is a tile index for a neutral city on the map."""
    discovered_tiles:list[bool]
    """A list of booleans indicating whether each tile has been discovered."""
    enemy_tiles:list[OwnedTile] #format is bracketed tuples, TODO swap to NamedTuple
    """A list of lists of the enemy tiles. Each entry is a tuple of the form (tile, strength)."""
    own_tiles:list[OwnedTile]
    """A list of lists of the bot's tiles. Each entry is a tuple of the form (tile, strength)."""
    terrain:list[int]
    """represents EITHER the TERRAIN_TYPE or a player index. """

    # swamps:list[any] # unused
    # lights:list[any] # unused

    def __init__(self, game_state_data:dict) -> None:
        self.height = game_state_data['height']
        self.width = game_state_data['width']
        self.size = game_state_data['size']

        self.player_index = game_state_data['playerIndex']
        
        self.armies = game_state_data['armies']
        self.cities = game_state_data['cities']
        self.discovered_tiles = game_state_data['discoveredTiles']
        self.enemy_tiles = [OwnedTile(tile[0],tile[1]) for tile in game_state_data['enemyTiles']]
        self.own_tiles = [OwnedTile(tile[0],tile[1]) for tile in game_state_data['ownTiles']]
        self.terrain = game_state_data['terrain']
        self.own_general = game_state_data['ownGeneral']

    def is_passable(self, tile:int,ignore_cities=True) -> bool:
        """Returns True if the given tile is passable.

        Args:
            tile (int): the tile index to check for passability
            ignore_cities (bool, optional): If the passable check should mark cities as impassible. Defaults to True.

        Returns:
            bool: _description_
        """
        return all([
            tile >= 0 and tile < self.size, # bounds check
            self.terrain[tile] not in [ #terrain check
                            TERRAIN_TYPES.MOUNTAIN, 
                            TERRAIN_TYPES.FOG_OBSTACLE, 
                            TERRAIN_TYPES.OFF_LIMITS],
            not self.is_city(tile) if ignore_cities else True
        ]) 
    
    def is_city(self, tile:int) -> bool:
        """Returns True if the given tile is a city."""
        return tile in self.cities
    
    def is_enemy(self, tile:int) -> bool:
        """Returns True if the given tile is an enemy tile."""
        return self.terrain[tile] >= 0 and self.terrain[tile] != self.playerIndex

    def get_adjacent_tiles(self,tile:int) -> list[int]:
        """Returns a list of the passable tiles adjacent to the given tile."""
        adjacent_tiles:list[int] = []
        if tile % self.width != 0 and self.is_passable(tile - 1):
            adjacent_tiles.append(tile - 1) # right
        if tile % self.width != self.width - 1 and self.is_passable(tile + 1):
            adjacent_tiles.append(tile + 1) # left
        if tile >= self.width and self.is_passable(tile - self.width):
            adjacent_tiles.append(tile - self.width) # up
        if tile < self.size - self.width and self.is_passable(tile + self.width):
            adjacent_tiles.append(tile + self.width) # down
        return adjacent_tiles
    
    def manhattan_distance(self,start:int,end:int) -> int:
        start_coord = self.as_coordinates(start)
        end_coord = self.as_coordinates(end)
        return abs(start_coord[0]-end_coord[0])+abs(start_coord[1]-end_coord[1])
    
    def is_valid_move(self, start:int,target:int) -> bool:
        """checks that the move is at most 1 tile away and the tile is passable"""
        all([
            self.manhattan_distance(start,target) == 1,
            self.is_passable(start),
            self.is_passable(target)
        ])

    def get_moveable_army_tiles(self) -> list[int]:
        """Returns a list of the tiles that contain armies that can move."""
        return [tile for (tile,strength) in self.own_tiles if strength > 1]
    
    def as_coordinates(self,tile:int) -> tuple[int,int]:
        """Returns the coordinates of the given tile."""
        return (tile % self.width, tile // self.width)
    
    def as_tile(self,coordinates:tuple[int,int]) -> int:
        """Returns the tile at the given coordinates."""
        return coordinates[0] + coordinates[1] * self.width

class Move(NamedTuple):
    start:int
    end:int
    is50:bool=False
    interrupt:bool=False

import heapq

class Algorithms:

    def aStar(map:Map, start: int, target: int) -> list[Move]:
        """Finds a path from the start to the target tiles using the A* search algorithm. The path is returned as a list of Move objects (named tuples).

        Heuristic: This method uses manhattan distance as the heuristic estimator which can be replaced by a custom estimator via the aStar_custom_h() method.

        Args:
            map (Map): the game's Map object.
            start (int): index of the start tile
            target (int): index of the target (destination) tile.

        Returns:
            list[Move]: An ordered list of Move objects that connect the start tile to the end tile.

        Note: If no valid path exists, an empty list is returned.
        """
        return Algorithms.aStar_custom_h(map,start,target,map.manhattan_distance)
    
    def aStar_custom_h(map:Map, start: int, target: int, heuristic:Callable[[int,int],int] = lambda s,t:0 ) -> list[Move]:
        """Same as aStar(), but allows for a custom heuristic. See the aStar() documentation for more details."""

        # Initialize priority queue with start node
        queue = [(0, start)]
        
        # Initialize g-costs (from start to node) with infinite values
        g_costs = {node: float('infinity') for node in range(map.size)}
        g_costs[start] = 0
        
        # Initialize f-costs (g-cost + heuristic) with infinite values
        f_costs = {node: float('infinity') for node in range(map.size)}
        f_costs[start] = heuristic(start,target)
        
        # Initialize parents (for path reconstruction)
        parents = {node: None for node in range(map.size)}
        
        while queue:
            # Pop the node with smallest f-cost
            _, curr_node = heapq.heappop(queue)
            
            # If we reached the target, reconstruct and return the path
            if curr_node == target:
                path = []
                while curr_node is not None:
                    path.append(Move(parents[curr_node], curr_node))
                    curr_node = parents[curr_node]
                return path[::-1] # reverse list order
            
            # If not, update the costs of its neighbors
            for neighbor in map.get_adjacent_tiles(curr_node):
                new_g_cost = g_costs[curr_node] + 1
                if new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    f_costs[neighbor] = new_g_cost + heuristic(neighbor, target)
                    parents[neighbor] = curr_node
                    heapq.heappush(queue, (f_costs[neighbor], neighbor))

        print(f'Algorithm.aStar(): No path exists from {start} to {target}',sys.stderr)
        return []  # Return an empty list if there's no path
    
@final
class GameInstance:
    """Created whenever a game starts. Contains all the information about the game that is available to the bot.
    Access the map with game.map, and the current turn with game.turn.

    Args:
        game_start_message (dict): The game_start message from the redis channel.
    """

    player_index:int
    player_colors:list[int]
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
        self.player_index= game_start_message['playerIndex']
        self.player_colors = game_start_message['playerColors']
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

class PythonBot(metaclass=ABCMeta):
    """An abstract base class for Python bots. Subclass this to create a bot. 

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

    @abstractmethod
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
    def __register_redis__(self, r_conn:redis.Redis, pubsub:redis.Redis.pubsub):
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

@final
class RedisConnectionManager:
    """This class will handle all Redis connections and subscriptions for Python bots. 
    Register bots with the register() method. The bot will be notified when a 
    game starts or when a turn is taken.

    Thread Safety: It is recommended to use a separate RedisConnectionManager for each bot instance.

    Args:
        redis_config (dict): a dictinoary used to instantiate a Redis object (cf. params for Redis.__init__). Typically includes keys: host, port, username, password, ssl
    """

    def __init__(self,redis_config:dict) -> None:
            # Connect to Redis
            self.__redis_connection__ = redis.Redis(**(redis_config), client_name="Python_RedisConnectionManager")
            
            # Subscribe to redis channel(s)
            self.__pubsub__ = self.__redis_connection__.pubsub()

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
        """Registers a bot with this connection manager. 
        
        All messages received on the bot's
        channels will be passed to the bot's handler methods. This method also sets the bot's 
        redis connection to the one used by this connection manager; however, a new pubsub 
        instance is set (for thread-safety). Any messages (e.g., move commands) sent by the bot  
        will be sent over the connection hosted by the RedisConnectionManager.

        Args:
            bot (PythonBot): the bot to be registered with this connection manager
        """
        bot.__register_redis__(self.__redis_connection__, self.__redis_connection__.pubsub())
        self.__channel_map__[bot.channel_list.TURN] = bot.__handle_turn_channel_message__
        self.__channel_map__[bot.channel_list.STATE] = bot.__handle_state_channel_message__
    
    __channel_map__:dict[str, callable] = {}

    __redis_connection__:redis.Redis
    __pubsub__:redis.client.PubSub

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.__redis_connection__.close()
