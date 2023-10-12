from ggbot.core import PythonBot, GameInstance,TERRAIN_TYPES
import ggbot.utils
from ggbot.algorithms import aStar
from typing import NamedTuple

class FloBot(PythonBot):

    is_collecting: bool = False
    collect_area: list[int] = []

    is_infiltrating: bool = False
    spread_count:int = 0

    def do_turn(self) -> None:
        Strategy.pick_strategy(self)

class Strategy:

    INITIAL_WAIT_TICKS:int = 23
    REINFORCEMENT_INTERVAL:int = 50
    SPREADING_TIMES:int = 4
    ATTACK_TICKS_BEFORE_REINFORCEMENTS:int = 10
    
    @staticmethod
    def pick_strategy(bot: 'FloBot'):

        if bot.game.enemy_general != -1: # enemy general is visible!
            Strategy.end_game(bot) 
        elif bot.is_infiltrating:
            Infiltrate.infiltrate(bot)
        elif Strategy.is_spread_needed(bot):
            Spread.spread(bot)
        elif bot.game.tick < Strategy.REINFORCEMENT_INTERVAL:
            Strategy.early_game(bot, bot.game.tick)
        else:
            Strategy.mid_game(bot, bot.game.tick)

    @staticmethod
    def is_spread_needed(bot:'FloBot'):
        # spread every 50 ticks, but only a fixed amount of times, unless no enemies are detected
        is_reinforcement_tick = bot.game.tick % Strategy.REINFORCEMENT_INTERVAL == 0
        spreads_allowable = bot.game.tick / Strategy.REINFORCEMENT_INTERVAL <= Strategy.SPREADING_TIMES
        enemy_not_detected = not bot.game.enemy_tiles

        return is_reinforcement_tick and (spreads_allowable or enemy_not_detected)
    
    @staticmethod
    def early_game(bot: 'FloBot', tick: int):
        if tick <= Strategy.INITIAL_WAIT_TICKS:
            return
        elif tick == Strategy.INITIAL_WAIT_TICKS + 1:
            Discover.first_discover_branch(bot, Strategy.INITIAL_WAIT_TICKS)
        elif bot.queued_moves == 0:
            Discover.second_dicover_branch(bot, Strategy.INITIAL_WAIT_TICKS)

    @staticmethod
    def mid_game(bot: 'FloBot', tick: int):

        # Enemy tile was detected and a path was found. Check further if attack should start already
        # collectArea of 1 means there's no end node to attack anymore
        if bot.game.enemy_tiles and len(bot.collect_area) > 1 and (tick + Strategy.ATTACK_TICKS_BEFORE_REINFORCEMENTS + len(bot.collect_area) - 1) % Strategy.REINFORCEMENT_INTERVAL == 0:
            # reinforcements from general along the found path should be there a fixed amount of ticks before new reinforcements come
            # tick the "attack" should start from general
            # e.g., path is 7 long, atk_ticks_before_reinforcements are 10. 34+10+7 -1 %50 = 0 (start at tick 34 to arrive at tick 40)
            
            if len(bot.collect_area) == 2:
                # gathered units moved next to enemy tile. start to attack
                # infilstrating is True until no adj enemies to attack found. Focus moves on them
                bot.is_infiltrating = True
            
            start = bot.collect_area.pop(0)
            end = bot.collect_area[0]
            bot.move(start,end,caller='mid_game')
        elif not bot.is_infiltrating:
            # bot is not moving to enemy tile and isn't infiltrating => collect
            bot.collect_area = Collect.get_collect_area(bot)
            if bot.queued_moves == 0:
                Collect.collect(bot)

    @staticmethod
    def end_game(bot: 'FloBot'):
        if not bot.is_infiltrating:
            RushGeneral.rush(bot)
        else:
            # try_to_kill clears infiltrating flag
            if not RushGeneral.try_to_kill_general(bot):
                # finish infiltrating first (enemy can be discovered diagonally. Move to adj tile first)
                path_to_general = aStar(bot.game,bot.last_attacked_tile,[bot.game.enemy_general])

                if len(path_to_general) < 2 or bot.game.remaining_armies_after_attack(path_to_general[0], path_to_general[1]) <= 1:
                    bot.is_infiltrating = False

                if len(path_to_general) > 2:
                    bot.move(path_to_general[0], path_to_general[1], caller='end_game')

class Tile(NamedTuple):
    tile: int
    weight: int

class Heuristics:

    @staticmethod
    def choose_discover_tile(map:GameInstance, tiles:list[tuple[int,int]]):
        """Returns the furthest possible tile index from the general with maximum distance to edge
        Passable_tiles is an array of tuples of the form (tile index, distance to start tile)"""

        optimal_tile = (-1,-1) # (index, edge_weight) Careful! This is different from passable_tiles tuple format

        max_general_distance = tiles[len(tiles) - 1][1]

        # first elements are the closest to the general
        for tile in reversed(tiles): # note reversed order!
            edge_weight = Heuristics.__edge_weight_for_index__(map, tile[0])

            # general distance is not at maximum anymore. ignore other tiles
            if tile[1] < max_general_distance:
                return optimal_tile[0]

            # a tile with maximum general_distance and
            if edge_weight > optimal_tile[1]:
                optimal_tile = (tile[0], edge_weight)

        # loop stopped but optimal tile was found(meaning it was only 1 step away from general)
        if optimal_tile[0] != -1:
            return optimal_tile[0]
        else:
            print(f'No tile found. Something is going wrong at choose_discover_tile!')


    @staticmethod
    def __edge_weight_for_index__(map:GameInstance, index:int):
            # Get tile coordinates from index
            y, x = divmod(index, map.width)

            # Calculate distances to the map edges
            upper_edge = y
            right_edge = map.width - 1 - x
            down_edge = map.height - 1 - y
            left_edge = x

            # Return calculated edge weight
            return min(upper_edge, down_edge) * min(left_edge, right_edge)

    @staticmethod
    def choose_enemy_target_tile_by_lowest_army_fog_adjacent(map:GameInstance) -> tuple[int,int] | None:
        """From among the visible, fog-adjacent enemy tiles, selects the weakest

        Args:
            map (Map): the game map

        Returns:
            int: the enemy tile with the weakest army. Returns None if no fog-adjacent enemy tiles are visible.
        """
        tiles_with_fog: list[tuple[int,int]] = []

        # loop through all visible enemy tiles 
        for key, value in map.enemy_tiles:
            if Heuristics.is_adjacent_to_fog(map, key):
                tiles_with_fog.append((key,value))

        if len(tiles_with_fog) == 0:
            return None

        # return tile with lowest army value
        return min(tiles_with_fog, key=lambda t: t[1])

    @staticmethod
    def calc_capture_weight(player_index:int, terrain_value:int):
        # terrain must be walkable
        # 0 if it belongs to himself, 1 for empty and 3 for enemy tile
        if terrain_value == player_index:
            return 0
        elif terrain_value == TERRAIN_TYPES.EMPTY or terrain_value == TERRAIN_TYPES.FOG:
            return 1
        elif terrain_value <= 0:
            # tile belongs to enemy
            return 3
            
    def is_adjacent_to_fog(map:GameInstance, tile:int):
        adj_tiles = map.get_adjacent_tiles(tile)

        for next_tile in adj_tiles:
            if not map.discovered_tiles[next_tile]:
                return True
        return False

from functools import reduce
class FloAlgorithm:

    def bfs(bot:PythonBot, start_tile:int, radius:int) -> list[tuple[int,int]]:
        """Returns all reachable tiles within a given radius. Return format is a list of tuples, where the first value is the tile index and the second value is the distance from the start tile."""
        map = bot.game
        is_visited = [False] * map.size
        is_visited[start_tile] = True

        queue = [start_tile]
        cur_layer = 0
        cur_layer_tiles = 1
        next_layer_tiles = 0
        found_nodes = []

        while queue:
            cur_tile = queue.pop(0)

            # don't add starting node
            if cur_layer != 0:
                found_nodes.append((cur_tile,cur_layer))
            
            adj_tiles = map.get_adjacent_tiles(cur_tile)

            for next_tile in adj_tiles:
                if not is_visited[next_tile]:
                    # tile can be moved on (ignoring cities)
                    queue.append(next_tile)
                    is_visited[next_tile] = True
                    next_layer_tiles += 1
            
            # check if all tiles of current depth are already visited
            cur_layer_tiles -=1
            if cur_layer_tiles == 0:
                cur_layer += 1
                
                if cur_layer < radius:
                    cur_layer_tiles = next_layer_tiles
                    next_layer_tiles = 0
        return found_nodes
    
    def dijkstra(bot:PythonBot, start: int, target: int) -> list[int]:
        """"returns shortest path (as array) between start and end index without considering node weigths"""
        map = bot.game

        is_visited = [False] * map.size
        previous = [i for i in range(map.size)]

        previous[start] = -1

        queue = [start]


        while queue:
            cur_tile = queue.pop(0)
            is_visited[cur_tile] = True

            adj_tiles = map.get_adjacent_tiles(cur_tile)

            for next_tile in adj_tiles:
                if not is_visited[next_tile] and not cur_tile in queue:
                    previous[next_tile] = cur_tile
                    if next_tile == target:
                        return FloAlgorithm.__construct_dijkstra_path__(start, target, previous)
                    queue.append(next_tile)

        print(f'Dijkstra found no path! start: {start} end: {target}')
        return []
    
    def __construct_dijkstra_path__(start:int, end:int, previous: list[int]):
        
        path = [end]
        prev_index = previous[end]
        
        # start node has -1 as previous
        while prev_index != -1:
            # build the path backwards, from end to start
            path.append(prev_index)
            prev_index = previous[prev_index]

        return path[::-1] # reverse the path 
    
    def dec_tree_search(player_index:int, map:GameInstance, possible_starting_points:list[int], max_ticks:int):
        """
        Performs a limited depth-first search (DFS) from each of 
        the `starting_points` and determines the best move based 
        on the calculated weights of potential moves.

        Args:
            player_index (int): index of own bot
            map (Map): the game map
            possible_starting_points (list[int]): list of tiles which could serve as the starting point
            max_ticks (int): dfs depth limit
        """

        moves = []

        for start in possible_starting_points:
            moves.append(FloAlgorithm.dec_tree_search_rec(player_index, map, start, max_ticks))

        best = FloAlgorithm.get_best_move(moves)
        return best['start'], best['end']
    
    def dec_tree_search_rec(player_index:int, map:GameInstance, start:int, ticks:int, weight:int = 0):
            possible_moves = []

            if ticks != 0:
                adj_tiles = map.get_adjacent_tiles(start)
                for next_tile in adj_tiles:
                    next_weight = Heuristics.calc_capture_weight(player_index,map.terrain[next_tile])
                    possible_moves.append(FloAlgorithm.dec_tree_search_rec(player_index, map, next_tile, ticks-1, next_weight))

                # try waiting a tick without moving
                possible_moves.append(FloAlgorithm.dec_tree_search_rec(player_index, map, start, ticks-1, 0))

            if len(possible_moves) == 0:
                return {"start":start,"end":-1,"weight":weight}
            elif len(possible_moves) == 1:
                best_path = possible_moves.pop(0)
                return {"start":start, "end":best_path['start'], "weight": weight + best_path['weight']}
            else:
                best_path = FloAlgorithm.get_best_move(possible_moves)
                return {"start":start, "end":best_path['start'], "weight": weight + best_path['weight']}

    def get_best_move(moves):
        return reduce((lambda prev, cur: prev if prev['weight'] > cur['weight'] else cur), moves)
    

class Collect:
    
    @staticmethod
    def get_collect_area(bot: 'FloBot') -> list[int]:
        map = bot.game
        
        bot.is_collecting = True

        # enemy tile found
        if map.enemy_tiles:
            enemy_target = Heuristics.choose_enemy_target_tile_by_lowest_army_fog_adjacent(map)
            if enemy_target is not None:
                path_to_enemy = aStar(map, map.own_general, [enemy_target[0]])
                return path_to_enemy
        
        # not enemy found, gather on own_general
        return [map.own_general] 
    
    @staticmethod
    def collect(bot: 'FloBot'):
        highest_army_index = Collect.get_highest_army_index(bot.game.own_tiles, bot.collect_area)
        if highest_army_index == -1:
            # skip collecting, no tiles found
            bot.is_collecting = False
        else:
            path_to_attacking_path = aStar(bot.game, highest_army_index, bot.collect_area)
            if len(path_to_attacking_path) > 1:
                bot.move(highest_army_index, path_to_attacking_path[1], caller='collect')

    @staticmethod
    def get_highest_army_index(tiles: list[tuple], path: list[int]):
        tile = -1
        armies = 0
        for key, value in tiles:
            if value > armies and value > 1 and key not in path:
                tile = key
                armies = value
        
        return tile

from math import ceil
class Discover:

    @staticmethod
    def first_discover_branch(bot:PythonBot, wait_ticks):
        """Referred to as 'first' in flobot. Discovers new tiles toward the center"""
        radius = Discover.armies_received_till_tick(wait_ticks + 1)
        reachable_tiles = FloAlgorithm.bfs(bot, start_tile=bot.game.own_general, radius=radius)
        discover_tile = Heuristics.choose_discover_tile(bot.game, reachable_tiles)

        # moves = FloAlgorithm.dijkstra(bot, start=bot.game.own_general, target=discover_tile)
        moves = aStar(bot.game, bot.game.own_general, targets=[discover_tile])

        bot.queue_moves(moves, caller='first_discover_branch')


    @staticmethod
    def second_dicover_branch(bot:PythonBot, wait_ticks):
        """Referred to as 'second' in flobot. takes as many tiles as possible until reinforcements come"""
        ticks = ceil((wait_ticks + 1) / 2 / 2 )
        moveable_tiles = bot.game.get_moveable_army_tiles()
        
        if moveable_tiles:
            start, end = FloAlgorithm.dec_tree_search(bot.game.player_index, bot.game, moveable_tiles, ticks)

            bot.move(start, end, caller='second_dicover_branch')

    @staticmethod
    def armies_received_till_tick(tick):
        return (tick / 2 ) + 1

class Infiltrate:
    
    @staticmethod
    def infiltrate(bot: 'FloBot'):
        enemy_neighbour = -1

        if Infiltrate.last_attacked_tile_is_valid(bot):
            attack_source = bot.last_attacked_tile
            adj_tiles = bot.game.get_adjacent_tiles(attack_source)

            for next_tile in adj_tiles:

                if all([
                        bot.game.is_enemy(next_tile),
                        bot.game.is_passable(next_tile),
                        Heuristics.is_adjacent_to_fog(bot.game, next_tile)
                        ]):
                    
                    if enemy_neighbour == -1 or bot.game.armies[next_tile] < bot.game.armies[enemy_neighbour]:
                        enemy_neighbour = next_tile
            
            start = attack_source
            end = -1

            if enemy_neighbour != -1:
                end = enemy_neighbour
            else:
                # no adj enemy tile found, that could lead to enemy general search for nearest
                path: list[int] = Infiltrate.get_path_to_next_tile(bot, attack_source)
                if len(path) > 1: 
                    # path gets recalculated every move/infiltrate call
                    end = path[1]
            
            if end == -1 or bot.game.remaining_armies_after_attack(start, end) <= 1:
                bot.is_infiltrating = False

            if end != -1 and bot.game.remaining_armies_after_attack(start, end) >= 1:
                bot.move(start,end,caller='infiltrate')
        else:
            bot.is_infiltrating = False

    @staticmethod
    def last_attacked_tile_is_valid(bot: PythonBot) -> bool:
        return bot.last_attacked_tile != -1 and bot.game.terrain[bot.last_attacked_tile] == bot.game.player_index
    
    @staticmethod
    def get_path_to_next_tile(bot: 'FloBot', start: int) -> list[int]:
        """gets nearest tile to start which is adjacent to fog"""
        tiles_with_fog: list[int] = []
        for t, s in bot.game.enemy_tiles:
            if Heuristics.is_adjacent_to_fog(bot.game, t):
                tiles_with_fog.append(t)

        
        shortest_path = aStar(bot.game, start, tiles_with_fog)
        
        return shortest_path

class RushGeneral:
    
    COLLECT_TICKS: int = 20
    collect_ticks_left = COLLECT_TICKS

    @staticmethod
    def rush(bot:'FloBot'):
        
        if not RushGeneral.try_to_kill_general(bot):
            if RushGeneral.collect_ticks_left > 0:
                # collect units along the path toward the enemy general
                bot.collect_area = aStar(bot.game ,bot.game.own_general,[bot.game.enemy_general])
                if bot.collect_area:
                    bot.collect_area.pop()
                Collect.collect(bot)
                RushGeneral.collect_ticks_left -= 1
            elif RushGeneral.collect_ticks_left == 0:
                RushGeneral.move_to_general(bot,bot.game.own_general)
                RushGeneral.collect_ticks_left = -1
            else:
                RushGeneral.move_to_general(bot, bot.last_attacked_tile)
    
    @staticmethod
    def move_to_general(bot: 'FloBot', start: int):
        path_from_highest_army_to_general = aStar(bot.game,start,[bot.game.enemy_general])

        # if length would be 2 there is only the general left to attack, but there aren't enough armies to kill him
        if len(path_from_highest_army_to_general) > 2:
            bot.move(start, path_from_highest_army_to_general[1], caller='move_to_general')
        else:
            RushGeneral.collect_ticks_left = RushGeneral.COLLECT_TICKS

    @staticmethod
    def try_to_kill_general(bot: PythonBot):
        """If player is adj to enemy general and has enough armies, attack and return True. Else return False"""
        adj_tiles = bot.game.get_adjacent_tiles(bot.game.enemy_general)
        attackable_neighbours = []

        for next_tile in adj_tiles:
            if bot.game.terrain[next_tile] == bot.game.player_index:
                if RushGeneral.has_enough_armies_to_attack_general(bot, next_tile):
                    bot.move(next_tile, bot.game.enemy_general, caller='try_to_kill_general')
                    bot.is_infiltrating = False
                    return True
                elif bot.game.armies[next_tile] > 1:
                    attackable_neighbours.append(next_tile)
        
        if len(attackable_neighbours) > 1:
            return RushGeneral.try_group_attack(bot, attackable_neighbours)
        
        return False
    
    @staticmethod
    def try_group_attack(bot: 'FloBot', attackable_neighbours:list[int]) -> bool:
        highest_army = -1
        highest_army_tile = -1
        attackable_army_sum = 0
        for neighbour in attackable_neighbours:
            armies = bot.game.armies[neighbour]
            attackable_army_sum += armies -1
            if armies > highest_army:
                highest_army = armies
                highest_army_tile = neighbour
        
        if attackable_army_sum > bot.game.armies[bot.game.enemy_general]:
            bot.move(highest_army_tile,bot.game.enemy_general, caller='try_group_attack')
            return True

        return False
    
    @staticmethod
    def has_enough_armies_to_attack_general(bot: 'FloBot', target_tile: int) -> bool:
        """Checks if own armies on the given tile are strong enough to conquer the enemy general on the next tick.
        Performs a check account for defender general's additional army gained on even ticks """
        next_tick_army_gain = 0
        
        #generals get an extra army every even tick
        if bot.game.tick % 2 != 0:
            next_tick_army_gain = 1

        return (bot.game.armies[target_tile] -1) > (bot.game.armies[bot.game.enemy_general] + next_tick_army_gain)
   
class Spread:

    class __SpreadNode__():
        tile: int
        moves: list[int]

        def __init__(self,tile:int, moves:list[int]):
            self.tile = tile
            self.moves = moves

        def __lt__(self, value:'Spread.__SpreadNode__'):
            return len(self.moves) < len(value.moves)
    
    @staticmethod
    def spread(bot: 'FloBot'):
        map = bot.game

        moveable_tiles = map.get_moveable_army_tiles()
        possible_moves: list[Spread.__SpreadNode__] = []

        for tile in moveable_tiles:
            adj_tiles = map.get_adjacent_tiles(tile)
            neighbour_moves = []

            for next_tile in adj_tiles:
                if map.terrain[next_tile] == TERRAIN_TYPES.EMPTY and not map.is_city(next_tile):
                    neighbour_moves.append(next_tile)
            
            if neighbour_moves:
                possible_moves.append(Spread.__SpreadNode__(tile,neighbour_moves))
        
        # sort from most neighbours to least
        possible_moves.sort(reverse=True, key = lambda n : len(n.moves))

        while possible_moves:

            cur_node = possible_moves.pop() #get move w/ most neighbours

            if len(cur_node.moves) >= 1:
                chosen_tile = cur_node.moves.pop()

                bot.move(cur_node.tile, chosen_tile, caller='spread')
                Spread.remove_already_occupied_tiles(possible_moves, chosen_tile)

                possible_moves.sort(reverse=True, key = lambda n: len(n.moves))

    @staticmethod
    def remove_already_occupied_tiles(possible_moves: list[__SpreadNode__], chosen_tile: int):
        """removes `tile` from move list of each Node in the list of starting tiles"""
        for n in possible_moves:
            n.moves = [v for v in n.moves if v != chosen_tile]


if __name__ == "__main__":

    #config = ggbot.utils.get_config_from_file("../config.json")
    config = ggbot.utils.get_config_from_cmdline_args()    
    
    FloBot().with_config(config).run()
