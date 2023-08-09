from ggbot.core import PythonBot, Move, GameInstance,TERRAIN_TYPES
from ggbot.algorithms import aStar
from typing import NamedTuple

class FloBot(PythonBot):

    is_collecting: bool = False
    collect_area: list[int] = []

    is_infiltrating: bool = False
    spread_count:int = 0

    def __init__(self, game_config) -> None:
        super().__init__(game_config)
        pass

    def do_turn(self) -> None:
        Strategy.pick_strategy(self)

class Strategy:

    INITIAL_WAIT_TICKS:int = 23
    REINFORCEMENT_INTERVAL:int = 50
    SPREADING_TIMES:int = 4
    ATTACK_TICKS_BEFORE_REINFORCEMENTS:int = 10
    
    @staticmethod
    def pick_strategy(bot: 'FloBot'):

        if bot.game.enemy_general != -1:
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
        if tick < Strategy.INITIAL_WAIT_TICKS + 1:
            return
        elif tick == Strategy.INITIAL_WAIT_TICKS + 1:
            Discover.expand_territory(bot,Strategy.INITIAL_WAIT_TICKS)
        elif bot.queued_moves == 0:
            Discover.strategic_relocation(bot,Strategy.INITIAL_WAIT_TICKS)

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
                    bot.move(path_to_general[0], path_to_general[1])


class Tile(NamedTuple):
    tile: int
    weight: int

class Heuristics:

    @staticmethod
    def choose_discover_tile(map:GameInstance, passable_tiles):
    
        # Generate a list of tuples containing each tile's index and its distance to the general
        tiles: list[Tile] = [Tile(tile=index, weight=map.manhattan_distance(map.own_general, index)) for index in passable_tiles]
        
        # Sort the tiles by their distance to the general in descending order
        tiles.sort(key=lambda x: x.weight, reverse=True)
        
        # Initialize the optimal tile data
        optimal_tile:Tile = Tile(-1,-1)

        # Extract the maximum general distance from the first element
        max_general_distance = tiles[0].weight

        # Iterate over the tiles
        for tile_index, general_distance in tiles:
            # If the tile's general distance is less than the maximum, we've passed the optimal tile
            if general_distance < max_general_distance:
                return optimal_tile.tile
            # Calculate the tile's edge weight
            edge_weight = Heuristics.__edge_weight_for_index__(map, tile_index)
            # If the tile's edge weight is greater than the optimal tile's, update the optimal tile
            if edge_weight > optimal_tile.weight:
                optimal_tile = Tile(tile=tile_index,weight=edge_weight)

        # If we've gone through all the tiles and found an optimal tile, return it
        if optimal_tile.tile != -1:
            return optimal_tile.tile
        else:
            print("No tile found. Something is going wrong here!")

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
    def choose_enemy_target_tile_by_lowest_army_fog_adjacent(map:GameInstance) -> Tile | None:
        """From among the visible, fog-adjacent enemy tiles, selects the weakest

        Args:
            map (Map): the game map

        Returns:
            int: the enemy tile with the weakest army. Returns None if no fog-adjacent enemy tiles are visible.
        """
        tiles_with_fog: list[Tile] = []

        # loop through all visible enemy tiles 
        for key, value in map.enemy_tiles:
            if Heuristics.is_adjacent_to_fog(map, key):
                tiles_with_fog.append(Tile(key,value))

        if len(tiles_with_fog) == 0:
            return None

        # return tile with lowest army value
        return min(tiles_with_fog, key=lambda t: t.weight)

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
    
        # return any([map.terrain[t] == TERRAIN_TYPES.FOG for t in map.get_adjacent_tiles(tile)])

import heapq
from functools import reduce
class FloAlgorithm:

    def bfs(bot:PythonBot,start_tile:int,radius:int):
        visited = set([start_tile])  # Keep track of visited tiles
        queue = [(start_tile, 0)]  # Use a queue to perform BFS, store tile with its hop count
        tiles_within_k_steps = []

        while queue:
            tile, step_count = queue.pop(0)

            if step_count <= radius:
                tiles_within_k_steps.append(tile)

                if step_count < radius:  # Only get neighbors if current hop count is less than k
                    for neighbor in bot.game.get_adjacent_tiles(tile):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, step_count + 1))

        return tiles_within_k_steps
    
    def dijkstra(bot:PythonBot, start: int, target: int) -> list[int]:
        map = bot.game

        # Initialize priority queue with start node
        queue = [(0, start)]
        
        # Initialize distances with infinite values
        distances = {node: float('infinity') for node in range(map.size)}
        distances[start] = 0
        
        # Initialize parents (for path reconstruction)
        parents = {node: None for node in range(map.size)}
        
        while queue:
            # Pop the node with smallest distance
            curr_distance, curr_node = heapq.heappop(queue)
            
            # If we reached the target, reconstruct and return the path
            if curr_node == target:
                path = []
                while curr_node is not None:
                    path.append(curr_node)
                    curr_node = parents[curr_node]
                return path[::-1]
            
            # If not, update the distances of its neighbors
            for neighbor in map.get_adjacent_tiles(curr_node):
                new_distance = curr_distance + 1
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = curr_node
                    heapq.heappush(queue, (new_distance, neighbor))
                    
        return []  # Return an empty list if there's no path
    
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

        def dec_tree_search_rec( start:int, ticks:int, weight:int = 0):
            possible_moves = []

            if ticks != 0:
                adj_tiles = map.get_adjacent_tiles(start)
                for next_tile in adj_tiles:
                    next_weight = Heuristics.calc_capture_weight(player_index,map.terrain[next_tile])
                    possible_moves.append(dec_tree_search_rec(next_tile,ticks-1,next_weight))

                # try waiting a tick without moving
                possible_moves.append(dec_tree_search_rec(start,ticks-1,0))

            if len(possible_moves) == 0:
                return {"start":start,"end":-1,"weight":weight}
            else:
                best_path = get_best_move(possible_moves)
                return {"start":start,"end":best_path['start'],"weight":weight+best_path['weight']}

        def get_best_move(moves):
            return reduce((lambda prev,cur: prev if prev['weight'] > cur['weight'] else cur),moves)

        moves = []

        for start in possible_starting_points:
            moves.append(dec_tree_search_rec(start,max_ticks))

        best = get_best_move(moves)
        return Move(best['start'],best['end'])
    

class Collect:
    
    @staticmethod
    def get_collect_area(bot: 'FloBot') -> list[int]:
        map = bot.game
        
        bot.is_collecting = True

        # enemy tile found
        if map.enemy_tiles:
            enemy_target = Heuristics.choose_enemy_target_tile_by_lowest_army_fog_adjacent(map)
            if enemy_target is not None:
                path_to_enemy = aStar(map, map.own_general, [enemy_target.tile])
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
    def expand_territory(bot:PythonBot, wait_ticks):
        """Referred to as 'first' in flobot"""
        radius = Discover.armies_received_till_tick(wait_ticks + 1)
        reachable_tiles = FloAlgorithm.bfs(bot,start_tile=bot.game.own_general, radius=radius)
        discover_tile = Heuristics.choose_discover_tile(bot.game, reachable_tiles)

        moves = FloAlgorithm.dijkstra(bot, start=bot.game.own_general, target=discover_tile)

        start = bot.game.own_general
        for move in moves:
            bot.move(start = start, end = move,caller='expand_territory')
            start = move

    @staticmethod
    def strategic_relocation(bot:PythonBot, wait_ticks):
        """Referred to as 'second' in flobot"""
        ticks = ceil((wait_ticks + 1) / 4)
        moveable_tiles = bot.game.get_moveable_army_tiles()
        if moveable_tiles:
            move = FloAlgorithm.dec_tree_search(bot.game.player_index,bot.game, moveable_tiles, ticks)
            moves = aStar(bot.game,start=move.start,targets=[move.end])

            moves = Move.ints_as_moves(moves)

            for move in moves:
                bot.move(move.start,move.end,caller='strategic_relocation')

    @staticmethod
    def armies_received_till_tick(tick):
        return (tick / 2) + 1

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
                path: list[Move] = Infiltrate.get_path_to_next_tile(bot, attack_source)
                if path: 
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
    def get_path_to_next_tile(bot: 'FloBot', start: int) -> list[Move]:
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
                moves = aStar(bot.game ,bot.game.own_general,[bot.game.enemy_general])
                moves = Move.ints_as_moves(moves)

                bot.collect_area = [move.start for move in moves]

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
        path_from_highest_army_to_general = Move.ints_as_moves(path_from_highest_army_to_general)


        # if length would be 1 there is only the general left to attack, but there aren't enough armies to kill him
        if len(path_from_highest_army_to_general) > 1: # using 1 (vice 2) since our Move encapsulates start AND end, whereas javascript used list[int]
            bot.move(start,path_from_highest_army_to_general.pop().end,caller='move_to_general')
        else:
            RushGeneral.collect_ticks_left = RushGeneral.COLLECT_TICKS

    @staticmethod
    def try_to_kill_general(bot: 'FloBot'):
        """If player is adj to enemy general and have enough armies, attack and return True. Else return False"""
        adj_tiles = bot.game.get_adjacent_tiles(bot.game.enemy_general)
        attackable_neighbours = []

        for next_tile in adj_tiles:
            if bot.game.terrain[next_tile] == bot.game.player_index:
                if RushGeneral.has_enough_armies_to_attack_general(bot, next_tile):
                    bot.move(next_tile, bot.game.enemy_general)
                    bot.is_infiltrating = False
                    return True
                elif bot.game.armies[next_tile] > 0:
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
            bot.move(highest_army_tile,bot.game.enemy_general)
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

                bot.move(cur_node.tile, chosen_tile)
                Spread.remove_already_occupied_tiles(possible_moves, chosen_tile)

                possible_moves.sort(reverse=True, key = lambda n: len(n.moves))

    @staticmethod
    def remove_already_occupied_tiles(possible_moves: list[__SpreadNode__], chosen_tile: int):
        """removes `tile` from move list of each Node in the list of starting tiles"""
        for n in possible_moves:
            n.moves = [v for v in n.moves if v != chosen_tile]


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
        rcm.register(FloBot(config['gameConfig']))

        # Start listening for Redis messages
        rcm.run()
