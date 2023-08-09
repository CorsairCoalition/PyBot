from pythonbot.core import GameInstance, TERRAIN_TYPES
import heapq

def aStar(map: 'GameInstance', start: int, targets: list[int]) -> list[int]:
    """Finds a path from the start to the nearest tile in targets using the A* search algorithm. The path is returned as a list of ints.

    Heuristic: This method uses manhattan distance as the heuristic estimator which can be replaced by a custom estimator via the aStar_custom_h() method.

    Args:
        map (Map): the game's Map object.
        start (int): index of the start tile
        target (int): index of the target (destination) tile.

    Returns:
        list[int]: An ordered list of ints that connect form a contiguous path from the start tile to the nearest end tile

    Note: If no valid path exists, an empty list is returned.
    """

    if not isinstance(targets, list):
        targets = [targets]

    TILE_COST = 1

    search_list: list[__aStarNode__] = []

    for i in range(map.size):
        search_list.append(__aStarNode__(i))

    open_heap: list[__aStarNode__] = []
    heapq.heappush(open_heap,search_list[start])

    while open_heap:
        cur_node = heapq.heappop(open_heap)

        if cur_node.tile in targets:
            #build path from end to start, then reverse it    
            
            path = [cur_node.tile]
            while cur_node.parent is not None:
                path.append(cur_node.parent.tile)
                cur_node = cur_node.parent

            return path[::-1] #reversed

        cur_node.closed = True

        adj_tiles = map.get_adjacent_tiles(cur_node.tile)

        for next_tile in adj_tiles:

            neighbour_node = search_list[next_tile]
            if neighbour_node.closed:
                continue

            # g score is the cost from start to the current node
            gscore = cur_node.g + TILE_COST
            visisted = neighbour_node.visited

            #if tile is owned by enemy, add extra weight
            if map.is_enemy(next_tile):
                gscore += map.armies[next_tile]
            elif map.terrain[next_tile] == TERRAIN_TYPES.EMPTY:
                gscore += 1

            if not visisted or gscore < neighbour_node.g:
                neighbour_node.visited = True
                neighbour_node.parent = cur_node
                neighbour_node.g = gscore
                neighbour_node.h = neighbour_node.h if neighbour_node.h > 0 else __aStar_get_nearest_endpoint_h__(map, start, targets)
                neighbour_node.f = neighbour_node.g + neighbour_node.h
                neighbour_node.armies = map.armies[next_tile]

                if not visisted:
                    heapq.heappush(open_heap,neighbour_node)
                else:
                    #already seen node, need to update heap for correct sorting
                    heapq.heapify(open_heap)
    return []

class __aStarNode__():
    tile:int
    g:int
    h:int
    f:int
    armies:int
    visited:bool
    closed: bool
    parent: '__aStarNode__'

    def __init__(self,tile:int,g=0,h=0,f=0,armies=0,visited=False,closed=False,parent=None) -> None:
        self.tile = tile
        self.g = g
        self.h = h
        self.f = f
        self.armies = armies
        self.visited = visited
        self.closed = closed
        self.parent = parent

    def __lt__(self, value:'__aStarNode__'):
        return self.armies < value.armies if self.f == value.f else self.f < value.f
        
def __aStar_get_nearest_endpoint_h__(map:'GameInstance', start:int, ends:list[int]) -> int:
        min = float('inf')

        for end in ends:
            dist = map.manhattan_distance(start,end)
            if dist < min:
                min = dist

        return min    
