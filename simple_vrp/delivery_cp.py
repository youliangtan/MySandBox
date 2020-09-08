"""
Vehicles Routing Problem (VRP).
Refered to example code here: 
    https://developers.google.com/optimization/routing/vrp
"""

from __future__ import print_function
import math, time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from task_allocation_viz import TaskAllocationViz
from task_allocation_viz import load_yaml

def get_task_allocation(manager, routing, solution, num_agents):
    """Prints solution on console."""
    sum_route_distance = 0
    sum_cumul_distance = 0
    routes = []
    for vehicle_id in range(num_agents):
        cumul_distance = 0.0
        node_index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        routes.append([])

        while not routing.IsEnd(node_index):
            location_idx = manager.IndexToNode(node_index)
            waypoint_idx = location_idx - num_agents

            if (waypoint_idx>=0):
                routes[vehicle_id].append(waypoint_idx)
                plan_output += ' {} -> '.format( waypoint_idx )
            
            previous_index = node_index
            node_index = solution.Value(routing.NextVar(node_index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, node_index, vehicle_id)
            
            cumul_distance += route_distance
        
        cumul_distance -= route_distance
        sum_route_distance += route_distance
        sum_cumul_distance += cumul_distance
        
        # prints
        print(plan_output)
        print(' - Distance of the route: {}m'.format(route_distance))
        print(' - Cumul Distance: {}m\n'.format(cumul_distance))
    print('Sum of the route distances: {}m'.format(sum_route_distance))
    print('Sum of the cumulated distances: {}m'.format(sum_cumul_distance))
    return routes # aka tasks allocation

################################################################################

def main():
    """Solve the VRP problem."""
    # Instantiate the data problem.
    
    start_time = time.time()
    agents, tasks, _ = load_yaml("task_config.yaml")

    plot_viz = TaskAllocationViz(agents, tasks)
    all_locations = agents + tasks
    num_agents = len(agents)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(all_locations),
                                           num_agents,
                                           list(range(num_agents)),
                                           list(range(num_agents)))

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def transit_callback_fn(from_index, to_index):
        """Returns the cost between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # due hack to force no cost to return back starting points 
        # (to comply with routeindexmanager ending points)
        if to_node in list(range(num_agents)):
            return 0
        # Current return distance as the only cost, TODO: add duration
                # Euclidean distance
        start = all_locations[from_node] 
        end = all_locations[to_node]
        travel_distance = math.hypot((start[0] - end[0]), (start[1] - end[1]))
        cost = travel_distance
        return cost

    # Define cost of each arc.
    transit_callback_index = routing.RegisterTransitCallback(transit_callback_fn)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = 'TransitCost'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    routing.GetDimensionOrDie(dimension_name).SetGlobalSpanCostCoefficient(1)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # search_parameters.log_search = True
    # search_parameters.time_limit.seconds = 5
    # search_parameters.local_search_metaheuristic = (
    #         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Print solution on console.
    if solution:
        allocation = get_task_allocation(manager, routing, solution, num_agents)
        plot_viz.plot_task_queues(allocation)

if __name__ == '__main__':
    main()
