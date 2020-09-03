"""
Vehicles Routing Problem (VRP).
Refered to example code here: 
    https://developers.google.com/optimization/routing/vrp
"""

from __future__ import print_function
import math, time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['waypoints'] = [
        (5, 5),         #// 0
        (-5, 5),       #// 1
        (10, -10),     #// 2
        (1, 0),        #// 3
        (2, -30),       #// 4
        (-2, 0),        # 5
        (-7, -1),        #6      
        (6, 2)          #7
    ]
    data['starts'] = [
        (0, 0),         # origin 1
        (5, 0),         # orgiin 2
        (-3, 4),        # orgin 
    ]
    data['resolution'] = 1
    return data

def compute_euclidean_distance_matrix(locations, res):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))*res)
    return distances

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    sum_route_distance = 0
    sum_cumul_distance = 0
    routes = []
    num_vehicles = len(data['starts'])
    for vehicle_id in range(num_vehicles):
        cumul_distance = 0.0
        node_index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        routes.append([])

        while not routing.IsEnd(node_index):
            location_idx = manager.IndexToNode(node_index)
            waypoint_idx = location_idx - num_vehicles

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
    return routes

################################################################################

class State:
    travel_distance = 0.0
    finish_time = 0.0
    waypoints = []

    def reset(self):
        self.travel_distance = 0.0
        self.finish_time = 0.0
        self.waypoints = []

def plot_route(data, routes): 
    """plot routes"""
    print("\n\n--------------- plotting Graph for validation --------------")
    _, ax = plt.subplots()
    ax.axis([-30, 30, -30, 30])
    sum_cost = 0.0
    sum_accum = 0.0

    for idx, start in enumerate(data["starts"]):
        plt.annotate("Veh " + str(idx), start)

    for veh_num, route in enumerate(routes):
        # starting point 
        string_path_data = [(mpath.Path.MOVETO, data["starts"][veh_num])]
        
        for idx in route:
            coor = data["waypoints"][idx]
            string_path_data.append((mpath.Path.LINETO, coor))

        codes, verts = zip(*string_path_data)
        string_path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(string_path, 
            facecolor="none", edgecolor=np.random.rand(3,), lw=2)
        ax.add_patch(patch)

        # dumb way to validate and cal accum again
        p_x = verts[0]
        cost = 0
        accum_cost = 0
        for i, x in enumerate(verts):
            if (i == 0):
                continue;
            d_cost = math.hypot(x[0] - p_x[0], x[1] - p_x[1])
            accum_cost += cost + d_cost
            cost += d_cost
            p_x = x
        print(" Total cost: {}, Accumulate cost {}".format(cost, accum_cost))
        sum_cost += cost
        sum_accum += accum_cost
        
    print(" Grand sum ==> Total cost: {}, Accumulate cost {}".format(
        sum_cost, sum_accum))

    plt.grid()
    plt.show()

def main():
    """Solve the VRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    start_time = time.time()
    veh_state = State()
    veh_state.reset()

    data['locations'] = data['starts'] + data['waypoints']
    num_vehicles = len(data['starts'])

    """
    Uncomment this to validate toyproblem.cpp cost function
    """
    # routes = [
    #     [3,  5,  4],
    #     [7,  0,  2],
    #     [1,  6]
    # ]
    # plot_route(data, routes)
    # return

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           num_vehicles,
                                           list(range(num_vehicles)),
                                           list(range(num_vehicles)))

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'],
                                                        data['resolution'])

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # hack to force no cost to return back starting points 
        # (to comply with routeindexmanager)
        veh_state.waypoints.append(from_node)
        if to_node in list(range(num_vehicles)):
            # veh_state.reset()
            return 0
        
        cost = distance_matrix[from_node][to_node] # duration
        return cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(1)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # search_parameters.log_search = True
    # search_parameters.time_limit.seconds = 3
    # search_parameters.local_search_metaheuristic = (
    #         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        routes = print_solution(data, manager, routing, solution)
        plot_route(data, routes)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
