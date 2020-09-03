"""
Vehicles Routing Problem (VRP).
Refered to example code here: 
    https://developers.google.com/optimization/routing/vrp
"""

from __future__ import print_function
import math, time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['locations'] = [
        (0, 0),         # origin 1
        (5, 0),         # orgiin 2
        (-3, 4),        # orgin 
        (5, 5),         #// 0
        (-5, 5),       #// 1
        (10, -10),     #// 2
        (1, 0),        #// 3
        (2, -30),       #// 4
        (-2, 0),        # 5
        (-7, -1),        #6      
        (6, 2)          #7
    ]
    data['starts'] = [0, 1, 2]
    # data['ends'] = [3, 7, 9] #046
    data['num_vehicles'] = len(data['starts'])
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
    for vehicle_id in range(data['num_vehicles']):
        cumul_distance = 0.0
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(    
                manager.IndexToNode(index)- data['num_vehicles'] )
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)/2
            cumul_distance += route_distance
        cumul_distance -= route_distance
        plan_output += '{} \n'
        print('Distance of the route: {}m'.format(route_distance))
        print('Cumul Distance: {}m'.format(cumul_distance))
        print(plan_output)
        sum_route_distance += route_distance
        sum_cumul_distance += cumul_distance
    print('Sum of the route distances: {}m'.format(sum_route_distance))
    print('Sum of the cumulated distances: {}m'.format(sum_cumul_distance))

################################################################################################

class State:
    travel_distance = 0.0
    finish_time = 0.0
    waypoints = []

    def reset(self):
        self.travel_distance = 0.0
        self.finish_time = 0.0
        self.waypoints = []

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    start_time = time.time()
    veh_state = State()
    veh_state.reset()
    print(veh_state.travel_distance)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], 
                                           data['starts'], data['starts'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'],
                                                        data['resolution'])
    hello= "hello"

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # print(from_node, to_node)
        # hack to force no cost to return back starting points 
        # (to comply with routeindexmanager)
        veh_state.waypoints.append(from_node)
        if to_node in data['starts']:
            # print(veh_state.waypoints)
            # veh_state.reset()
            return 0
        
        # veh_state.travel_distance +=  distance_matrix[from_node][to_node]
        # cost = veh_state.travel_distance
        # print(veh_state.travel_distance)
        # return cost
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        20,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # distance_dimension.SetGlobalSpanCostCoefficient(1000)
    
    # (Test) this will affect distance by factor of 2
    distance_dimension.SetSpanCostCoefficientForAllVehicles(1) 

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # search_parameters.log_search = True
    # search_parameters.time_limit.seconds = 2
    # search_parameters.local_search_metaheuristic = (
    #         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
