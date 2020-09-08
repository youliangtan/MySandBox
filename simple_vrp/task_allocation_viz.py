
import math, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll

import random
import yaml

################################################################################
# NOT BEING USED NOW

class RequestTask:
    class Profile:
        def __init__(self, task_id, waypoints):
            self.task_id = task_id
            self.waypoints = waypoints
            self.priority = False
            self.time_window = (0,0)

    class Plan:
        def __init__(self):
            self.travel_distance = 0.0  # location_T-1 to locatoin_T
            self.duration = 0.0         # travel_distance/speed
            self.finish_time = 0.0      # n_sum(durations)

    def __init__(self, task_id, waypoints):
        self.profile = self.Profile(task_id, waypoints)
        self.plan = self.Plan()

class Agent:
    def __init__(self, agent_id, waypoint):
        self.agent_id = agent_id
        self.waypoint = waypoint
        self.speed = 1.0
        self.assignments = []

    def assign_task(self, task):
        assert isinstance(task, RequestTask)
        self.assignments.append(task)

################################################################################

class TaskAllocationViz:

    def __init__(self, agents, tasks):
        """Init Task allocation viz"""
        self.agents = agents    # [(x,y), (x, y).... ]
        self.tasks = tasks      # [(x,y), (x, y).... ]

    def plot_task_queues(self, all_task_queues):
        """Plot task queues
        arg: task queue: example [[1,0,3],[5,2],[4]]
                represents indexs for the tasks
        """
        print("\n\n--------------- plotting Graph for validation --------------")
        assert len(all_task_queues) == len(self.agents)

        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Agent Task Routes")
        ax2.set_title("Agent Task Durations")
        
        sum_cost = 0.0
        sum_accum = 0.0

        all_task_costs = []

        for idx, start in enumerate(self.agents):
            ax1.annotate("Agent" + str(idx), start)

        for veh_num, task_queue in enumerate(all_task_queues):
            # starting point 
            verts = [self.agents[veh_num]]           
            codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*len(task_queue)

            for task_idx in task_queue:
                verts.append(self.tasks[task_idx])
           
            string_path = mpath.Path(verts, codes)
            self.colorline(ax1, string_path)

            # TODO Clean all these below
            # dumb way to validate and cal accum cost again
            all_task_costs.append([]) # todo
            p_x = verts[0]
            cost = 0
            accum_cost = 0
            for i, x in enumerate(verts):
                if (i == 0): #ignore agent's current position
                    continue
                d_cost = math.hypot(x[0] - p_x[0], x[1] - p_x[1])
                accum_cost += cost + d_cost
                cost += d_cost
                all_task_costs[veh_num].append(d_cost) # todo
                p_x = x
            print(" Total cost: {}, Accumulate cost {}".format(cost, accum_cost))
            sum_cost += cost
            sum_accum += accum_cost
            
        print(" Grand sum ==> Total cost: {}, Accumulate cost {}".format(
            sum_cost, sum_accum))

        # plot gantt chart here
        for agent_id, costs in enumerate(all_task_costs):
            sum_cost=0
            for cost in costs:
                ax2.barh(agent_id,width=cost,left=sum_cost)
                sum_cost+=cost

        ax2.set_yticks(range(len(self.agents)))
        ax2.set_yticklabels([f'agent{i}' for i in range(len(self.agents))])

        ax1.set_facecolor((0.9, 0.9, 0.9))
        ax1.autoscale()
        ax1.grid()
        ax2.grid()
        plt.show()


    def colorline(self, ax, path, linewidth=4, alpha=1.0):
        """ Function to plot color gradient lines on graph """
        verts = path.interpolated(steps=10).vertices
        x, y = verts[:, 0], verts[:, 1]
        z = np.linspace(0, 1, len(x))
        cmaps = ['Oranges', 'Greens', 'Purples', 'Blues', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        
        z = np.asarray(z)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        norm=plt.Normalize(0.0, 0.8)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, array=z, cmap=random.choice(cmaps), norm=norm,
                                linewidth=linewidth, alpha=alpha)

        ax.add_collection(lc)

################################################################################

def load_yaml(task_path, allocation_path="", task_type="station"):
    """ Load Yaml to Visualize VRP """
    
    agents = []
    tasks = [] # here will break down delivery to 2 tasks
    allocation = []
    
    with open(task_path, 'r') as stream:
        task_config = yaml.safe_load(stream)
        grid_size = task_config["grid_size"]
        grid_length = task_config["grid_length"]
        graph = []

        # Construct Grid World Graph
        for i in range(grid_size):
            for j in range(grid_size):
                graph.append((i*grid_length, j*grid_length))

        assert len(graph) != 0

        for _agent in task_config["agents"]:
            agents.append(graph[_agent["wp"]])

        # populate tasks
        if task_type == "delivery":
            for _task in task_config["tasks"]["delivery"]:
                tasks.append(graph[_task["pickup"]])
                tasks.append(graph[_task["dropoff"]])
        elif task_type == "station":
            for _task_idx in task_config["tasks"]["station"]:
                tasks.append(graph[_task_idx])

    assert len(agents) != 0
    assert len(tasks) != 0

    if allocation_path == "":
        return agents, tasks, allocation    

    # If allocation.yaml is provided
    # Delivery, thus abit tricky to dealt with
    with open(allocation_path, 'r') as stream:
        allocation_config = yaml.safe_load(stream)
        task_queues = allocation_config[task_type]
        for agent_idx, queue in enumerate(task_queues):
            allocation.append([])
            for task_idx in queue:
                if (task_type == "delivery"):
                    allocation[agent_idx].append(task_idx*2)
                    allocation[agent_idx].append(task_idx*2+1)
                elif (task_type == "station"):
                    allocation[agent_idx].append(task_idx)

    return agents, tasks, allocation

################################################################################

if __name__ == '__main__':
    agents, tasks, allocation = load_yaml(
        "task_config.yaml", "allocation.yaml", "delivery")
    plot_viz = TaskAllocationViz(agents, tasks)
    plot_viz.plot_task_queues(allocation)
