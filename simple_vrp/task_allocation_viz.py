
import math, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

################################################################################
# NOT BEING USED NOW

class RequestTask:
    class Profile:
        def __init__(self, task_id, location):
            self.task_id = task_id
            self.locaton = location
            self.priority = False
            self.time_window = (0,0)

    class Plan:
        def __init__(self):
            self.travel_distance = 0.0  # location_T-1 to locatoin_T
            self.duration = 0.0         # travel_distance/speed
            self.finish_time = 0.0      # n_sum(durations)

    def __init__(self, task_id, location):
        self.profile = self.Profile(task_id, location)
        self.plan = self.Plan()

class Agent:
    def __init__(self, agent_id, location):
        self.agent_id = agent_id
        self.locaton = location
        self.speed = 1.0
        self.assignments = []

    def assign_task(self, task):
        assert isinstance(task, RequestTask)
        self.assignments.append(task)

################################################################################

class TaskAllocationViz:

    def __init__(self, agents, tasks):
        """Init Task allocation viz"""
        self.agents = agents
        self.tasks = tasks

    def plot_task_queues(self, all_task_queues):
        """Plot task queues
        arg: task queue: example [[1,0,3],[5,2],[4]]
                represents indexs for the tasks
        """
        print("\n\n--------------- plotting Graph for validation --------------")
        assert len(all_task_queues) == len(self.agents)
        num_tasks = sum( [len(task_queue) for task_queue in all_task_queues])
        assert len(self.tasks) == num_tasks
        
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Agent Task Routes")
        ax2.set_title("Agent Task Durations")
        ax1.axis([-30, 20, -30, 20])
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
            patch = mpatches.PathPatch(string_path, 
                facecolor="none", edgecolor=np.random.rand(3,), lw=2)
            ax1.add_patch(patch)

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

        ax1.grid()
        ax2.grid()
        plt.show()
