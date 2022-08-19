from distutils.log import ERROR
from typing import Callable, List, Optional
import enum

import rmf_adapter as adpt


#####################################################################
class Position:
    x: float
    y: float
    yaw: float
    map_name: str

class Status(enum.IntEnum):
    UNINITIALIZED = 1
    OFFLINE = 2
    SHUTDOWN = 3
    IDLE = 4
    CHARGING = 5
    WORKING = 6
    ERROR = 7

class RobotState:
    position: Optional[Position] = None
    status: Optional[Status] = None
    battery: Optional[float] = None

#####################################################################

class SimpleFleetAdapter:

    class RobotConfig:
        def __init__(
                self,
                fleet_name: str,
                graph,
                update_frequency: float = 1.0
            ):
            pass

    class TaskConfig:
        def __init__(
                self,
                accept_patrol: bool,
                accept_delivery: bool,
                accept_action: List[str]
            ):
            pass

    """
    @args:
    :param node_name:       name of the fleet adapter node
    :param robot_config:    robot configurations
    :param task_config:     task configurations
    :param server_uri:      websocket server uri, default none
    """
    def __init__(
            self,
            node_name: str,
            robot_config: RobotConfig,
            task_config: Optional[TaskConfig] = None,
            server_uri: str = ""
        ):
        pass


    """
    @args:
    :param robot_name:      robot name should be unique
    :param get_state_cb:    callback function for robot to provide
                            the current robot state to adapter
    :param navigate_cb:     callback function to notify the robot
                            where to navigate to. return a completion
                            callback for robot to notify the adapter
                            that the action is completed
    :param action_cb:       callback function to notify the robot to
                            execute a rmf action. return a completion
                            callback for robot to notify the adapter
                            that the action is completed
    :return if robot is added successfully
    """
    def add_robot(
            self,
            robot_name: str,
            get_state_cb: Callable[[], RobotState],
            navigate_cb: Callable[[Position], Callable[[], bool]],
            action_cb: Callable[[str], Callable[[], bool]],
        ) -> None:
        pass

    """
    start running adapter
    """
    def start():
        pass

    """
    This should only be used if user wish to access the internal
    RobotUpdateHandle api of rmf fleet adapter

    :param robot_name:      name of the added robot
    """
    def get_update_handle(
            robot_name: str
        ) -> Optional[adpt.RobotUpdateHandle]:
        pass


#####################################################################
#####################################################################
# Example

class RobotAPI:
    __init__():
        self.pos = rospy.sub(tf)


    def get_state() -> RobotState:
        state = RobotState

        state.pos = self.pos
        return state

    def navigate(
        target: Position) -> Callable[[], bool]:
        self.goal.send(target)
        def completion():
            return True

        return completion

    def start_action(
        json_data: str) -> Callable[[], bool]:
        self.do_something(json(str))
        def completion():
            return True

        return completion

#####################################################################
# tutorial 1: Create a simple fleet adapter

from simple_fleet_adapter import SimpleFleetAdapter
import rmf_adapter.graph as graph
import rmf_adapter.vehicletraits as traits
import rmf_adapter.geometry as geometry
from os.path import exists

nav_path= "/home/youliang/openrmf_ws/install/ctf_maps/share/ctf_maps/maps/expo/nav_graphs/1.yaml"
print("nav graph exists? ", exists(nav_path))

profile = traits.Profile(
    geometry.make_final_convex_circle(1),
    geometry.make_final_convex_circle(2))
traits = traits.VehicleTraits(
    traits.Limits(0.1), traits.Limits(0.8), profile)

graph = graph.parse_graph(nav_path, traits)

# robot profile
robot_config = SimpleFleetAdapter.RobotConfig(
    "dumbots", graph)

# what task to accept
task_config = SimpleFleetAdapter.TaskConfig(
    True, False, ["manual_control", "cleaning"])

fleet_adapter = SimpleFleetAdapter(
    "bot_node", robot_config, task_config, "localhost:5000/_internal")

# add a full control robot
fleet_adapter.add_robot(
        "dummy",
        RobotAPI.get_state,
        RobotAPI.navigate,
        RobotAPI.start_action,
    )

print("done")

#####################################################################
## Tutorial 2: readonly robot

fleet_adapter2 = SimpleFleetAdapter(
    "bot2_node", robot_config, None, "localhost:5000/_internal")

# add a readonly robot
fleet_adapter2.add_robot(
        "readonly",
        RobotAPI.get_state,
        None,
        None
    )

