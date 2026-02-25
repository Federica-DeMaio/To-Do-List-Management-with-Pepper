import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from abc import ABC, abstractmethod
import re

class PepperIPNotSetError(Exception):
    def __init__(self, *args):
        super().__init__("You have to set Pepper IP as ROS2 param")

class PepperNode(Node):
    
    def __init__(self, node_name, *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        self.ip = self.declare_parameter("pepper_ip", Parameter.Type.STRING).get_parameter_value().string_value
        if self.ip == "":
            raise PepperIPNotSetError
        elif not re.match(r"\d+\.\d+\.\d+\.\d+", self.ip):
            raise Exception("I expected an IP received: {}".format(self.ip))
        self.port = self.declare_parameter("pepper_port", 9559).get_parameter_value().integer_value

class Context:
    """
    The Context class is responsible for managing the current state of the application.

    It allows for transitioning between different states and delegates the handling of 
    operations to the current state.

    Attributes:
        _state (State): The current state of the context.

    Methods:
        __init__(state): Initializes the context with a given state.
        transition_to(state): Transitions the context to a new state.
        run(): Executes the handle method of the current state.
    """

    _state = None

    def __init__(self, state) -> None:
        self.transition_to(state)

    def transition_to(self, state):
        print(f"Context: Transition to {type(state).__name__}")
        self._state = state
        self._state.context = self

    def run(self):
        self._state.handle()

class State(ABC):
    """
    Abstract base class representing a state in a state machine.

    This class defines the interface for a state, including a context
    property that allows the state to access shared data or functionality
    from the context in which it operates. Subclasses must implement the
    `handle` method to define the behavior of the state.

    Attributes:
        context (Context): The context in which the state operates.

    Methods:
        handle(): Abstract method that must be implemented by subclasses
                  to define the state's behavior.
    """
    
    @property
    def context(self) -> Context:
        return self._context

    @context.setter
    def context(self, context: Context) -> None:
        self._context = context

    @abstractmethod
    def handle(self) -> None:
        pass

class StateROS(State):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node