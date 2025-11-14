from abc import ABC, abstractmethod


class InputDeviceBase(ABC):
    """Base class for teleoperation input device."""

    def __init__(self):
        self.state = None
        self.connected = False

    @abstractmethod
    def connect(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def read(self):
        pass

    def is_ready(self):
        return True

    @abstractmethod
    def set_command_data(self):
        pass

    def is_active(self) -> bool:
        """
        Check if the device is currently receiving active input from the user.
        Returns `True` if active, `False` otherwise.
        """
        # Default implementation can return True, but it's better to be explicit
        # in subclasses.
        return True
