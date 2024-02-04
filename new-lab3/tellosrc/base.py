import abc
from threading import Thread, Event
from time import sleep


class StoppableThread(Thread, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_event = Event()
        self.stopped_event = Event()

    def stop(self):
        # The derived class should check the `stop_event` by calling `stopped()`
        # to determine whether it should stop.
        self.stop_event.set()
        counter = 0
        while self.is_alive():
            print(f"Wait for [{self.__class__.__name__}] to stop: {counter}")
            sleep(1)
            counter += 1
        print(f"[{self.__class__.__name__}] is stopped")

    def stopped(self):
        return self.stop_event.is_set()


class ResourceThread(StoppableThread):
    @abc.abstractmethod
    def get_result(self):
        """Get the result of the thread

        Return a copy of the state to avoid the state being modified
        by the caller.

        """
        return NotImplemented
