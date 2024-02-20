"""Base epidemic simulation datatypes"""

from typing import NamedTuple

class EpidemicEvent(NamedTuple):
    """Tracker of an event in an epidemic simulation

    Attributes:
        time (float): The time at which the event occurred.
        transition (int): The type of transition that occurred.
        individual (int): The individual involved in the event.
    """
    time: float
    transition: int
    individual: int
