"""Epidemic class for defining and simulating an epidemic model"""
from typing import Callable, NamedTuple

import tensorflow as tf

from epi_sim_base import EpidemicEvent

tensor = tf.Tensor


class Epidemic():
    """Computational representation of an epidemic simulation. Stores
    the key values needed to define an epidemic simulation and run it
    through the simulation function.

    Attributes:
        incidence_matrix (tf.Tensor): The incidence matrix representing the
            connections between individuals in the population. Rows represent the
            states, columns represent the transitions. 
        state (tf.Tensor): The initial state of the epidemic, specifying
            the infection status of each individual.
        transition_fn (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): The transition
            function that defines the epidemic dynamics.
    """
    def __init__(self,
                 incidence_matrix: tensor,
                 state: tensor,
                 transition_fn: Callable[[tensor, tensor], tensor]
                 ) -> None:
        """__init__ constructor for Epidemic class

        Outline the base attributes of the Epidemic class. This is the minimal
        requirement to define an epidemic model for computation.

        Args:
            incidence_matrix (tensor): a number of states by number of transitions
                matrix that defines the connections between states in the epidemic
                model. Rows represent the states, columns represent the transitions.
                This is a graph representation of the epidemic model. 
            state (tensor): a vector of the initial state of the epidemic. This vector
                contains the initial counts for all of the states in the model.
            transition_fn (Callable[[tensor, tensor], tensor]): a function that returns
                the transition rates for the epidemic. This function is to be defined
                as a closure over the data and parameters of the epidemic model.
        """
        self.incidence_matrix = incidence_matrix
        self.state = state
        self.transition_fn = transition_fn
        
    def simulate(self,
                 N: tensor,
                 stop_cond_fn: Callable[[tensor], bool],
                 seed: float
                 ) -> EpidemicEvent:
        """Simulate the epidemic model

        Simulate the epidemic model using the transition function and the
        initial state of the epidemic. This function will return the events
        that occur during the simulation.

        Args:
            t (float): the total time to simulate the epidemic
            dt (float): the time step to use in the simulation
            events (int): the number of events to simulate

        Returns:
            EpidemicEvent: a list of events that occur during the simulation
        """
        pass
    
    def log_prob(self,
                 events: EpidemicEvent
                 ) -> tf.Tensor:
        """Compute the log probability of the events given the epidemic model

        Compute the log probability of the events given the epidemic model. This
        function will return the log probability of the events given the epidemic
        model.

        Args:
            events (EpidemicEvent): a list of events that occur during the simulation

        Returns:
            tf.Tensor: the log probability of the events given the epidemic model
        """
        pass
