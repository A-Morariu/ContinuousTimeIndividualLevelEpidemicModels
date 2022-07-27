###############################################################################
# PREAMBLE
###############################################################################

from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

import matplotlib.pyplot as plt

# %matplotlib inline

tfd = tfp.distributions


###############################################################################
# Helper functions
###############################################################################

def _expand_state(state_vector):
    """
    :param state_vector: tensor of total individuals in each state at
                        arbitrary t
    REQUIRE: num rows of incidence matrix (i.e. states) must equal length
            of state vector
    return: long form representation of an epidemic, a (num_states, num_units)
            tensor
    """
    # initialize list to store values
    accumulator = []
    for ii in range(len(state_vector)):
        accumulator.append(
            tf.repeat([tf.one_hot(ii, depth=len(state_vector))],
                      state_vector[ii], axis=0)
        )
    return tf.transpose(
        tf.concat(
            accumulator, axis=0)
    )


def _compute_source_states(IncidenceMatrix, dtype=tf.int32):
    """
    Computes the indices of the source states for each
    transition in a state transition model.
    :returns: a tensor of shape `(R,)` containing source state indices.
    """
    source_states = tf.reduce_sum(
        tf.cumsum(
            tf.clip_by_value(
                -IncidenceMatrix, clip_value_min=0, clip_value_max=1
            ),
            axis=-2,
            reverse=True,
            exclusive=True,
        ),
        axis=-2,
    )
    return tf.cast(source_states, dtype)


def _update_state(current_state, individual_ID, state_update):
    """
        Performs update based on individual and current state (event can be
        inferred from current state and indiviudal ID)

        :param current_state: tensor of the current time step of the epidemic
        :param individual_ID:
        :param state_update: tensor of the output of an iteration of the Gillespie algo
        """
    # Calculate indices - based on tf.slice structure
    indices = tf.stack([tf.range(state_update.shape[-1]),
                        tf.broadcast_to(individual_ID,
                                        [state_update.shape[-1]])],
                       axis=-1)

    return tf.tensor_scatter_nd_add(current_state,
                                    indices,
                                    state_update)


# new type to keep track of the simulation
_EventList = namedtuple("EventList", ["time", "transition", "unit"])


###############################################################################
# Output formatting functions
###############################################################################


def _compute_state(initial_state, event_list, incidence_matrix):
    """
    Recreate a state timeseries given an initial state and event list
    :param initial_state: a [num_states, num_units] dense tensor of 0s and 1s.
    :param event_list: an EventList structure
    :param incidence_matrix: a [num_states, num_transitions] matrix
    :returns: a tuple of (time, dense form structure of [num_times, num_states,
            num_units])
    """
    def fn(state, event):
        time, transition, unit = event
        state_incr = tf.gather(incidence_matrix, indices=transition, axis=-1)
        return _update_state(state, unit, state_incr)

    state_timeseries = tf.scan(fn, elems=event_list, initializer=initial_state)

    return (tf.cumsum(tf.concat([[0.0], event_list.time], axis=-1)),
            tf.concat([tf.expand_dims(initial_state, -3), state_timeseries],
                      axis=-3),
            )


def numerical_summary(epidemic_states):

    summary = tf.concat([tf.expand_dims(epidemic_states[0], axis=1, name='time'),
                         tf.reduce_sum(epidemic_states[1], axis=-1)],
                        axis=-1,
                        name="full_epidemic")
    format_summary = pd.DataFrame(summary.numpy())

    new_names = dict(zip(format_summary.columns,
                         ['time'] + state_names)
                     )

    format_summary = format_summary.rename(columns=new_names)

    return format_summary


def visual_summary(epidemic_states):
    return numerical_summary(epidemic_states).iloc[:, ].plot(x='time')


###############################################################################
# Build class to simulate epidemic
###############################################################################


class ContinuousTimeStateTransitionModelSimulation:

    def __init__(
            self,
            incidence_matrix,
            initial_state,
            transition_rate_fn,
            stop_condition_fn=None,
            seed=None) -> None:
        """
        Initialize an epidemic

        :param incidence_matrix: a [S, R] matrix for S states and R transitions
                             denoting the topology of the state transition
                             model.
        :param initial_state: a [1, num_states] tensor of total individuals in
                            each state at t_0
        :param seed: seed for the simulation

        :return: Null

        """
        self.IncidenceMatrix = tf.convert_to_tensor(incidence_matrix)
        self.InitialState = _expand_state(
            tf.convert_to_tensor(initial_state, dtype=np.int32))
        self.TransitionRateFn = transition_rate_fn
        self.StopConditionFn = stop_condition_fn
        self.Seed = seed

    # Graph-compile the simulate function
    def simulate_continuous_time_state_transition_model(self, trace_fn=True):
        """Simulates from a continuous time state transition model

        :param trace_fn: TO DO
        """
        seed = samplers.sanitize_seed(
            self.Seed, salt="simulate_continuous_time_state_transition_model")
        source_states = _compute_source_states(self.IncidenceMatrix)

        # Inner dimension represents the unit
        num_units = self.InitialState.shape[-1]

        def Gillespie_iteration(step, time, state, seed, accum):
            """
            Perform 1 iteration of the Gillespie algorithm for an epidemic model.
            Record all necessary information and return the updated state of the
            system.
            """
            # 1. Calculate transition rates *** THIS NEEDS TO BE FIXED ***
            rates = self.TransitionRateFn(time, state) * \
                tf.gather(state, source_states)
            rate_sum = tf.reduce_sum(rates)

            # Random seeds
            seeds = samplers.split_seed(seed, n=2)

            # 2. Sample required variables
            # 2.1 Time *of* next event
            t_star = tfd.Exponential(rate=rate_sum).sample(seed=seeds[0])
            next_time = time + t_star

            # 2.2 Individual who undergoes event
            categorical_ID = tfd.Categorical(
                probs=tf.reshape(rates, [-1])).sample(seed=seeds[1])
            individual_ID = categorical_ID % num_units
            transition_ID = categorical_ID // num_units

            # 3. Update the state matrix
            # `state` is [num_states, num_units]
            # we need to index `incidence_matrix[:, transition_ID]` into the
            # `individual_ID`th column of `state`.
            next_state = _update_state(state,
                                       individual_ID,
                                       tf.gather(self.IncidenceMatrix,
                                                 indices=transition_ID,
                                                 axis=-1)
                                       )

            # 4. Remainder of book keeping. Consistent with the functional flavour
            # of TF, we mustcapture and return the output from the .write()
            # operations otherwise they will not be added to the computational
            # graph
            accum = _EventList(time=accum.time.write(step, next_time),
                               transition=accum.transition.write(
                step, transition_ID),
                unit=accum.unit.write(step, individual_ID))

            # return statement must match the input for the function to it can
            # flow into the tf.while_loop
            return step + 1, next_time, next_state, seeds[1], accum

        # accum buffer
        eventList = _EventList(
            time=tf.TensorArray(
                self.InitialState.dtype, size=0, dynamic_size=True), transition=tf.TensorArray(
                tf.int32, size=0, dynamic_size=True), unit=tf.TensorArray(
                tf.int32, size=0, dynamic_size=True))

        # Simple stopping condition, in this case when the rate drops to 0
        #   - Exercise: make this user-definable!

        def cond(step, t, state, *args):
            # keep inside here since we need to access the transition rate fn
            # Condition 1
            rate = tf.reduce_sum(
                self.TransitionRateFn(t, state) *
                tf.gather(state, source_states)
            )
            # Condition 2: stop after 3 steps (for debugging at the moment)
            return rate > 0
        self.StopConditionFn = cond

        # Run while loop to simulate to stopping point
        step, time, state, seed, eventList = tf.while_loop(
            self.StopConditionFn,
            Gillespie_iteration,
            loop_vars=(0, 0.0, self.InitialState, seed, eventList)
        )

        return tf.nest.map_structure(lambda x: x.stack(), eventList)


###############################################################################
# User end
###############################################################################

def make_rate_fn(age, age_factor, beta, gamma):
    # keep data variables and parameters on the outside
    def rate_fn(t, state):
        # t and state change with each iteration of the Gillespie so they are
        # a loop variable and thus inside the rate_fn
        si_rate = age * age_factor + beta * \
            tf.reduce_sum(state[1, :]) / tf.reduce_sum(state, axis=-2)
        ir_rate = tf.broadcast_to([gamma], shape=si_rate.shape)
        return tf.stack([si_rate, ir_rate], axis=0)  # [R x M]
    return rate_fn


# Define variables to govern epidemic
sir_graph = np.array([[-1, 0],
                      [1, -1],
                      [0, 1]], dtype=np.float32)

initial_state = np.array([[9],
                          [1],
                          [0]], dtype=np.float32)

ages = np.random.uniform(
    low=10, high=60, size=np.sum(initial_state, dtype=np.int32))

state_names = ['Susceptible', 'Infected', 'Removed']

example_epidemic = ContinuousTimeStateTransitionModelSimulation(
    incidence_matrix=sir_graph,
    initial_state=initial_state,
    transition_rate_fn=make_rate_fn(
        ages,
        0.08,
        0.1,
        0.05))
# simulate the epidemic
time_stamps, transition_types, individual = example_epidemic.simulate_continuous_time_state_transition_model()