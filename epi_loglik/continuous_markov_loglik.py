"Base functions for likelihood computation of continuous-time Markov processes"
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gemlib.distributions.continuous_markov import (
    EpidemicEvent,
    _update_state,
    _one_hot_expand_state,
)


# aliasing for convenience
tfd = tfp.distributions
Tensor = tf.Tensor
DTYPE = tf.float32


# helper functions for computing the likelihood of a trajectory
def event_list_to_table(
    event: EpidemicEvent, num_individuals: int, num_transitions: int
) -> Tensor:
    """
    Converts an epidemic event list into a table representation.

    Args:
        event (EpidemicEvent): The epidemic event list.
        num_individuals (int): The number of individuals in the population.
        num_transitions (int): The number of transitions in the epidemic.

    Returns:
        table (List[List[int]]): A table representation of the epidemic event list.
            Each row represents an individual, and each column represents a transition.
            The value at each cell indicates the state of the individual at that transition.
    """
    # create blank
    blank = tf.fill(dims=[num_individuals, num_transitions], value=np.inf)

    indices = tf.stack([event.individual, event.transition], axis=-1)

    update = event.time

    return tf.tensor_scatter_nd_update(blank, indices, update)


def _compute_time_deltas(time: Tensor) -> Tensor:
    """
    Compute the time differences between consecutive elements in the given time tensor.

    Args:
        time (Tensor): A tensor containing the time values. This comes
        from an EpidemicEvent.time attribute.

    Returns:
        Tensor: A tensor containing the time differences between
        consecutive elements in the time tensor.
    """
    intial_time = time[0]

    time_deltas = tf.experimental.numpy.diff(time)

    return tf.concat(
        [
            tf.expand_dims(intial_time, axis=-1),
            time_deltas,
        ],
        axis=0,
    )


def _sanitize_epidemic_event(event: EpidemicEvent) -> EpidemicEvent:
    """Internal function to return only valid events of an
    EpidemicEvent object.

    Removes any events with non-finite times. This is used to avoid
    NaN errors when computing the likelihood of the event.

    Args:
        event (EpidemicEvent): a realization of an epidemic process

    Returns:
        EpidemicEvent: a sanitized version of the input event (if all
        event times are finite then the input event is returned unchanged)
    """
    # check if all times are finite
    times = event.time

    valid_event = tf.math.is_finite(times)

    if tf.math.reduce_all(valid_event):
        return event

    # filter out invalid events
    valid_times = tf.boolean_mask(times, valid_event)
    valid_transitions = tf.boolean_mask(event.transition, valid_event)
    valid_event = tf.boolean_mask(event.individual, valid_event)

    # rebuild event
    valid_event = EpidemicEvent(valid_times, valid_transitions, valid_event)
    return valid_event


def expand_epidemic_event(
    incidence_matrix: Tensor,
    initial_state: Tensor,
    num_jumps: int,
    event: EpidemicEvent,
) -> Tensor:
    """
    Expands a single epidemic event into a sequence of states based on transition rules.

    This function takes a single epidemic event (`event`) and simulates its propagation
    across a network represented by an incidence matrix (`incidence_matrix`) for a specified
    number of jumps (`num_jumps`). It starts with an initial state (`initial_state`) and
    iteratively updates the state based on the transition rules defined in the `event`
    and the network structure.

    Args:
      incidence_matrix: A tensor representing the network structure. This is typically
        a sparse adjacency matrix where non-zero entries indicate connections between individuals.
      initial_state: A tensor representing the initial state of the individuals in the network.
        The format of this tensor depends on the specific epidemic model being used.
      num_jumps: An integer specifying the number of simulation steps to perform.
      event: An `EpidemicEvent` object containing information about the initial event,
        including the time of the event, the transition type, and the individual involved.

    Returns:
      A tensor representing the sequence of states after each jump. The structure of
      the returned tensor matches the format of the `initial_state`.
    """

    # Create a TensorArray to accumulate state updates.
    accum = tf.TensorArray(DTYPE, size=num_jumps, dynamic_size=True)
    # Write the initial state to the accumulator
    accum = accum.write(0, _one_hot_expand_state(initial_state))

    # Loop condition: Continue as long as the jump counter is less than num_jumps.
    def cond(ii, *_):
        return ii < num_jumps - 1

    # Loop body: Update state and accumulate the result.
    def body(ii, expanded_state, accum):
        """
        Updates the state of the network based on the current event and
        accumulates the updated state in the TensorArray.
        """

        # Generate a new event object with time and transition information
        # from current loop iteration.
        next_event = EpidemicEvent(
            event.time[ii],
            event.transition[ii],
            event.individual[ii],
        )

        next_state = _update_state(incidence_matrix, expanded_state, next_event)

        # Write the updated state to the TensorArray at the current jump index.
        accum = accum.write(ii + 1, next_state)

        # Increment jump counter, return updated state and TensorArray.
        return ii + 1, next_state, accum

    # Run the while loop with initial loop variables.
    # Start with the initial state that is already in the accumulator
    # to create left closed - right open intervals for the rates
    _, _, accum = tf.while_loop(
        cond, body, loop_vars=(0, _one_hot_expand_state(initial_state), accum)
    )

    # Stack the accumulated states to form the final output tensor.
    return tf.nest.map_structure(lambda x: x.stack(), accum)


def continuous_time_log_likelihood(
    transition_rate_fn: Callable,
    incidence_matrix: Tensor,
    initial_state: Tensor,
    num_jumps: int,
    event: EpidemicEvent,
) -> float:

    # TensorArray to store the log-likelihood of each event
    log_lik_accum = tf.TensorArray(DTYPE, size=num_jumps)
    log_lik_accum = log_lik_accum.write(0, 0.0)  # Initialize with zero log-likelihood

    # Loop condition: Continue as long as the jump counter is less than num_jumps.
    # Warning: can loop over degenrate events which have P(x) = 0 but evaluate to NaN
    # due to the tfp.Exponential.log_prob(np.inf) -> NaN
    def cond(ii, *_):
        return ii < num_jumps - 1

    def body(ii, state, log_lik_accum):
        """
        Updates the state of the network based on the current event and
        accumulates the log-likelihood of the event in the TensorArray.
        """

        # Compute the transition rates for the current state
        transition_rates = transition_rate_fn(event.time[ii], state)
        transition_rates = tf.math.multiply(transition_rates, state[:-1, :])

        total_rate = tf.reduce_sum(transition_rates)
        event_rate = transition_rates[event.transition[ii], event.individual[ii]]

        # Compute the time interval between the current and next event
        time_delta = event.time[ii + 1] - event.time[ii]

        # Compute the log-likelihood of the event
        log_lik = tf.math.log(event_rate) - total_rate * time_delta
        print(f"Slice {ii} --> Time-delta {time_delta} --> {log_lik}")

        # Accumulate the log-likelihood in the TensorArray
        log_lik_accum = log_lik_accum.write(ii + 1, log_lik)

        # Update the state based on the event
        next_event = EpidemicEvent(
            event.time[ii],
            event.transition[ii],
            event.individual[ii],
        )
        next_state = _update_state(incidence_matrix, state, next_event)
        # Increment jump counter, return updated state and TensorArray.
        return ii + 1, next_state, log_lik_accum

    _, _, log_lik_accum = tf.while_loop(
        cond,
        body,
        loop_vars=(0, _one_hot_expand_state(initial_state), log_lik_accum),
    )
    log_lik_accum = tf.nest.map_structure(lambda x: x.stack(), log_lik_accum)

    return tf.reduce_sum(
        tf.boolean_mask(log_lik_accum, tf.math.is_finite(log_lik_accum))
    )


def _make_transition_rates(rate_fn):
    tx_rate_fn = rate_fn

    def _compute_transition_rates(event: EpidemicEvent, states):

        # need to materialize states
        transition_rates_of_state = tf.map_fn(
            fn=lambda x: tf.stack(tx_rate_fn(*x)),
            elems=[event.time, states],
            dtype=tf.float32,
        )

        indices = tf.stack(
            [
                event.time.shape(),
                event.transition,
                event.individual,
            ],
            axis=-1,
        )

        # compute event specific rate - NEED indices of the event that happened
        event_rate = tf.gather_nd(transition_rates_of_state, indices)
        # compute total rate for a state
        total_rate = tf.einsum("ijk -> i", transition_rates_of_state)

        return total_rate, event_rate

    return _compute_transition_rates


def continuous_time_log_likelihood_v1(
    transition_rate_fn: Callable,
    incidence_matrix: Tensor,
    initial_state: Tensor,
    num_jumps: int,
    event: EpidemicEvent,
) -> float:
    """
    Computes the log-likelihood of a continuous-time Markov process
    given the transition rate function,
    incidence matrix, initial state, number of jumps, and event data.

    Args:
        transition_rate_fn (Callable): A function that computes the
        transition rate given the current state and time.
        incidence_matrix (Tensor): The incidence matrix representing
        the connections between states.
        initial_state (Tensor): The initial state of the process.
        num_jumps (int): The number of jumps to simulate.
        event (EpidemicEvent): The event data containing the times
        and states.

    Returns:
        Tensor: The log-likelihood of the continuous-time Markov process.
    """
    # expand the event into a sequence of (sparse) states
    states = expand_epidemic_event(incidence_matrix, initial_state, num_jumps, event)
    # vectorized map since slices are independent
    transition_rates_per_state = tf.vectorized_map(
        fn=lambda x: tf.stack(transition_rate_fn(*x)),
        elems=[event.time, states],
        # dtype=DTYPE,
    )

    # event rates for the events which occured
    indices = tf.stack(
        [
            tf.range(num_jumps),
            event.transition,
            event.individual,
        ],
        axis=-1,
    )
    print(f"Indices: {indices}")
    print(f"Transition rates per state: {transition_rates_per_state}")
    event_rates = tf.gather_nd(transition_rates_per_state, indices)

    # time intervals between events
    time_deltas = _compute_time_deltas(event.time)

    event_log_likelihoods = tfd.Exponential(rate=event_rates).log_prob(time_deltas)

    return tf.reduce_sum(event_log_likelihoods)
