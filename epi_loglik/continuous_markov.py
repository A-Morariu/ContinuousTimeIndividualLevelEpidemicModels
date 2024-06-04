"""Function for continuous time simulation"""

from typing import Callable, List, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

# aliasing for convenience
tfd = tfp.distributions
Tensor = tf.Tensor
DTYPE = tf.float32


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


def _update_state(
    incidence_matrix: Tensor, state: Tensor, next_event: EpidemicEvent
) -> Tensor:
    """Updates the state of the epidemic

    Args:
        state (tensor): The current state of the epidemic.
        event (EpidemicEvent): The event that occurred in the epidemic.

    Returns:
        tensor: The updated state of the epidemic.
    """

    # cast previous event to tf.int so we can slice
    transition_colummn_index = tf.cast(next_event.transition, dtype=tf.int32)

    individual_index = tf.cast(next_event.individual, dtype=tf.int32)

    # pick up columns we want to add
    column_to_add = tf.cast(incidence_matrix[:, transition_colummn_index], tf.float32)

    num_states = incidence_matrix.shape[0]

    indices = tf.stack(
        [tf.range(num_states), tf.fill(num_states, individual_index)],
        axis=-1,
    )

    new_state = tf.tensor_scatter_nd_add(
        state,
        indices,
        column_to_add,
    )

    return new_state


def _one_hot_expand_state(condensed_state: tf.Tensor) -> tf.Tensor:
    """Expand the state of the epidemic to a one-hot representation
    Args:
        epidemic_state: The state of the epidemic
    Returns:
        The one-hot representation of the epidemic state
    """
    # Create one-hot encoded vectors for each state
    one_hot_states = tf.one_hot(
        tf.range(len(condensed_state)),
        depth=len(condensed_state),
        dtype=tf.float32,
    )
    # Repeat each one-hot state based on its corresponding count
    repeated_states = tf.repeat(one_hot_states, condensed_state, axis=0)

    # Reshape and transpose to get state per row representation
    return tf.transpose(repeated_states)


def exponential_propogate(
    transition_rate_fn: Callable, incidence_matrix: Tensor
) -> EpidemicEvent:
    """Generates a function for propogating an epidemic forward in time

    Closure over the transition rate function and the incidence matrix
    which outline the epidemic dynamics and model structure. The returned
    function can be used to simulate the epidemic forward in time one step.

    Args:
        transition_rate_fn (Callable): a function that takes the current
            state of the epidemic and returns the transition rates for each
            individual/meta-population.
        incidence_matrix (tensor): A matrix that describes the graph structure
            of the epidemic. The rows correspond to the transitions and the columns
            correspond to the individuals/meta-populations.

    Returns:
        EpidemicEvent: A NamedTuple that describes the next event in the epidemic.
    """

    def propogate_fn(time: float, state: Tensor, seed: int) -> List:

        # NEED TO FIX THIS TYPING
        # Before: -> List[float, Tensor], EpidemicEvent

        """Propogates the state of the epidemic forward in time

        Args:
            time (float): Wall clock of the epidemic - can easily recover
            the time delta
            state (tensor): The current state of the epidemic - one hot
            encoding of the individuals/meta-populations.

        Returns:
            EpidemicEvent: The next event in the epidemic.
        """
        # seed management
        seeds = tfp.random.split_seed(
            seed, n=2
        )  # Split the seed for the exponential draw and the categorical draw

        # pop. size - recomputed every time to allow for changing pop. size
        num_units = tf.cast(tf.reduce_sum(state), tf.int32)

        # compute event rates for all transitions and individuals
        transition_rates = transition_rate_fn(time, state)
        transition_rates = tf.math.multiply(transition_rates, state[:-1, :])
        # compute time stamp£
        t_next = tfd.Exponential(rate=tf.reduce_sum(transition_rates)).sample(
            seed=seeds[0]
        )

        # compute transition event
        categorical_id = tfd.Categorical(
            probs=tf.reshape(
                transition_rates,
                [-1],
            ),
            dtype=tf.int32,
        ).sample(seed=seeds[1])

        # compute the individual and transition type
        individual = tf.math.floormod(categorical_id, num_units)  # equivalent to %
        transition_type = tf.math.floordiv(
            categorical_id, num_units
        )  # equivalent to //

        next_event = EpidemicEvent(time + t_next, transition_type, individual)

        # state update
        new_state = _update_state(incidence_matrix, state, next_event)

        return [time + t_next, new_state], next_event

    return propogate_fn


def continuous_markov_simulation(
    transition_rate_fn: Callable,
    state: Tensor,
    incidence_matrix: Tensor,
    num_markov_jumps: int,
    time: float = 0.0,
    seed=None,
) -> EpidemicEvent:
    """
    Simulates a continuous-time Markov process

    Args:
        transition_rate_fn (Callable): A function that computes the transition rates
            given the current state and incidence matrix.
        state (Tensor): The initial state of the system.
        num_iterations (int): The number of iterations to simulate.
        incidence_matrix (Tensor): The incidence matrix representing the connections
            between individuals in the system.
        seed (Optional[int]): The random seed for reproducibility. This should be a

    Returns:
        EpidemicEvent: An object containing the simulated epidemic events.

    """

    seed = tfp.random.sanitize_seed(seed, salt="continuous_markov_simulation")

    # I don't like that I have to call incidence_matrix twice here - redundancy
    propagate = exponential_propogate(
        transition_rate_fn=transition_rate_fn, incidence_matrix=incidence_matrix
    )

    state = _one_hot_expand_state(state)

    accum = EpidemicEvent(
        time=tf.TensorArray(DTYPE, size=0, dynamic_size=True),
        transition=tf.TensorArray(tf.int32, size=0, dynamic_size=True),
        individual=tf.TensorArray(tf.int32, size=0, dynamic_size=True),
    )

    def cond(i, *_):
        return i < num_markov_jumps

    def body(i, time, expanded_state, seed, accum):
        next_seed, this_seed = tfp.random.split_seed(seed, salt="body")
        next_state, event = propagate(time, expanded_state, this_seed)

        # processing of intermediate quantities
        next_time = next_state[0]
        next_expanded_state = next_state[1]

        # output = output.write(**event)

        accum = EpidemicEvent(*[x.write(i, y) for x, y in zip(accum, event)])

        return i + 1, next_time, next_expanded_state, next_seed, accum

    _, _, _, _, output = tf.while_loop(
        cond, body, loop_vars=(0, time, state, seed, accum)
    )

    return tf.nest.map_structure(lambda x: x.stack(), output)
