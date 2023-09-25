"""Functions for continuous time simulation."""

###############################################################################
# PREAMBLE
###############################################################################
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp

tla = tf.linalg
tfd = tfp.distributions

EventTrace = namedtuple('EventTrace', 'time state transition unit')


def expand_state(epidemic_state):
    """Expands an epidemic state into a one-hot sparse tensor representation.

    The epidemic_state is vector of length number of states containing the count
    for the number of units in each state at a given time point. The state is
    expanded to a sparse tensor where each column corresponds to a one-hot vector
    specifying which of the R states the individual is in. 

    Args:
        epidemic_state (int): A shape `[R]` vector containing integer counts of the
            number of units in each state.
    Returns:
        int: A shape `[R,N]` tensor representing the expanded epidemic state, where
            N is the total number of units in the population.

    Example:
        >>> epidemic_state = [3, 1, 0]
        >>> expanded_state = expand_state(epidemic_state)
        >>> print(expanded_state)
        [[1., 0., 0.],
         [1., 0., 0.],
         [1., 0., 0.],
         [0., 1., 0.]]
    """
    # enumerate the states to pick up the location where the 1 needs to be
    state_enumerator = tf.range(start=0,
                                limit=len(epidemic_state))
    # expand the locations of the 1s by repeating them epidemic_state number of times
    indices = tf.repeat(input=state_enumerator,
                        repeats=epidemic_state,
                        axis=0)
    # generate the one hot vectors for each individual (represented by indices) and
    # the dim of each one hot vector is equal to the len(epidemic_state)
    row_wise_expanded_form = tf.one_hot(
        indices=indices, depth=len(epidemic_state))

    # transpose the out
    col_wise_expanded_form = tf.transpose(row_wise_expanded_form)
    return col_wise_expanded_form


def update_sparse_epidemic_state(
        stoichiometry, epidemic_state, unit_id, transition_id):
    """Update a sparse epidemic state

    Use the  stoichiometry matrix to pick up the state update for a given
    transition. Each transition correponds to adding the column vector of
    the stoichiometry matrix to the column representing the unit in the 
    population undergoing that transition.

    Args:
        stoichiometry (float): a `[R, M]` matrix describing the state update
                for each transition
        epidemic_state (float): a sparse `[R, N]` matrix encoding the state
                each individual is in 
        unit_id (float): scalar indicating the unit undergoing the event
        transition_id (float): a scalar indicating the transition type

    Returns:
        float: a sparse `[R, N]` matrix encoding the new state of the epidemic 

    Example: 
    >>> stoichiometry = np.array([[-1, 0],
                                [1, -1],
                                [0, 1]], dtype=np.float32)
    >>> epidemic_state = epidemic_state = [3,1,0])
    >>> update_sparse_epidemic_state(epidemic_state = epidemic_state,
                             individual_ID = 2,
                             transition_ID=0)
    [[1., 1., 1., 0.],
     [0., 0., 0., 1.],
     [0., 0., 0., 0.]]                        
    """
    # cast the categorical samples to type int for working with tf.gather
    transition_id = tf.cast(transition_id, dtype=tf.int32)
    unit_id = tf.cast(unit_id, dtype=tf.int32)

    state_change = tf.gather(params=stoichiometry,
                             indices=transition_id, axis=-1)

    indices = tf.stack(
        # collect the index for *all* rows of the epidemic state
        # note that the num rows stoichiometry = num rows epidemic state
        [tf.range(tf.shape(state_change)[-1]),
            # repeat the individual_id num rows many times
            tf.broadcast_to(
                unit_id, shape=[tf.shape(state_change)[-1]])],
        axis=-1)
    # stack along the last axis to get a tensor of dimension [num rows epidemic state, 2]
    # where the first column contains the row indices and the 2nd column contains the
    # column index we add tp

    next_epidemic_state = tf.tensor_scatter_nd_add(tensor=epidemic_state,
                                                   indices=indices,
                                                   updates=state_change)

    return next_epidemic_state


def compute_source_states(stoichometry, dtype=tf.int32):
    """Computes the source states for a given stoichiometry matrix.

    Compute the states of in which each transition originates from. This
    can be interpreted as which state the unit was in at the current state.

    Args:
        stoichometry (tf.Tensor): A stoichiometry matrix of shape `[R,N]`, where
            R is the number of state transitions and N is the number of states.
        dtype (tf.DType): The data type of the output tensor.

    Returns:
        tf.Tensor: A tensor of shape `[R]`, representing the source states for each
            transition.

    Example:
        >>> stoichiometry = tf.constant([[-1, 1, 0], [0, -1, 1]])
        >>> source_states = compute_source_states(stoichometry)
        >>> print(source_states)
        tf.Tensor([0 1], shape=(2,), dtype=int32)
    """
    # TO DO: add comments to explain the intermediate steps here!
    source_states = tf.reduce_sum(
        tf.cumsum(
            tf.clip_by_value(
                -stoichometry, clip_value_min=0, clip_value_max=1
            ),
            axis=-2,
            reverse=True,
            exclusive=True,
        ),
        axis=-2,
    )
    return tf.cast(source_states, dtype)


def continuous_time_propogate(hazard_rate_fn, stoichiometry):
    """Propogates the state of the population according to the continuous time epidemic dyanmics 

    Create a closure around the hazard rate function/epidemic dynamics and
    the stoichiometry matrix which propograte an epidemic state forward one
    event time. 

    Args:
        hazard_rate_fn (fn): function returning the transition rate between 
                    states. This function   returns a [R,N] tensor of floats  
        stoichiometry (tf.tensor): A stoichiometry matrix of shape `[R,N]`, where
            R is the number of state transitions and N is the number of states.

    Returns:
        fn: a one step fn which moves the epidemic state forward one event
    """
    source_states = compute_source_states(stoichiometry)

    def propagate_fn(time, epidemic_state, covars=0):
        """Update the epidemic state with a next event

        Implement a single iteration of the Gillespie algorithm for a continuous
        time epidemic model. The overall event rate is calculated as the sum of
        all possible transitions, and the next event time is simulated from an
        exponential distribution. The unit undergoing the event, as well as the
        type of event is then sampled from a categorical distribution (this allows
        for flexibility for meta-population or individual level models). Lastly,
        the state is updated to reflect the new event.

        Args:
            time (float): timestamp of the epidemic_state
            epidemic_state (tf.tensor): a sparse `[R, N]` matrix encoding 
                        the state each individual is in 
            covars (int, optional): Hi Jess :) Help! . Defaults to 0.

        Returns:
            next_time (float): timestamp of the epidemic_state
            next_state (tf.tensor): a sparse `[R, N]` matrix encoding 
                        the state each individual is in 
            next_covars (tf.tensor): Hi Jess :) Help!
        """
        # should this be pulled out since it is static? Assume closed pop.
        num_units = tf.reduce_sum(epidemic_state)

        # 1. Compute the hazard rate for the epidemic_state
        rates = hazard_rate_fn(time, epidemic_state)
        rates = tf.multiply(rates, tf.gather(epidemic_state, source_states))

        # 2. Sample the time of the next event and the unit undergoing it
        system_event_rate = tf.reduce_sum(rates)

        # sample the time component
        t_star = tfd.Exponential(rate=system_event_rate).sample()
        next_time = time + t_star

        # sample the unit component <- allocate to a function?
        categorical_id = tfd.Categorical(probs=tf.reshape(rates, [-1])).sample()

        individual_id = tf.math.floormod(
            num_units, categorical_id, name='get_individual_id')
        transition_id = tf.math.floordiv(
            num_units, categorical_id, name='get_transition_id')

        # 3. Update the next state
        next_state = update_sparse_epidemic_state(stoichiometry=stoichiometry,
                                                  epidemic_state=epidemic_state,
                                                  unit_id=individual_id,
                                                  transition_id=transition_id)

        next_covars = covars

        track_event = EventTrace(next_time, next_state,
                                 transition_id, individual_id)

        return next_time, next_state, next_covars, track_event
    return propagate_fn


def continuous_markov_simulation(initial_state):
    # this function implements the tf.while loop to generate 1 path of the epidemic
    initial_state = expand_state(initial_state)

    # initiate the EventTrace
    initial_trace = EventTrace(time=tf.constant(0., dtype=tf.float64),
                               state=initial_state,
                               transition=tf.constant(0., dtype=tf.float64),
                               unit=tf.constant(0., dtype=tf.float64))

    # create TensorArray to store epidemic progression
    epidemic_trace = tf_map(
        lambda x: tf.TensorArray(dtype=x.dtype, size=1, dynamic_size=True),
        initial_trace)

    # create the body fn for the while loop

    # create the stop condition (cond fn) for the while loop

    # run an iteration

    return epidemic_trace
