"""State management"""

import tensorflow as tf

from Epidemic import Epidemic

def _one_hot_expand_state(epidemic_state: Epidemic) -> tf.Tensor:
    """Expand the state of the epidemic to a one-hot representation
    Args:
        epidemic_state: The state of the epidemic
    Returns:
        The one-hot representation of the epidemic state
    """
    state_vector = epidemic_state.state_count
    
    # Create one-hot encoded vectors for each state
    one_hot_states = tf.one_hot(tf.range(len(state_vector)),
                                depth=len(state_vector),
                                dtype=tf.float32)
    # Repeat each one-hot state based on its corresponding count
    repeated_states = tf.repeat(one_hot_states, state_vector, axis=0)

    # Reshape and transpose to get state per row representation
    return tf.transpose(repeated_states)


def _compute_source_states(incidence_matrix: tf.Tensor,
                             dtype: tf.DType = tf.float32) -> tf.Tensor:
    """
    Computes the indices of the source states for each transition in a state
    transition model represented by an incidence matrix.   
     
    Args:
      incidence_matrix: A tensor of shape `[num_states, num_transitions]`
          representing the state transition model. Each element `(i, j)`
          indicates the transition from state `i` to state `j`. Negative values
          indicate transitions occurring out of the system.
      dtype: The desired output data type, defaults to `tf.float32`.    
    
    Returns:
      A tensor of shape `[num_transitions]` containing the source state indices
      for each transition.  
      
    Raises:
      ValueError: If the input tensor has incorrect dimensions or non-negative
          values beyond the diagonal.

    Example usage:
    incidence_matrix = tf.constant([[-1, 0], 
                                    [1, -1],
                                    [0, 1]])
    source_states = _compute_source_states(incidence_matrix)
    print(source_states)  >> Output: [0, 1]
    """

    if not tf.is_tensor(incidence_matrix):
        raise ValueError("incidence_matrix must be a TensorFlow tensor.")

    if incidence_matrix.shape.rank != 2:
        raise ValueError("incidence_matrix must have exactly two dimensions.")
    
    clipped_matrix = tf.clip_by_value(-incidence_matrix, 0, 1)
    
    condense_states = tf.cumsum(clipped_matrix,
                                axis=-2,
                                reverse=True,
                                exclusive=True
                                )
    
    source_states = tf.reduce_sum(condense_states, axis=-2)
        
    if not tf.equal(tf.reduce_max(clipped_matrix, axis=-1), 1).all():
        raise ValueError("Invalid incidence matrix: Each state must have one incoming"
                       " transition (excluding diagonal).")

    return tf.cast(source_states, dtype)
