"Simualate an epidemic to as our base case for log-likehood evaluation"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

from continuous_markov import EpidemicEvent, continuous_markov_simulation

# aliasing for convenience
tfd = tfp.distributions
Tensor = tf.Tensor
DTYPE = tf.float32

# define the epidemic simulation parameters
initial_population = np.array([4, 1, 0])
population_size = np.sum(initial_population)

incidence_matrix = np.array(
    [
        [-1, 0, 0],
        [1, -1, 0],
        [0, 1, 0],
    ]
)


def rate_fn(t, state):
    """
    Calculates the rates of infection and recovery based on the current state.

    Args:
        t (float): The current time.
        state (tf.Tensor): The current state of the system.

    Returns:
        tuple: A tuple containing the infection rate and recovery rate.

    Raises:
        None

    """
    # Calculate the infection rate based on the density of susceptible individuals
    si_rate = 0.2 * state[0, :] / tf.reduce_sum(state, axis=0)

    # The recovery rate is constant for all individuals
    ir_rate = tf.broadcast_to([0.14], si_rate.shape)

    return si_rate, ir_rate


# simulate the epidemic
small_scale_epidemic = continuous_markov_simulation(
    transition_rate_fn=rate_fn,
    state=initial_population,
    incidence_matrix=incidence_matrix,
    num_markov_jumps=10,
    seed=[1, 0],
)

# plotting the epidemic - want this in another file?


def event_list_to_table(
    event: EpidemicEvent, num_individuals: int, num_transitions: int
):
    """
    Converts an EpidemicEvent list into a table representation with columns representing the transition type and the rows representing the individuals.

    Args:
        event (EpidemicEvent): The list of epidemic events.
        num_individuals (int): The number of individuals.
        num_transitions (int): The number of transitions.

    Returns:
        tf.Tensor: The table representation of the epidemic events.

    """
    blank = tf.fill(dims=[num_individuals, num_transitions], value=np.inf)

    update = event.time
    valid_events = tf.math.is_inf(update)
    valid_events = tf.cast(valid_events, tf.bool)  # Convert to boolean type
    update = tf.boolean_mask(update, ~valid_events)

    indices = tf.stack([event.individual, event.transition], axis=-1)
    indices = indices[0 : len(update), :]  # remove null events

    return tf.tensor_scatter_nd_update(blank, indices, update)


event_times_for_individuals = event_list_to_table(
    event=small_scale_epidemic,
    num_individuals=population_size,
    num_transitions=2,
)

line_plot_data = np.hstack(
    [
        tf.cast(
            tf.expand_dims(
                np.arange(
                    tf.reduce_sum(initial_population),
                ),
                axis=-1,
            ),
            DTYPE,
        ),
        event_times_for_individuals,
    ]
)
print(line_plot_data)
# create and export the figure
sns.set_theme()
sns.set_palette("muted")

fig = plt.figure(figsize=(8, 12))
ax = fig.add_subplot()

ax.scatter(line_plot_data[:, 1], line_plot_data[:, 0], color="red", s=25)
ax.scatter(line_plot_data[:, 2], line_plot_data[:, 0], color="green", s=25)

plt.yticks(np.arange(population_size))

# plot susceptible period
for ii in range(population_size):
    x = [0, line_plot_data[ii, 1]]
    y = [ii, ii]
    ax.plot(x, y, "b-", alpha=1, linewidth=1, label="Sus")

# plot infective period
for ii in range(population_size):
    x = [line_plot_data[ii, 1], line_plot_data[ii, 2]]
    y = [ii, ii]
    ax.plot(x, y, "r-", alpha=1, linewidth=3, label="Inf")

# plot removed period
for ii in range(population_size):
    x = [line_plot_data[ii, 2], np.max(line_plot_data[:, 2])]
    y = [ii, ii]
    ax.plot(x, y, "g-", alpha=1, linewidth=1, label="Rem")

# plot initial infective period
x_initial_inf = [0, line_plot_data[4, 2]]
y_initial_inf = [4, 4]

ax.plot(x_initial_inf, y_initial_inf, "r:", alpha=1, linewidth=3, label="Initial Inf")

legend_elements = [
    Line2D([0], [0], color="b", lw=3, label="Sus"),
    Line2D([0], [0], color="r", lw=3, label="Inf"),
    Line2D([0], [0], color="g", lw=3, label="Rem"),
    Line2D([0], [0], color="r", lw=3, linestyle=":", label="Initial Inf"),
]

plt.xlabel("Time")
plt.ylabel("Individual")
plt.title(f"{population_size} unit epidemic progression")

plt.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.999),
    ncol=4,
    fancybox=True,
    shadow=True,
)
ax.plot()
plt.savefig("small_scale_epidemic.png")
