import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp 
tfd = tfp.distributions

class ContinuousTimeIndividualLevelEpidemicModel(tfd.Distribution):

    def __init__(self,
        transition_rates, 
        incident_matrix,
        num_steps,
        name = 'CtsTimeStateTransitionModel'):
        """
        :param transition_rates:        a function of the form 'fn(t, state)' which takes in a time stamp and the state of
                                        the epidemic, and returns the infectious pressure on each individual in the population

        :param incident_matrix:         a stoichimetry matrix for the state transition model. The states of the models are
                                        along the rows and the transition events are across the columns. Eg. in a typical
                                        SIR model the incident matrix would be 3x2 with entries [[-1, 0], [1,-1], [0,1]]
        
        :param num_steps:               the number of time steps the model runs 
        """

        parameters = dict(locals())

    @property
    def transition_rates(self):
        return self._transition_rates

    @property
    def indicent_matrix(self):
        return self._indicent_matrix

    @property
    def num_steps(self):
        return self._num_steps

    