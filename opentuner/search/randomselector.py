# Random Configuration Selector
#
# Author: Hans Giesen (giesen@seas.upenn.edu)
#######################################################################################################################

import abc
import logging

from .bandittechniques import AUCBanditMetaTechnique
from .differentialevolution import DifferentialEvolutionAlt
from .evolutionarytechniques import NormalGreedyMutation, UniformGreedyMutation
from .learningtechniques import ConfigurationSelector
from .modeltuner import ModelTuner
from .simplextechniques import RandomNelderMead

log = logging.getLogger(__name__)


class RandomSelector(ConfigurationSelector):
  """
  Configuration selector that tries the optimal configuration according to the model unless it has already been tried,
  in which case a random configuration is chosen.
  """

  __metaclass__ = abc.ABCMeta

  def select_configuration(self):
    """
    Suggest a new configuration to evaluate.  If the configuration that the model thinks is best has not been tried
    yet, we will suggest it.  Otherwise, we suggest a random configuration.
    """

    # Suggest a random configuration if we don't have any data points yet.
    if len(self.data_set) == 0:
      return self.manipulator.random()

    technique = AUCBanditMetaTechnique([
        DifferentialEvolutionAlt(),
        UniformGreedyMutation(),
        NormalGreedyMutation(mutation_rate = 0.3),
        RandomNelderMead(),
      ])
    tuner = ModelTuner(self.model, technique, self.objective, self.manipulator)
    best_cfg = tuner.tune()

    # Return the best configuration if it has not been evaluated yet.
    if best_cfg not in self.data_set:
      return best_cfg

    # Return a random configuration if we have already tried the best configuration.
    return self.manipulator.random()

