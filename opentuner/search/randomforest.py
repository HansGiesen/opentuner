# Random Forest Model
#
# Author: Hans Giesen (giesen@seas.upenn.edu)
#######################################################################################################################

import logging
import sys

from .learningtechniques import LearningTechnique, Model
from .randomselector import RandomSelector
from .technique import register
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger(__name__)


class RandomForest(Model):
  """
  Random Forest Model
  """

  def __init__(self):
    """
    Initialize the random forest model object.
    """

    # Assume the data set is empty.
    self.data_size = 0

    # Create a random forest object.
    self.regressor = RandomForestRegressor()


  def predict(self, cfg):
    """
    Predict the result for the given configuration.
    """

#    log.info("Making a prediction for %s...", str(cfg.values()))

    # Retrain the regressor if the data set has changed.
    if len(self.data_set) != self.data_size:
      log.info("Retraining model with %d samples.", len(self.data_set))
      # Retrain the regressor.
      X = [result.configuration.data.values() for result in self.data_set]
      y = [result.run_time for result in self.data_set]
      y[y == float('inf')] = sys.float_info.max
      self.regressor.fit(X, y)
      # Update the size of the data set.
      self.data_size = len(self.data_set)

    # Make a prediction for the configuration.
    result = self.regressor.predict([cfg.values()])[0]

#    log.info("Prediction: %s", str(result))

    return result


  def estimate_uncertainty(self, cfg):
    """
    Give a measure of uncertainty for the given configuration.  Random forests don't give uncertainty measures.
    """
    return 0


# Add the random forest model to the list such that we can use it.
register(LearningTechnique(RandomForest(), RandomSelector(), name = "RandomForest"))

