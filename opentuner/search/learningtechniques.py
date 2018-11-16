# Machine Learning Search Techniques
#
# <Description>
#
# Author: Hans Giesen (giesen@seas.upenn.edu)
#######################################################################################################################

import abc
import logging

from .technique import SearchTechnique

log = logging.getLogger(__name__)


class LearningTechnique(SearchTechnique):
  """
  Machine Learning Search Technique
  """

  def __init__(self, model, *pargs, **kwargs):
    """
    Initialize the machine learning search technique object.
    """

    # Call the constructor of the parent object.
    super(LearningMetaTechnique, self).__init__(*pargs, **kwargs)

    # Remember machine-learning model to use.
    self.model = model

    # Remember configuration selector to use.
    self.selector = selector

    # Create an empty dataset.
    self.data_set = set()

    # Provide the dataset to the model.
    model.set_data_set(self.data_set)


  def desired_configuration(self):
    """
    Suggest a new configuration to evaluate.
    """

    # Return new configuration chosen by selector.
    return selector.select_configuration()


  def handle_requested_result(self, result):
    """
    This callback is invoked by the search driver to report new results.
    """

    # Add a new result to the data set.
    self.data_set.add(result)


class Model(object):
  """
  Abstract base class for machine-learning models
  """

  __metaclass__ = abc.ABCMeta

  def set_data_set(self, data_set):
    """
    Set data set for model.
    """

    # Store the data set.
    self.data_set = data_set


  @abc.abstractmethod
  def predict(self, configuration):
    """
    Predict the result for the given configuration.
    """


  @abc.abstractmethod
  def estimate_uncertainty(self, configuration):
    """
    Give a measure of uncertainty for the given configuration.
    """


class ConfigurationSelector(object):
  """
  Abstract base class for selecting configurations based on machine learning
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, model, *pargs, **kwargs):
    """
    Initialize the configuration selector object.
    """

    # Remember machine-learning model to use.
    self.model = model


  @abc.abstractmethod
  def select_configuration(self):
    """
    Suggest a new configuration to evaluate.
    """

