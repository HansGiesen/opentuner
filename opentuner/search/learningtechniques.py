# Machine Learning Search Techniques
#
# Base classes for search techniques that use machine-learning models
#
# TODO: Get rid of ConfigurationSelector.  Integrate it into LearningTechnique instead.
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

  def __init__(self, model, selector, *pargs, **kwargs):
    """
    Initialize the machine learning search technique object.
    """

    # Call the constructor of the parent object.
    super(LearningTechnique, self).__init__(*pargs, **kwargs)

    # Remember configuration selector to use.
    self.selector = selector

    # Create an empty dataset.
    self.data_set = set()

    # Provide the dataset to the model and configuration selector.
    model.set_data_set(self.data_set)
    selector.set_data_set(self.data_set)

    # Provide the model to the configuration selector.
    selector.set_model(model)


  def set_driver(self, driver):
    """
    Set the search driver.
    """

    # Use the parent object to perform most of the functionality.
    super(LearningTechnique, self).set_driver(driver)

    # Provide the manipulator to the configuration selector.  We need it because it contains information about the
    # parameters.
    self.selector.set_manipulator(driver.manipulator)

    # Provide the objective to the configuration selector.
    self.selector.set_objective(driver.objective)


  def desired_configuration(self):
    """
    Suggest a new configuration to evaluate.
    """

    # Return new configuration chosen by selector.
    return self.selector.select_configuration()


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
    self.data_set = data_set


  @abc.abstractmethod
  def predict(self, cfg):
    """
    Predict the result for the given configuration.
    """


  @abc.abstractmethod
  def estimate_uncertainty(self, cfg):
    """
    Give a measure of uncertainty for the given configuration.
    """


class ConfigurationSelector(object):
  """
  Abstract base class for selecting configurations based on machine learning
  """

  __metaclass__ = abc.ABCMeta

  def set_data_set(self, data_set):
    """
    Set data set.
    """
    self.data_set = data_set


  def set_model(self, model):
    """
    Set model.
    """
    self.model = model


  def set_manipulator(self, manipulator):
    """
    Set manipulator.
    """
    self.manipulator = manipulator


  def set_objective(self, objective):
    """
    Set objective.
    """
    self.objective = objective


  @abc.abstractmethod
  def select_configuration(self):
    """
    Suggest a new configuration to evaluate.
    """

