# Tuner for Regression Models
#
# Author: Hans Giesen (giesen@seas.upenn.edu)
#######################################################################################################################

import copy
import logging

from opentuner.driverbase import DriverBase
from opentuner.resultsdb.models import Configuration, Result

log = logging.getLogger(__name__)


class QueryResult(object):
  """
  This class is replaces the Query object returned by results_query() in the absence of a database.
  """

  def __init__(self, result):
    self.result = result
    self.done = False

  def __iter__(self):
    return self

  def next(self):
    if self.done:
      raise StopIteration
    self.done = True
    return self.result

  def count(self):
    return 1


class ModelDriver(DriverBase):
  """
  Driver with necessary callbacks for ModelTuner class
  """

  def __init__(self, model, objective, manipulator):
    """
    Create a new driver object.
    """

    # Create an empty dictionary for search results.
    self.results = {}

    # Set the search objective.
    self.objective = objective
    # Set the parameter manipulator.
    self.manipulator = manipulator
    
    # Initialize the best result found so far.
    self.best_result = None

    # Some techniques need these attributes.
    self.generation = None
    self.tuning_run = None


  def add_plugin(self, plugin):
    """
    Callback to install a plugin into the search driver.
    """
    # We don't support plugins in this driver, so we ignore any call to this function.
    pass


  def has_results(self, cfg):
    """
    Callback to check whether results for the given configuration are ready.
    """
    return cfg.hash in self.results


  def get_configuration(self, cfg):
    """
    Callback for creating Configuration objects
    """
    self.manipulator.normalize(cfg)
    hash = self.manipulator.hash_config(cfg)
    return Configuration(hash = hash, data=cfg)


  def add_result(self, result):
    """
    Remember a result such that search techniques can look it up.
    """
    self.results[result.configuration.hash] = result


  def results_query(self, config):
    """
    Look up the result obtained for a given configuration.
    """
    return QueryResult(self.results[config.hash])


  def register_result_callback(self, desired_result, callback):
    """
    Register a callback function to handle the result of evaluating a configuration.
    """
    self.result_callback = callback


  def invoke_result_callback(self, result):
    """
    Invoke the callback function to provide the result to the search technique.
    """
    self.result_callback(result)
    
    
  def set_best_result(self, result):
    """
    Set the best result.  Some search techniques rely on this attribute.
    """
    self.best_result = result


class ModelTuner(object):
  """
  This class provides tuning functionality for the machine-learning search techniques, which can apply them to the
  regression models.
  """

  def __init__(self, model, technique, objective, manipulator):
    """
    Create a new tuner object.
    """

    # Copy the objective because we need to use our own driver.
    self.objective = copy.copy(objective)

    # Create a new driver.
    self.driver = ModelDriver(model, self.objective, manipulator)

    # Model that has to be tuned
    self.model = model

    # Search technique to be applied
    self.technique = technique
    # Inform the search technique about the driver.
    self.technique.set_driver(self.driver)

    # Tell the objective to use our driver.
    self.objective.set_driver(self.driver)


  def tune(self):

    best_result = None
    for iter in range(100):
      desired_result = self.technique.desired_result()
      if desired_result == None:
        # desired_result is None if a technique keeps suggesting configurations that have already been tested.
        break
#      if desired_result == False:
#        raise RunTimeError("The search technique returns False.  I wasn't sure what to do with that.")
      cfg = desired_result.configuration
      result = Result()
      result.configuration = cfg
      result.run_time = self.model.predict(cfg.data)
      self.driver.add_result(result)
      self.driver.invoke_result_callback(result)
      if best_result == None or self.objective.lt(result, best_result):
        best_result = result
      self.driver.set_best_result(best_result)

    return best_result.configuration

