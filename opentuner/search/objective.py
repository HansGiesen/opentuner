import abc
import logging
import math

from fn import _

import opentuner
from opentuner.resultsdb.models import *
from sqlalchemy.sql.expression import func

log = logging.getLogger(__name__)


class SearchObjective(object):
  """
  delegates the comparison of results and configurations
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def result_order_by_terms(self):
    """return database columns required to order by the objective"""
    return []

  @abc.abstractmethod
  def result_compare(self, result1, result2):
    """cmp() compatible comparison of resultsdb.models.Result"""
    return

  def config_compare(self, config1, config2):
    """cmp() compatible comparison of resultsdb.models.Configuration"""
    return self.result_compare(self.driver.results_query(config=config1).one(),
                               self.driver.results_query(config=config2).one())

  @abc.abstractmethod
  def result_relative(self, result1, result2):
    """return None, or a relative goodness of resultsdb.models.Result"""
    return

  def config_relative(self, config1, config2):
    """return None, or a relative goodness of resultsdb.models.Configuration"""
    return self.result_relative(self.driver.results_query(config=config1).one(),
                                self.driver.results_query(config=config2).one())


  def __init__(self):
    self.driver = None

  def set_driver(self, driver):
    self.driver = driver

  def result_order_by(self, q):
    return q.order_by(*self.result_order_by_terms())

  def compare(self, a, b):
    """cmp() compatible compare"""
    if isinstance(a, Configuration):
      return self.config_compare(a, b)
    if isinstance(a, Result):
      return self.result_compare(a, b)
    assert False

  def relative(self, a, b):
    if isinstance(a, Configuration):
      return self.config_relative(a, b)
    if isinstance(a, Result):
      return self.result_relative(a, b)
    assert None

  def lt(self, a, b):
    return self.compare(a, b) < 0

  def lte(self, a, b):
    return self.compare(a, b) <= 0

  def gt(self, a, b):
    return self.compare(a, b) > 0

  def gte(self, a, b):
    return self.compare(a, b) >= 0

  def min(self, *l):
    if len(l) == 1:
      l = l[0]
    rv = l[0]
    for i in l[1:]:
      if self.lt(i, rv):
        rv = i
    return rv

  def max(self, *l):
    if len(l) == 1:
      l = l[0]
    rv = l[0]
    for i in l[1:]:
      if self.gt(i, rv):
        rv = i
    return rv

  def limit_from_config(self, config):
    """
    a time limit to kill a result after such that it can be compared to config
    """
    results = self.driver.results_query(config=config)
    if results.count() == 0:
      return None
    else:
      return max(map(_.run_time, self.driver.results_query(config=config)))


  def project_compare(self, a1, a2, b1, b2, factor=1.0):
    """
    linearly project both a and b forward to see how they will compare in the
    future
    """
    a3 = Result()
    b3 = Result()
    a3.run_time = _project(a1.run_time, a2.run_time, factor)
    a3.accuracy = _project(a1.accuracy, a2.accuracy, factor)
    a3.energy = _project(a1.energy, a2.energy, factor)
    a3.confidence = _project(a1.confidence, a2.confidence, factor)
    return self.result_compare(a3, b3)

  def display(self, result):
    """
    produce a string version of a resultsdb.models.Result()
    """
    rv = []
    for k in ('run_time', 'accuracy', 'energy', 'size', 'confidence'):
      v = getattr(result, k)
      if v is not None:
        rv.append('%s=%.4f' % (k, float(v)))
    return ', '.join(rv)

  def filter_acceptable(self, query):
    """Return a Result() query that only returns acceptable results"""
    return query

  def is_acceptable(self, result):
    """Test if a Result() meets thresholds"""
    return True

  def stats_quality_score(self, result, worst_result, best_result):
    """return a score for statistics"""
    if not self.is_acceptable(result):
      return worst_result.run_time
    else:
      return result.run_time


def _project(a1, a2, factor):
  if a1 is None or a2 is None:
    return None
  return a2 + factor * (a2 - a1)


class MinimizeTime(SearchObjective):
  """Minimize runtime.

  Attributes
  ----------
  opt_metrics : list of str
    Names of metrics to be optimized
  constrained_metrics : list of str
    Names of metrics to be constrained
  cosntraints : list of int
    Values of constraints
  """
  opt_metrics = ['run_time']
  constrained_metrics = ()
  constraints = ()

  def result_order_by_terms(self):
    """return database columns required to order by the objective"""
    return [Result.run_time]

  def result_compare(self, result1, result2):
    """cmp() compatible comparison of resultsdb.models.Result"""
    return cmp(result1.run_time, result2.run_time)

  def config_compare(self, config1, config2):
    """cmp() compatible comparison of resultsdb.models.Configuration"""
    return cmp(min(map(_.run_time, self.driver.results_query(config=config1))),
               min(map(_.run_time, self.driver.results_query(config=config2))))

  def result_relative(self, result1, result2):
    """return None, or a relative goodness of resultsdb.models.Result"""
    if result2.run_time == 0:
      return float('inf') * result1.run_time
    return result1.run_time / result2.run_time


class MaximizeAccuracy(SearchObjective):
  """
  maximize Result().accuracy
  """

  def result_order_by_terms(self):
    """return database columns required to order by the objective"""
    return [-Result.accuracy]

  def result_compare(self, result1, result2):
    """cmp() compatible comparison of resultsdb.models.Result"""
    # note opposite order
    return cmp(result2.accuracy, result1.accuracy)

  def result_relative(self, result1, result2):
    """return None, or a relative goodness of resultsdb.models.Result"""
    # note opposite order
    if result1.accuracy == 0:
      return float('inf') * result2.accuracy
    return result2.accuracy / result1.accuracy

  def stats_quality_score(self, result, worst_result, best_result):
    """return a score for statistics"""
    if not self.is_acceptable(result):
      return worst_result.run_time
    else:
      return result.run_time

  def stats_raw_score(self, result):
    return result.accuracy


class MaximizeAccuracyMinimizeSize(MaximizeAccuracy):
  """
  maximize Result().accuracy, break ties with Result().size
  """

  def result_order_by_terms(self):
    """return database columns required to order by the objective"""
    return [-Result.accuracy, Result.size]

  def result_compare(self, result1, result2):
    """cmp() compatible comparison of resultsdb.models.Result"""
    return cmp((-result1.accuracy, result1.size),
               (-result2.accuracy, result2.size))

  def display(self, result):
    """
    produce a string version of a resultsdb.models.Result()
    """
    return "accuracy=%.8f, size=%.1f" % (result.accuracy, result.size)

  def result_relative(self, result1, result2):
    """return None, or a relative goodness of resultsdb.models.Result"""
    # unimplemented for now
    log.warning('result_relative() not yet implemented for %s',
                self.__class__.__name__)
    return None


class ThresholdAccuracyMinimizeTime(SearchObjective):
  """
  if accuracy >= target:
    minimize run_time
  else:
    maximize accuracy
  """

  def __init__(self, accuracy_target, low_accuracy_limit_multiplier=10.0):
    self.accuracy_target = accuracy_target
    self.low_accuracy_limit_multiplier = low_accuracy_limit_multiplier
    super(ThresholdAccuracyMinimizeTime, self).__init__()

  def result_order_by_terms(self):
    """return database columns required to order by the objective"""

    return ["min(accuracy, %f) desc" % self.accuracy_target,
            opentuner.resultsdb.models.Result.run_time]

  def result_compare(self, result1, result2):
    """cmp() compatible comparison of resultsdb.models.Result"""
    return cmp((-min(self.accuracy_target, result1.accuracy),
                result1.run_time),
               (-min(self.accuracy_target, result2.accuracy), result2.run_time))

  def config_compare(self, config1, config2):
    """cmp() compatible comparison of resultsdb.models.Configuration"""
    return self.result_compare(
      self.driver.results_query(config=config1, objective_ordered=True)[0],
      self.driver.results_query(config=config2, objective_ordered=True)[0])

  def limit_from_config(self, config):
    """
    a time limit to kill a result after such that it can be compared to config
    """
    results = self.driver.results_query(config=config)
    if results.count() == 0:
      return None
    if self.accuracy_target > min(map(_.accuracy, results)):
      m = self.low_accuracy_limit_multiplier
    else:
      m = 1.0
    return m * max(map(_.run_time, results))


  def filter_acceptable(self, query):
    """Return a Result() query that only returns acceptable results"""
    return query.filter(opentuner.resultsdb.models.Result.accuracy
                        >= self.accuracy_target)

  def is_acceptable(self, result):
    """Test if a Result() meets thresholds"""
    return result.accuracy >= self.accuracy_target

  def result_relative(self, result1, result2):
    """return None, or a relative goodness of resultsdb.models.Result"""
    # unimplemented for now
    log.warning('result_relative() not yet implemented for %s',
                self.__class__.__name__)
    return None


class ThresholdAreaMinimizeTime(SearchObjective):
  """Minimize runtime under area constraints.

  If the area constraint cannot be met, the objective minimizes area instead.
  
  Attributes
  ----------
  opt_metrics : list of str
    Names of metrics to be optimized
  constrained_metrics : list of str
    Names of metrics to be constrained
  constraints : tuple with 4 ints
    Resource constraints on the LUTs, CLB registers, DSPs, and BRAM18s
    respectively
  """
  opt_metrics = ['run_time']
  constrained_metrics = ("luts", "regs", "dsps", "brams")

  def __init__(self, constraints):
    self.constraints = constraints
    super(ThresholdAreaMinimizeTime, self).__init__()

  def result_compare(self, result1, result2):
    """cmp() compatible comparison of resultsdb.models.Result"""
    return cmp((max(result1.luts  / self.constraints[0],
                    result1.regs  / self.constraints[1],
                    result1.dsps  / self.constraints[2],
                    result1.brams / self.constraints[3],
                    1.0),
                result1.run_time),
               (max(result2.luts  / self.constraints[0],
                    result2.regs  / self.constraints[1],
                    result2.dsps  / self.constraints[2],
                    result2.brams / self.constraints[3],
                    1.0),
                result2.run_time))

  def result_order_by_terms(self):
    """return database columns required to order by the objective"""
    expressions = []
    for metric, constraint in zip(self.constrained_metrics, self.constraints):
      if not math.isinf(constraint):
        expressions.append(getattr(Result, metric) / constraint)
    return [func.greatest(1.0, *expressions), Result.run_time]

  def result_relative(self, result1, result2):
    """return None, or a relative goodness of resultsdb.models.Result"""
    raise NotImplementedError()
