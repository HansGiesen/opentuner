import argparse
import logging
import time
import socket
import os
from multiprocessing.pool import ThreadPool
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import literal

from opentuner.driverbase import DriverBase
from opentuner.resultsdb.models import *

log = logging.getLogger(__name__)

argparser = argparse.ArgumentParser(add_help=False)
argparser.add_argument('--machine-class',
                       help="name of the machine class being run on")


class MeasurementDriver(DriverBase):
  """
  manages the measurement process, reading DesiredResults and creating Results
  """

  def __init__(self,
               measurement_interface,
               input_manager,
               **kwargs):
    super(MeasurementDriver, self).__init__(**kwargs)

    if not self.args.machine_class:
      self.args.machine_class = 'default'

    self.interface = measurement_interface
    self.input_manager = input_manager
    self.commit = self.tuning_run_main.commit
    self.upper_limit_multiplier = 10.0
    self.default_limit_multiplier = 2.0

    self.machine = self.get_machine()

    self.thread_pool = None
    self.slots = None

  def get_machine(self):
    """
    get (or create) the machine we are currently running on
    """
    hostname = socket.gethostname().split('.')[0]
    try:
      self.session.flush()
      return self.session.query(Machine).filter_by(name=hostname).one()
    except sqlalchemy.orm.exc.NoResultFound:
      m = Machine(name=hostname,
                  cpu=_cputype(),
                  cores=_cpucount(),
                  memory_gb=_memorysize() / (
                  1024.0 ** 3) if _memorysize() else 0,
                  machine_class=self.get_machine_class())
      self.session.add(m)
      return m

  def get_machine_class(self):
    """
    get (or create) the machine class we are currently running on
    """
    return MachineClass.get(self.session, name=self.args.machine_class)

  def run_time_limit(self, desired_result, default=3600.0 * 24 * 365 * 10):
    """return a time limit to apply to a test run (in seconds)"""
    best = self.results_query(objective_ordered=True).first()
    if best is None:
      if desired_result.limit:
        return desired_result.limit
      else:
        return default

    if desired_result.limit:
      return min(desired_result.limit, self.upper_limit_multiplier * best.run_time)
    else:
      return self.default_limit_multiplier * best.run_time

  def report_result(self, desired_result, result, input=None):
    result.configuration = desired_result.configuration
    result.input = input
    result.tuning_run = self.tuning_run
    result.collection_date = datetime.now()
    self.session.add(result)
    desired_result.result = result
    desired_result.state = 'COMPLETE'
    self.input_manager.after_run(desired_result, input)
    diff = result.collection_date - desired_result.start_date
    result.collection_cost = diff.total_seconds()
    self.session.flush()  # populate result.id
    log.debug(
        'Result(id=%d, cfg=%d, time=%.4f, accuracy=%.2f, collection_cost=%.2f)',
        result.id,
        result.configuration.id,
        result.run_time,
        result.accuracy if result.accuracy is not None else float('NaN'),
        result.collection_cost)
    self.commit()

  def run_desired_result(self, desired_result, compile_result=None,
                         exec_id=None):
    """
    create a new Result using input manager and measurment interface
    Optional compile_result paramater can be passed to run_precompiled as
    the return value of compile()
    Optional exec_id paramater can be passed to run_precompiled in case of
    locating a specific executable
    """
    desired_result.limit = self.run_time_limit(desired_result)

    input = self.input_manager.select_input(desired_result)
    self.session.add(input)
    self.session.flush()

    log.debug('running desired result %s on input %s', desired_result.id,
              input.id)

    self.input_manager.before_run(desired_result, input)

    if self.interface.parallel_compile:
        result = self.interface.run_precompiled(desired_result, input,
                                                desired_result.limit,
                                                compile_result, exec_id)
    else:
        result = self.interface.compile_and_run(desired_result, input,
                                                desired_result.limit)

    self.report_result(desired_result, result, input)

  def claim_desired_result(self, desired_result):
    """
    claim a desired result by changing its state to running
    return True if the result was claimed for this process
    """
    self.commit()
    try:
      self.session.refresh(desired_result)
      if desired_result.state == 'REQUESTED':
        desired_result.state = 'RUNNING'
        desired_result.start_date = datetime.now()
        self.commit()
        return True
    except SQLAlchemyError:
      self.session.rollback()
    return False

  def query_pending_desired_results(self):
    q = (self.session.query(DesiredResult)
         .filter_by(tuning_run=self.tuning_run,
                    state='REQUESTED')
         .order_by(DesiredResult.generation,
                   DesiredResult.priority.desc()))
    return q

  def process_all(self):
    """
    process all desired_results in the database
    """
    q = self.query_pending_desired_results()

    if self.interface.parallel_compile:
      desired_results = {}
      thread_args = []

      def compile_result(args):
        results = []
        for interface, data, result_id, fidelity in args:
          if fidelity == 0:
            kwargs = {}
          else:
            kwargs = {'fidelity': fidelity}
          result = interface.compile(data, result_id, **kwargs)
          results.append(result)
        return results

      for dr in q.all():
        if self.claim_desired_result(dr):
          if dr.thread is not None:
            desired_results.setdefault(dr.thread, []).append(dr)
          else:
            desired_results[len(desired_results)] = [dr]
      desired_results = desired_results.values()
      desired_results = [grp for grp in desired_results if len(grp) > 0]
      if len(desired_results) == 0:
        return

      for dr_grp in desired_results:
        arg_grp = []
        for dr in dr_grp:
          cfg_data = dr.configuration.data
          driver = self.tuning_run_main.search_driver
          fidelity = dr.configuration.fidelity
          args = (self.interface, cfg_data, dr.id, fidelity)
          arg_grp.append(args)
        thread_args.append(arg_grp)
      thread_pool = ThreadPool(len(desired_results))
      # print 'Compiling %d results' % len(thread_args)
      try:
        # Use map_async instead of map because of bug where keyboardinterrupts are ignored
        # See http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
        result = thread_pool.map_async(compile_result, thread_args)
        # HG: SQL databases are closed after a certain period of inactivity, so
        # query every 15 minutes to keep the database open.
        while not result.ready():
          self.session.query(literal(1)).scalar()
          result.wait(900)
        compile_results = result.get()
      except KeyboardInterrupt, Exception:
        # Need to kill other processes because only one thread receives
        # exception
        self.interface.kill_all()
        raise
      desired_results = [dr for grp in desired_results for dr in grp]
      compile_results = [result for grp in compile_results for result in grp]

      for compile_result in compile_results:
        if hasattr(compile_result, 'host'):
          try:
            host = compile_result.host
            name = host['name']
            self.session.flush()
            machine = self.session.query(Machine).filter_by(name=name).one()
          except sqlalchemy.orm.exc.NoResultFound:
            machine = Machine(**host)
            self.session.add(machine)
          compile_result.machine = machine

      # print 'Running %d results' % len(thread_args)
      for dr, compile_result in zip(desired_results, compile_results):
        # Make sure compile was successful
        self.run_desired_result(dr, compile_result, dr.id)
        try:
          self.interface.cleanup(dr.id)
        except RuntimeError, e:
          print e
          # print 'Done!'
      thread_pool.close()
    else:
      for dr in q.all():
        if self.claim_desired_result(dr):
          self.run_desired_result(dr)

  def create_slots(self):
    self.slots = []
    if self.args.async_compile:
      for slot in range(self.args.parallelism):
        self.slots.append({'fidelity': None,
                           'max_threads': None,
                           'state': 'IDLE',
                           'desired_result': None,
                           'result': None})
    else:
      for entry in self.interface.tuner_cfg['core_maps'][self.args.core_map]:
        fidelity = entry['fidelity']
        max_threads = entry['max_threads']
        for slot in range(entry['count']):
          self.slots.append({'fidelity': fidelity,
                             'max_threads': max_threads,
                             'state': 'IDLE',
                             'desired_result': None,
                             'result': None})
      self.args.parallelism = len(self.slots)
    self.thread_pool = ThreadPool(self.args.parallelism)

  def get_free_slots(self):
    return [slot for slot in self.slots if slot['state'] == 'IDLE']

  def get_completed_slots(self):
    slots = []
    for slot in self.slots:
      if slot['state'] == 'RUNNING' and slot['result'].ready():
        slots.append(slot)
    return slots

  def wait_for_completion(self):
    try:
      duration = 0
      while len(self.get_completed_slots()) == 0:
        duration += 1
        if duration == 900:
          self.session.query(literal(1)).scalar()
          duration = 0
        time.sleep(1.0)
    except KeyboardInterrupt, Exception:
      # Need to kill other processes because only one thread receives
      # exception
      log.info('Killing all processes...')
      self.interface.kill_all()
      log.info('All processes were killed...')
      raise

  def process_results(self):
    for slot in self.get_completed_slots():
      dr = slot['desired_result']
      compile_result = slot['result'].get()[0]

      slot['state'] = 'IDLE'
      slot['desired_result'] = None
      slot['result'] = None

      if hasattr(compile_result, 'host'):
        try:
          host = compile_result.host
          name = host['name']
          self.session.flush()
          machine = self.session.query(Machine).filter_by(name=name).one()
        except sqlalchemy.orm.exc.NoResultFound:
          machine = Machine(**host)
          self.session.add(machine)
        compile_result.machine = machine
      
      self.run_desired_result(dr, compile_result, dr.id)
      try:
        self.interface.cleanup(dr.id)
      except RuntimeError, e:
        print e

  def start_compile(self, slot):
    def compile_result(args):
      interface, data, result_id, fidelity, max_threads = args
      kwargs = {}
      if fidelity != 0:
        kwargs['fidelity'] = fidelity
      if max_threads is not None:
        kwargs['max_threads'] = max_threads
      return interface.compile(data, result_id, **kwargs)

    slot['state'] = 'RUNNING'
    dr = self.query_pending_desired_results()[0]
    if self.claim_desired_result(dr):
      fidelity = dr.configuration.fidelity
      thread_args = (self.interface, dr.configuration.data, dr.id, fidelity,
                     slot['max_threads'])
      # Use map_async instead of map because of bug where keyboardinterrupts
      # are ignored.  See
      # http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
      result = self.thread_pool.map_async(compile_result, [thread_args])
      slot['desired_result'] = dr
      slot['result'] = result

def _cputype():
  try:
    return re.search(r"model name\s*:\s*([^\n]*)",
                     open("/proc/cpuinfo").read()).group(1)
  except:
    pass
  try:
    # for OS X
    import subprocess

    # The close_fds argument makes sure that file descriptors that should be
    # closed do not remain open because they are copied to the subprocess.
    subproc = subprocess.Popen(["sysctl", "-n", "machdep.cpu.brand_string"],
                               stdout=subprocess.PIPE, close_fds=True)
    return subproc.communicate()[0].strip()
  except:
    log.warning("failed to get cpu type")
  return "unknown"


def _cpucount():
  try:
    return int(os.sysconf("SC_NPROCESSORS_ONLN"))
  except:
    pass
  try:
    return int(os.sysconf("_SC_NPROCESSORS_ONLN"))
  except:
    pass
  try:
    return int(os.environ["NUMBER_OF_PROCESSORS"])
  except:
    pass
  try:
    return int(os.environ["NUM_PROCESSORS"])
  except:
    log.warning("failed to get the number of processors")
  return 1


def _memorysize():
  try:
    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
  except:
    pass
  try:
    return int(os.sysconf("_SC_PHYS_PAGES") * os.sysconf("_SC_PAGE_SIZE"))
  except:
    pass
  try:
    # for OS X
    import subprocess

    # The close_fds argument makes sure that file descriptors that should be
    # closed do not remain open because they are copied to the subprocess.
    subproc = subprocess.Popen(["sysctl", "-n", "hw.memsize"],
                               stdout=subprocess.PIPE, close_fds=True)
    return int(subproc.communicate()[0].strip())
  except:
    log.warning("failed to get total memory")
  return 1024 ** 3

