from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, backref
from sqlalchemy import (
  Column, Integer, String, DateTime, Boolean, Enum,
  Float, PickleType, ForeignKey, Text, func, Index,
  UniqueConstraint)
import sqlalchemy
import sqlalchemy.types as types
from sqlalchemy.databases import mysql
import hashlib
import math
import re

import logging
log = logging.getLogger(__name__)

from cPickle import dumps, loads
from gzip import zlib
class CompressedPickler(object):
  @classmethod
  def dumps(cls, obj, protocol=2):
    s = dumps(obj, protocol)
    sz = zlib.compress(s, 9)
    if len(sz) < len(s):
      return sz
    else:
      return s

  @classmethod
  def loads(cls, string):
    try:
      s = zlib.decompress(string)
    except:
      s = string
    return loads(s)

class LargePickleType(PickleType):
  """Pickle that can store up to 16 MB
  """
  impl = mysql.MSMediumBlob

class FloatWithInfinity(types.TypeDecorator):
  """Store infinity as -1 because SQL does not support infinity."""  
  impl = types.Float

  def process_bind_param(self, value, dialect):
    if isinstance(value, float) and math.isinf(value):
      value = 3.4e38
    return value

  def process_result_value(self, value, dialect):
    if isinstance(value, float) and value >= 3.4e38:
      value = float('inf')
    return value

class Base(object):
  @declared_attr
  def __tablename__(cls):
    """convert camel case to underscores"""
    return re.sub(r'([a-z])([A-Z])', r'\1_\2', cls.__name__).lower()

  id = Column(Integer, primary_key=True, index=True)


Base = declarative_base(cls=Base)

class _Meta(Base):
  """ meta table to track current version """
  db_version = Column(String(128), unique=True)

  @classmethod
  def get_version(cls, session):
    try:
      session.flush()
      x = session.query(_Meta).one()
      return x.db_version
    except sqlalchemy.orm.exc.NoResultFound:
      return None

  @classmethod
  def add_version(cls, session, version):
    try:
      session.flush()
      session.add(_Meta(db_version=version))
      session.commit()
    except sqlalchemy.exc.IntegrityError:
      session.rollback()


class Program(Base):
  project = Column(String(128))
  name = Column(String(128))

  @classmethod
  def get(cls, session, project, name):
    try:
      session.flush()
      t = Program(project=project, name=name)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(Program).filter_by(project=project, name=name).one()


UniqueConstraint(Program.project, Program.name, name='uix_program_custom1')


class ProgramVersion(Base):
  program_id = Column(ForeignKey(Program.id, ondelete='CASCADE'))
  program = relationship(Program, backref='versions')
  version = Column(String(128))
  parameter_info = Column(Text)
  hash = Column(String(64))

  @property
  def name(self):
    return self.program.name

  @property
  def project(self):
    return self.program.project

  @classmethod
  def get(cls, session, project, name, version, parameter_info=None):
    program = Program.get(session, project, name)
    try:
      session.flush()
      if parameter_info is not None:
        hash = hashlib.sha256(parameter_info).hexdigest()
      else:
        hash = None
      t = ProgramVersion(program=program, version=version, parameter_info=parameter_info,
                         hash=hash)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(ProgramVersion).filter_by(program=program,
                                                     version=version,
                                                     hash=hash).one()

      
UniqueConstraint(ProgramVersion.program_id, ProgramVersion.version,
                 ProgramVersion.hash, name='uix_program_version_custom1')


class Configuration(Base):
  program_id = Column(ForeignKey(Program.id, ondelete='CASCADE'))
  program = relationship(Program)
  hash = Column(String(64))
  data = Column(LargePickleType(pickler=CompressedPickler))
  # HG: Search algorithms not supporting multiple fidelities set the fidelity
  # to 0.
  fidelity = Column(Integer, default=0)

  @classmethod
  def get(cls, session, program, hashv, datav, fidelity):
    try:
      session.flush()
      t = Configuration(program=program, hash=hashv, data=datav,
                        fidelity=fidelity)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return (session.query(Configuration)
              .filter_by(program=program, hash=hashv, fidelity=fidelity).one())


Index('ix_configuration_custom1', Configuration.program_id, Configuration.hash,
      Configuration.fidelity)
UniqueConstraint(Configuration.program_id, Configuration.hash,
                 Configuration.fidelity, name='uix_configuration_custom1')


class MachineClass(Base):
  name = Column(String(128), unique=True)

  @classmethod
  def get(cls, session, name):
    try:
      session.flush()
      t = MachineClass(name=name)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(MachineClass).filter_by(name=name).one()


class Machine(Base):
  name = Column(String(128))

  cpu = Column(String(128))
  cores = Column(Integer)
  memory_gb = Column(Float)

  machine_class_id = Column(ForeignKey(MachineClass.id, ondelete='CASCADE'))
  machine_class = relationship(MachineClass, backref='machines')


class Platform(Base):
  name = Column(String(128), unique=True)

  luts = Column(Integer)
  regs = Column(Integer)
  dsps = Column(Integer)
  brams = Column(Integer)

  axi_bus_width = Column(Integer)
  proc_freq = Column(Integer)
  
  @classmethod
  def get(cls, session, name):
    try:
      session.flush()
      p = Platform(name=name)
      session.add(p)
      session.commit()
      return p
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(Platform).filter_by(name=name).one()


class InputClass(Base):
  program_id = Column(ForeignKey(Program.id, ondelete='CASCADE'))
  program = relationship(Program, backref='inputs')

  name = Column(String(128))
  size = Column(Integer)

  @classmethod
  def get(cls, session, program, name='default', size=-1):
    try:
      session.flush()
      t = InputClass(program=program, name=name, size=size)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(InputClass).filter_by(program=program,
                                                 name=name,
                                                 size=size).one()

      
UniqueConstraint(InputClass.program_id, InputClass.name, InputClass.size,
                 name='uix_input_class_custom1')


class Input(Base):
  #state          = Column(Enum('ANY_MACHINE', 'SINGLE_MACHINE', 'DELETED'),
  #                        default='ANY_MACHINE', name='t_input_state')

  input_class_id = Column(ForeignKey(InputClass.id, ondelete='CASCADE'))
  input_class = relationship(InputClass, backref='inputs')

  #optional, set only for state='SINGLE_MACHINE'
  #machine_id     = Column(ForeignKey(MachineClass.id))
  #machine        = relationship(MachineClass, backref='inputs')

  #optional, for use by InputManager
  path = Column(Text)
  extra = Column(PickleType(pickler=CompressedPickler))


class Test(Base):
  name = Column(String(128), unique=True)
  description = Column(String(128))
  
  @classmethod
  def get(cls, session, name, description):
    try:
      session.flush()
      t = Test(name=name, description=description)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(Test).filter_by(name=name).one()


class TuningRun(Base):
  uuid = Column(String(32), index=True, unique=True)

  program_version_id = Column(ForeignKey(ProgramVersion.id, ondelete='CASCADE'))
  program_version = relationship(ProgramVersion, backref='tuning_runs')

  machine_id = Column(ForeignKey(Machine.id, ondelete='CASCADE'))
  machine = relationship(Machine, backref='tuning_runs')

  input_class_id = Column(ForeignKey(InputClass.id, ondelete='CASCADE'))
  input_class = relationship(InputClass, backref='tuning_runs')

  test_id = Column(ForeignKey(Test.id, ondelete='CASCADE'))
  test = relationship(Test, backref='tuning_runs')

  name = Column(String(128), default='unnamed')
  args = Column(PickleType(pickler=CompressedPickler))
  objective = Column(PickleType(pickler=CompressedPickler))

  state = Column(Enum('QUEUED', 'RUNNING', 'COMPLETE', 'ABORTED',
                      name='t_tr_state'),
                 default='QUEUED')
  start_date = Column(DateTime, default=func.now())
  end_date = Column(DateTime)

  final_config_id = Column(ForeignKey(Configuration.id, ondelete='CASCADE'))
  final_config = relationship(Configuration)

  seed = Column(Integer)

  platform_id = Column(ForeignKey(Platform.id, ondelete='CASCADE'))
  platform = relationship(Platform, backref='platforms')

  #__mapper_args__ = {'primary_key': uuid}

  @property
  def program(self):
    return self.program_version.program


class Result(Base):
  #set by MeasurementDriver:
  configuration_id = Column(ForeignKey(Configuration.id, ondelete='CASCADE'))
  configuration = relationship(Configuration)
  
  machine_id = Column(ForeignKey(Machine.id, ondelete='CASCADE'))
  machine = relationship(Machine, backref='results')

  input_id = Column(ForeignKey(Input.id, ondelete='CASCADE'))
  input = relationship(Input, backref='results')

  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'), index=True)
  tuning_run = relationship(TuningRun, backref='results')

  collection_date = Column(DateTime, default=func.now())
  collection_cost = Column(Float)

  #set by MeasurementInterface:
  state = Column(String(7))
  msg = Column(String(128))
  run_time = Column(FloatWithInfinity, default = float('inf'))
  accuracy = Column(Float)
  energy = Column(Float)
  size = Column(Float)
  confidence = Column(Float)
  #extra = Column(PickleType)

  build_time = Column(FloatWithInfinity, default = float('inf'))
  copy_time = Column(FloatWithInfinity, default = float('inf'))
  cleanup_time = Column(FloatWithInfinity, default = float('inf'))

  mem_usage = Column(FloatWithInfinity, default = float('inf'))
  
  luts = Column(FloatWithInfinity, default = float('inf'))
  regs = Column(FloatWithInfinity, default = float('inf'))
  brams = Column(FloatWithInfinity, default = float('inf'))
  dsps = Column(FloatWithInfinity, default = float('inf'))

  #set by SearchDriver
  was_new_best = Column(Boolean)


Index('ix_result_custom1', Result.tuning_run_id, Result.was_new_best)


class DesiredResult(Base):
  #set by the technique:
  configuration_id = Column(ForeignKey(Configuration.id, ondelete='CASCADE'))
  configuration = relationship(Configuration)
  limit = Column(FloatWithInfinity)

  #set by the search driver
  priority = Column(Float)
  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'))
  tuning_run = relationship(TuningRun, backref='desired_results')
  generation = Column(Integer)
  requestor = Column(String(128))
  request_date = Column(DateTime, default=func.now())

  #set by the measurement driver
  state = Column(Enum('UNKNOWN', 'REQUESTED', 'RUNNING',
                      'COMPLETE', 'ABORTED',
                      name="t_dr_state"),
                 default='UNKNOWN')
  result_id = Column(ForeignKey(Result.id, ondelete='CASCADE'), index=True)
  result = relationship(Result, backref='desired_results')
  start_date = Column(DateTime)

  #input_id        = Column(ForeignKey(Input.id))
  #input           = relationship(Input, backref='desired_results')

  thread = Column(Integer)
  search_time = Column(Float)


Index('ix_desired_result_custom1', DesiredResult.tuning_run_id,
      DesiredResult.generation)

Index('ix_desired_result_custom2', DesiredResult.tuning_run_id,
      DesiredResult.configuration_id)


# track bandit meta-technique information if a bandit meta-technique is used for a tuning run.
class BanditInfo(Base):
  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'))
  tuning_run = relationship(TuningRun, backref='bandit_info')
  # the bandit exploration/exploitation tradeoff
  c = Column(Float)
  # the bandit window
  window = Column(Integer)

class BanditSubTechnique(Base):
  bandit_info_id = Column(ForeignKey(BanditInfo.id, ondelete='CASCADE'))
  bandit_info = relationship(BanditInfo, backref='subtechniques')
  name = Column(String(128))


class HudsonTuningRun(Base):
  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'))
  tuning_run = relationship(TuningRun, backref='hudson_tuning_run')
  args = Column(PickleType(pickler=CompressedPickler))
  cfg = Column(PickleType(pickler=CompressedPickler))


class BayesError(Base):
  result_id = Column(ForeignKey(Result.id, ondelete='CASCADE'))
  result = relationship(Result, backref='errors')
  error = Column(FloatWithInfinity)

Index('ix_bayes_error_result_id', BayesError.result_id)

class BayesIteration(Base):
  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'))
  tuning_run = relationship(TuningRun, backref='iterations')
  generation = Column(Integer)

Index('ix_bayes_iteration_tuning_run_id', BayesIteration.tuning_run_id)

class BayesCost(Base):
  iteration_id = Column(ForeignKey(BayesIteration.id, ondelete='CASCADE'))
  iteration = relationship(BayesIteration, backref='costs')
  configuration_id = Column(ForeignKey(Configuration.id, ondelete='CASCADE'))
  configuration = relationship(Configuration, backref='costs')
  new_cfg = Column(Boolean)
  cost = Column(Float)
  no_div_cost = Column(Float)
  expec_impr = Column(Float)
  success_prob = Column(Float)
  constr_pen = Column(Float)
  div_pen = Column(Float)
  explore_offs = Column(Float)
  max_grad = Column(Float)
  optimum = Column(Float)

Index('ix_bayes_cost_iteration_id', BayesCost.iteration_id)
Index('ix_bayes_cost_configuration_id', BayesCost.configuration_id)

class BayesMetric(Base):
  name = Column(String(16))
  
  @classmethod
  def get(cls, session, name):
    try:
      session.flush()
      t = BayesMetric(name=name)
      session.add(t)
      session.commit()
      return t
    except sqlalchemy.exc.IntegrityError:
      session.rollback()
      return session.query(BayesMetric).filter_by(name=name).one()

UniqueConstraint(BayesMetric.name, name='uix_bayes_metric_name')

class BayesPrediction(Base):
  cost_id = Column(ForeignKey(BayesCost.id, ondelete='CASCADE'))
  cost = relationship(BayesCost, backref='predictions')
  metric_id = Column(ForeignKey(BayesMetric.id, ondelete='CASCADE'))
  metric = relationship(BayesMetric, backref='predictions')
  fidelity = Column(Integer)
  mean = Column(Float)
  std_dev = Column(Float)

Index('ix_bayes_prediction_metric_id', BayesPrediction.metric_id)
Index('ix_bayes_prediction_cost_id', BayesPrediction.cost_id)

class ParameterInfo(Base):
  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'))
  tuning_run = relationship(TuningRun, backref='parameters')
  name = Column(String(128))
  kind = Column(String(128))
  lower_bound = Column(Float)
  upper_bound = Column(Float)
  choices = Column(PickleType(pickler=CompressedPickler))

Index('ix_parameter_info_tuning_run_id', ParameterInfo.tuning_run_id)

class DebugInfo(Base):
  tuning_run_id = Column(ForeignKey(TuningRun.id, ondelete='CASCADE'))
  tuning_run = relationship(TuningRun, backref='debug_info')
  info = Column(PickleType(pickler=CompressedPickler))

Index('ix_debug_info_tuning_run_id', DebugInfo.tuning_run_id)


if __name__ == '__main__':
  #test:
  engine = create_engine('sqlite:///:memory:', echo=True)
  Base.metadata.create_all(engine)

