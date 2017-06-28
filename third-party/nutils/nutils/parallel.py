# -*- coding: utf8 -*-
#
# Module PARALLEL
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The parallel module provides tools aimed at parallel computing. At this point
all parallel solutions use the ``fork`` system call and are supported on limited
platforms, notably excluding Windows. On unsupported platforms parallel features
will disable and a warning is printed.
"""

from . import core, log, numpy, numeric
import os, sys, multiprocessing, tempfile, mmap, traceback, signal

procid = None # current process id, None for unforked

def shzeros( shape, dtype=float ):
  '''create zero-initialized array in shared memory'''

  if numeric.isint( shape ):
    shape = shape,
  else:
    assert all( numeric.isint(sh) for sh in shape )
  dtype = numpy.dtype( dtype )
  size = ( numpy.product( shape ) if shape else 1 ) * dtype.itemsize
  if size == 0:
    return numpy.zeros( shape, dtype )
  fd, name = tempfile.mkstemp( dir=core.getprop( 'shmdir', default=None ) )
  try:
    os.unlink(name)
    # Make sure the entire file is allocated by writing zeros.  If we omit
    # this, writing to the mmap array will cause the process to be killed with
    # SIGBUS if there is no memory available.
    with open( fd, 'wb', closefd=False ) as f:
      bs = 1024 * 1024
      if size >= bs:
        zeros = b'\0' * bs
        for i in range( size // bs ):
          f.write( zeros )
        del zeros
      f.write( b'\0' * (size % bs) )
      assert f.tell() == size
    buf = mmap.mmap( fd, size )
  finally:
    os.close(fd)
  array = numpy.frombuffer( buf, dtype ).reshape( shape )
  assert array.ravel()[0] == 0, '{!r} is not interpreted as 0 ({})'.format(b'\x00'*dtype.itemsize, dtype)
  return array

def pariter( iterable, nprocs ):
  '''iterate in parallel

  Fork into ``nprocs`` subprocesses, then yield items from iterable such that
  all processes receive a nonoverlapping subset of the total. It is up to the
  user to prepare shared memory and/or locks for inter-process communication.
  The following creates a data vector containing the first four quadratics.

  >>> data = shzeros( shape=[4], dtype=int )
  >>> for i in pariter( range(4) ):
  >>>   data[i] = i**2
  >>> data
  [ 0, 1, 4, 9 ]

  As a safety measure nested pariters are blocked by setting the global
  ``procid`` variable; all secundary pariters will be treated like normal
  serial iterators.
  
  Parameters
  ----------
  iterable : iterable
      The collection of items to be distributed over processors
  nprocs : int
      Maximum number of processers to use

  Yields
  ------
      Items from iterable, distributed over at most nprocs processors.
  '''

  global procid

  if procid is not None:
    log.warning( 'ignoring pariter for already forked process' )
    yield from iterable
    return

  try:
    nitems = len(iterable)
  except:
    pass
  else:
    nprocs = min( nitems, nprocs )

  if nprocs <= 1:
    yield from iterable
    return

  shared_iter = multiprocessing.RawValue( 'i', nprocs ) # shared integer pointing at first unyielded item
  lock = multiprocessing.Lock() # lock to avoid race conditions in incrementing shared_iter
  children = [] # list of forked processes, non-empty only in primary process

  try:

    for procid in range( 1, nprocs ):
      child_pid = os.fork()
      if not child_pid:
        signal.signal( signal.SIGINT, signal.SIG_IGN ) # disable sigint (ctrl+c) handler
        break
      children.append( child_pid )
    else:
      procid = 0

    iiter = procid # first index is 0 .. nprocs-1, with shared_iter at nprocs
    for n, it in enumerate( iterable ):
      if n < iiter: # fast forward to iiter
        continue
      assert n == iiter
      yield it
      with lock:
        iiter = shared_iter.value # claim next value
        shared_iter.value = iiter + 1

  except:

    fail = 1
    if procid == 0:
      raise # reraise in main process

    # in child processes print traceback then exit
    excval = sys.exc_info()[1]
    if isinstance( excval, GeneratorExit ):
      log.error( 'generator failed with unknown exception' )
    elif not isinstance( excval, KeyboardInterrupt ):
      log.error( traceback.format_exc() )

  else:

    fail = 0

  finally:

    if procid != 0: # before anything else can fail:
      os._exit( fail ) # cumminicate exit status to main process

    procid = None # unset global variable
    totalfail = fail
    while children:
      child_pid, child_status = os.wait()
      children.remove( child_pid )
      if child_status:
        totalfail += 1
    if fail: # failure in main process: exception has been reraised
      log.error( 'pariter failed in {} out of {} processes; reraising exception for main process'.format( totalfail, nprocs ) )
    elif totalfail: # failure in child process: raise exception
      raise Exception( 'pariter failed in {} out of {} processes'.format( totalfail, nprocs ) )

def parmap( func, iterable, nprocs, shape=(), dtype=float ):
  '''parallel equivalent to builtin map function

  Produces an array of ``func(item)`` values for all items in ``iterable``.
  Because of shared memory restrictions ``func`` must yield numpy arrays of
  predetermined shape and type.

  Parameters
  ----------
  func : python function
      Takes item from iterable, returns numpy array of ``shape`` and ``dtype``
  iterable : iterable
      Collection of items
  nprocs : int
      Maximum number of processers to use
  shape : tuple
      Return shape of ``func``, defaults to scalar
  dtype : tuple
      Return dtype of ``func``, defaults to float

  Returns
  -------
      Array of shape ``len(iterable),+shape`` and dtype ``dtype``
  '''


  n = len(iterable)
  out = shzeros( (n,)+shape, dtype=dtype )
  for i, item in pariter( enumerate(iterable), nprocs=min(n,nprocs) ):
    out[i] = func( item )
  return out

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=1
