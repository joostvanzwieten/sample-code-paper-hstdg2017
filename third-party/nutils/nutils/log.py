# -*- coding: utf8 -*-
#
# Module LOG
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The log module provides print methods ``debug``, ``info``, ``user``,
``warning``, and ``error``, in increasing order of priority. Output is sent to
stdout as well as to an html formatted log file if so configured.
"""

import sys, time, warnings, functools, itertools, re, abc, contextlib, html, urllib.parse, os, json, traceback, bdb, inspect, textwrap
from . import core

warnings.showwarning = lambda message, category, filename, lineno, *args: \
  warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )

LEVELS = 'error', 'warning', 'user', 'info', 'debug'


## LOG

class Log( metaclass=abc.ABCMeta ):
  '''The :class:`Log` object is what is stored in the ``__log__`` property. It
  should define a ``context`` method that returns a context manager which adds
  a contextual layer and a ``write`` method.'''

  def __enter__( self ):
    return self

  def __exit__( self, etype, value, tb ):
    if etype in (KeyboardInterrupt,SystemExit,bdb.BdbQuit):
      self.write( 'error', 'killed by user' )
    elif etype is not None:
      self.write( 'error', ''.join( traceback.format_exception( etype, value, tb ) ) )

  @abc.abstractmethod
  def context( self, title ):
    '''Return a context manager that adds a contextual layer named ``title``.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def write( self, level, text ):
    '''Write ``text`` with log level ``level`` to the log.

    .. Note:: This function is abstract.
    '''

class ContextLog( Log ):
  '''Base class for loggers that keep track of the current list of contexts.

  The base class implements :meth:`context` which keeps the attribute
  :attr:`_context` up-to-date.

  .. attribute:: _context

     A :class:`list` of contexts (:class:`str`\\s) that are currently active.
  '''

  def __init__( self ):
    self._context = []
    super().__init__()

  def _push_context( self, title ):
    self._context.append( title )

  def _pop_context( self ):
    self._context.pop()

  @contextlib.contextmanager
  def context( self, title ):
    '''Return a context manager that adds a contextual layer named ``title``.

    The list of currently active contexts is stored in :attr:`_context`.'''
    self._push_context( title )
    try:
      yield
    finally:
      self._pop_context()

class ContextTreeLog( ContextLog ):
  '''Base class for loggers that display contexts as a tree.

  .. automethod:: _print_push_context
  .. automethod:: _print_pop_context
  .. automethod:: _print_item
  '''

  def __init__( self ):
    super().__init__()
    self._printed_context = 0

  def _pop_context( self ):
    super()._pop_context()
    if self._printed_context > len( self._context ):
      self._printed_context -= 1
      self._print_pop_context()

  def write( self, level, text ):
    '''Write ``text`` with log level ``level`` to the log.

    This method makes sure the current context is printed and calls
    :meth:`_print_item`.
    '''
    from . import parallel
    if parallel.procid:
      return
    for title in self._context[self._printed_context:]:
      self._print_push_context( title )
      self._printed_context += 1
    if text is not None:
      self._print_item( level, text )

  @abc.abstractmethod
  def _print_push_context( self, title ):
    '''Push a context to the log.

    This method is called just before the first item of this context is added
    to the log.  If no items are added to the log within this context or
    children of this context this method nor :meth:`_print_pop_context` will be
    called.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def _print_pop_context( self ):
    '''Pop a context from the log.

    This method is called whenever a context is exited, but only if
    :meth:`_print_push_context` has been called before for the same context.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def _print_item( self, level, text ):
    '''Add an item to the log.

    .. Note:: This function is abstract.
    '''

class StdoutLog( ContextLog ):
  '''Output plain text to stream.'''

  def __init__( self, stream=sys.stdout ):
    self.stream = stream
    super().__init__()

  def _mkstr( self, level, text ):
    return ' > '.join( self._context + ([ text ] if text is not None else []) )

  def write( self, level, text, endl=True ):
    verbose = core.getprop( 'verbose', len(LEVELS) )
    if level not in LEVELS[verbose:]:
      from . import parallel
      if parallel.procid is not None:
        text = '[{}] {}'.format( parallel.procid, text )
      s = self._mkstr( level, text )
      self.stream.write( s + '\n' if endl else s )

class RichOutputLog( StdoutLog ):
  '''Output rich (colored,unicode) text to stream.'''

  # color order: black, red, green, yellow, blue, purple, cyan, white

  cmap = { 'path': (2,1), 'error': (1,1), 'warning': (1,0), 'user': (3,0) }

  def __init__( self, stream=sys.stdout, *, progressinterval=None ):
    super().__init__( stream=stream )
    # Timestamp at which a new progress line may be written.
    self._progressupdate = 0
    # Progress update interval in seconds.
    self._progressinterval = progressinterval or core.getprop( 'progressinterval', 0.1 )

  def __exit__( self, *exc_info ):
    # Clear the progress line.
    self.stream.write( '\033[K' )
    super().__exit__( *exc_info )

  def _mkstr( self, level, text ):
    if text is not None:
      string = ' · '.join( self._context + [text] )
      n = len(string) - len(text)
      # This is not a progress line.  Reset the update timestamp.
      self._progressupdate = 0
    else:
      string = ' · '.join( self._context )
      n = len(string)
      # Don't touch `self._progressupdate` here.  Will be done in
      # `self._push_context`.
    try:
      colorid, boldid = self.cmap[level]
    except KeyError:
      return '\033[K\033[1;30m{}\033[0m{}'.format( string[:n], string[n:] )
    else:
      return '\033[K\033[1;30m{}\033[{};3{}m{}\033[0m'.format( string[:n], boldid, colorid, string[n:] )

  def _push_context( self, title ):
    super()._push_context( title )
    from . import parallel
    if parallel.procid:
      return
    t = time.time()
    if t >= self._progressupdate:
      self._progressupdate = t + self._progressinterval
      self.stream.write( self._mkstr( 'progress', None ) + '\r' )

class HtmlInsertAnchor( Log ):
  '''Mix-in class for HTML-based loggers that inserts anchor tags for paths.

  .. automethod:: _insert_anchors
  '''

  def _path2href( self, match ):
    if match.group(0) not in core.listoutdir():
      return match.group(0)
    filename = html.unescape( match.group(0) )
    ext = html.unescape( match.group(1) )
    whitelist = ['.jpg','.png','.svg','.txt','.mp4','.webm'] + list( core.getprop( 'plot_extensions', [] ) )
    fmt = '<a href="{href}"' + (' class="plot"' if ext in whitelist else '') + '>{name}</a>'
    return fmt.format( href=urllib.parse.quote( filename ), name=html.escape( filename ) )

  def _insert_anchors( self, level, escaped_text ):
    '''Insert anchors for all paths in ``escaped_text``.

    .. Note:: ``escaped_text`` should be valid html (e.g. the result of ``html.escape(text)``).
    '''
    return re.sub( r'\b\w+([.]\w+)\b', self._path2href, escaped_text )

class HtmlLog( HtmlInsertAnchor, ContextTreeLog ):
  '''Output html nested lists.'''

  def __init__( self, file, *, title='nutils', scriptname=None ):
    if isinstance( file, (str, bytes) ):
      self._file = file = core.open_in_outdir( file, 'w' )
    else:
      self._file = None
    self._print = functools.partial( print, file=file )
    self._flush = file.flush
    self._title = title
    self._scriptname = scriptname
    super().__init__()

  def __enter__( self ):
    super().__enter__()
    # Copy dependencies.
    logpath = os.path.join( os.path.dirname( __file__ ), '_log' )
    for filename in os.listdir( logpath ):
      if not filename.startswith( '.' ):
        with open( os.path.join( logpath, filename ), 'rb' ) as src, core.open_in_outdir( filename, 'wb' ) as dst:
          dst.write( src.read() )
    # Write header.
    if self._file:
      self._file.__enter__()
    self._print( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">' )
    self._print( '<html><head>' )
    self._print( '<title>{}</title>'.format( html.escape( self._title ) ) )
    self._print( '<script type="text/javascript" src="viewer.js" ></script>' )
    self._print( '<link rel="stylesheet" type="text/css" href="style.css">' )
    self._print( '</head><body class="newstyle"><pre>' )
    if self._scriptname is not None:
      self._print( '<span id="navbar">goto: <a class="nav_latest" href="../../../../log.html?{1:.0f}">latest {0:}</a> | <a class="nav_latestall" href="../../../../../log.html?{1:.0f}">latest overall</a> | <a class="nav_index" href="../../../../../">index</a></span>'.format( self._scriptname, time.mktime(time.localtime()) ) )
    self._print( '<ul>' )
    return self

  def __exit__( self, etype, value, tb ):
    super().__exit__( etype, value, tb )
    self._print( '</ul>' )
    if etype not in (None,KeyboardInterrupt,SystemExit,bdb.BdbQuit):
      self.write_post_mortem( etype, value, tb )
    # Write footer.
    self._print( '</pre></body></html>' )
    if self._file:
      self._file.__exit__( etype, value, tb )

  def write( self, level, text ):
    '''Write ``text`` with log level ``level`` to the log.

    This method makes sure the current context is printed and calls
    :meth:`_print_item`.
    '''
    if level not in LEVELS[core.getprop( 'verbose', len(LEVELS) ):]:
      return super().write( level, text )

  def _print_push_context( self, title ):
    self._print( '<li class="context">{}</li><ul>'.format( html.escape( title ) ) )
    self._flush()

  def _print_pop_context( self ):
    self._print( '</ul>' )
    self._flush()

  def _print_item( self, level, text ):
    escaped_text = self._insert_anchors( level, html.escape( text ) )
    self._print( '<li class="{}">{}</li>'.format( html.escape( level ), escaped_text ) )
    self._flush()

  def write_post_mortem( self, etype, value, tb ):
    'write exception nfo to html log'

    _fmt = lambda obj: '=' + ''.join( s.strip() for s in repr(obj).split('\n') )
    self._print( '<span class="post-mortem">' )
    self._print( 'EXHAUSTIVE STACK TRACE' )
    self._print()
    for frame, filename, lineno, function, code_context, index in inspect.getinnerframes( tb ):
      self._print( 'File "{}", line {}, in {}'.format( filename, lineno, function ) )
      self._print( html.escape( textwrap.fill( inspect.formatargvalues(*inspect.getargvalues(frame),formatvalue=_fmt), initial_indent=' ', subsequent_indent='  ', width=80 ) ) )
      if code_context:
        self._print()
        for line in code_context:
          self._print( html.escape( textwrap.fill( line.strip(), initial_indent='>>> ', subsequent_indent='    ', width=80 ) ) )
      self._print()
    self._print( '</span>' )
    self._flush()

class IndentLog( HtmlInsertAnchor, ContextTreeLog ):
  '''Output indented html snippets.'''

  def __init__( self, file, *, progressfile=None, progressinterval=None ):
    self._logfile = file
    self._print = functools.partial( print, file=file )
    self._flush = file.flush
    self._prefix = ''
    self._progressfile = progressfile
    if self._progressfile:
      # Timestamp at which a new progress line may be written.
      self._progressupdate = 0
      # Progress update interval in seconds.
      self._progressinterval = progressinterval or core.getprop( 'progressinterval', 1 )
    super().__init__()

  def _print_push_context( self, title ):
    title = title.replace( '\n', '' ).replace( '\r', '')
    self._print( '{}c {}'.format( self._prefix, html.escape( title ) ) )
    self._flush()
    self._prefix += ' '

  def _print_pop_context( self ):
    self._prefix = self._prefix[:-1]

  def _print_item( self, level, text ):
    text = self._insert_anchors( level, html.escape( text ) )
    level = html.escape( level[0] )
    for line in text.splitlines():
      self._print( '{}{} {}'.format( self._prefix, level, line ) )
      level = '|'
    self._flush()
    if self._progressfile:
      self._print_progress( level, text )
      self._progressupdate = 0

  def _push_context( self, title ):
    super()._push_context( title )
    if not self._progressfile:
      return
    from . import parallel
    if parallel.procid:
      return
    t = time.time()
    if t < self._progressupdate:
      return
    self._print_progress( None, None )
    self._progressupdate = t + self._progressinterval

  def _print_progress( self, level, text ):
    if self._progressfile.seekable():
      self._progressfile.seek( 0 )
      self._progressfile.truncate( 0 )
    json.dump( dict( logpos=self._logfile.tell(), context=self._context, text=text, level=level ), self._progressfile )
    self._progressfile.write( '\n' )
    self._progressfile.flush()

class TeeLog( Log ):
  '''Simultaneously interface multiple logs'''

  def __init__( self, *logs ):
    self.logs = logs

  def __enter__( self ):
    self._stack = contextlib.ExitStack()
    self._stack.__enter__()
    for log in self.logs:
      self._stack.enter_context( log )
    return self

  def __exit__( self, *exc_info ):
    self._stack.__exit__( *exc_info )

  @contextlib.contextmanager
  def context( self, title ):
    with contextlib.ExitStack() as stack:
      for log in self.logs:
        stack.enter_context( log.context( title ) )
      yield

  def write( self, level, text ):
    for log in self.logs:
      log.write( level, text )
    
class CaptureLog( ContextLog ):
  '''Silently capture output to a string buffer while writing single character
  progress info to a secondary stream.'''

  def __init__( self, stream=sys.stdout ):
    self.stream = stream
    self.lines = []
    super().__init__()

  def write( self, level, text ):
    self.lines.append( ' > '.join( self._context + ([ text ] if text is not None else []) ) )
    self.stream.write( level[0] )
    self.stream.flush()

  @property
  def captured( self ):
    return '\n'.join( self.lines )


## INTERNAL FUNCTIONS

# references to objects that are going to be redefined
_range = range
_iter = iter
_zip = zip
_enumerate = enumerate

def _len( iterable ):
  '''Return length if available, otherwise None'''

  try:
    return len(iterable)
  except:
    return None

def _mklog():
  return ( RichOutputLog if core.getprop('richoutput') else StdoutLog )( sys.stdout )

def _getlog():
  log = core.getprop( 'log', None )
  if not isinstance( log, Log ):
    if log is not None:
      warnings.warn( '''Invalid logger object found: {!r}
        This is usually caused by manually setting the __log__ variable.'''.format(log), stacklevel=2 )
    log = _mklog()
  return log

def _print( level, *args ):
  return _getlog().write( level, ' '.join( str(arg) for arg in args ) )


## MODULE-ONLY METHODS

locals().update({ name: functools.partial( _print, name ) for name in LEVELS })

def path( *args ):
  warnings.warn( "log level 'path' will be removed in the future, please use any other log level instead", DeprecationWarning )
  return _print( 'info', *args )

def range( title, *args ):
  '''Progress logger identical to built in range'''

  items = _range( *args )
  log = _getlog()
  for item in items:
    with log.context( '{} {} ({:.0f}%)'.format( title, item, item * 100 / len(items) ) ):
      yield item

def iter( title, iterable, length=None ):
  '''Progress logger identical to built in iter'''

  if length is None:
    length = _len(iterable)
  log = _getlog()
  it = _iter( iterable )
  for index in itertools.count():
    text = '{} {}'.format( title, index )
    if length:
      text += ' ({:.0f}%)'.format( 100 * index / length )
    with log.context( text ):
      try:
        yield next(it)
      except StopIteration:
        break

def enumerate( title, iterable ):
  '''Progress logger identical to built in enumerate'''

  return iter( title, _enumerate(iterable), length=_len(iterable) )

def zip( title, *iterables ):
  '''Progress logger identical to built in enumerate'''

  lengths = [ _len(iterable) for iterable in iterables ]
  return iter( title, _zip(*iterables), length=all(lengths) and min(lengths) )

def count( title, start=0, step=1 ):
  '''Progress logger identical to itertools.count'''

  log = _getlog()
  for item in itertools.count(start,step):
    with log.context( '{} {}'.format( title, item ) ):
      yield item
    
def title( f ): # decorator
  '''Decorator, adds title argument with default value equal to the name of the
  decorated function, unless argument already exists. The title value is used
  in a static log context that is destructed with the function frame.'''

  assert getattr( f, '__self__', None ) is None, 'cannot decorate bound instance method'
  default = f.__name__
  argnames = f.__code__.co_varnames[:f.__code__.co_argcount]
  if 'title' in argnames:
    index = argnames.index( 'title' )
    if index >= len(argnames) - len(f.__defaults__ or []):
      default = f.__defaults__[ index-len(argnames) ]
    gettitle = lambda args, kwargs: args[index] if index < len(args) else kwargs.get('title',default)
  else:
    gettitle = lambda args, kwargs: kwargs.pop('title',default)
  @functools.wraps(f)
  def wrapped( *args, **kwargs ):
    __log__ = _getlog() # repeat as property for fast retrieval
    with __log__.context( gettitle(args,kwargs) ):
      return f( *args, **kwargs )
  return wrapped

def context( title ):
  return _getlog().context( title )


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
