# -*- coding: utf-8 -*-
"""Platforms.

Utilities dealing with platform specifics: signals, daemonization,
users, groups, and so on.
"""
from __future__ import absolute_import, print_function, unicode_literals

import atexit, errno, math, numbers, os, struct, sys, operator, warnings,functools
import platform as _platform, signal as _signal
from collections import namedtuple
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType


def try_import(module, default=None):
    """Try to import and return module.
    Returns None if the module does not exist.
    """
    try:
        return import_module(module)
    except ImportError:
        return default


bytes_t = try_import('bytes')
str_t = text_t =str if six.PY3 else unicode
safe_str =lambda s: s.decode(errors='replace') if isinstance(s, bytes) else s
current_process = try_import('billiard.process.current_process')
_setproctitle = try_import('setproctitle')
resource = try_import('resource')
pwd = try_import('pwd')
grp = try_import('grp')
mputil = try_import('multiprocessing.util')

# exitcodes
EX_OK = getattr(os, 'EX_OK', 0)
EX_FAILURE = 1
EX_UNAVAILABLE = getattr(os, 'EX_UNAVAILABLE', 69)
EX_USAGE = getattr(os, 'EX_USAGE', 64)
EX_CANTCREAT = getattr(os, 'EX_CANTCREAT', 73)

SYSTEM = _platform.system()
IS_macOS = SYSTEM == 'Darwin'
IS_WINDOWS = SYSTEM == 'Windows'

_range = namedtuple('_range', ('start', 'stop'))

SIGNAMES = { sig for sig in dir(_signal) if sig.startswith('SIG') and '_' not in sig }
SIGMAP = {getattr(_signal, name): name for name in SIGNAMES}


def isatty(fh):
    """Return true if the process has a controlling terminal."""
    try:
        return fh.isatty()
    except AttributeError:
        pass


def pyimplementation():
    """Return string identifying the current Python implementation."""
    if hasattr(_platform, 'python_implementation'):
        return _platform.python_implementation()
    elif sys.platform.startswith('java'):
        return 'Jython ' + sys.platform
    elif hasattr(sys, 'pypy_version_info'):
        v = '.'.join(str(p) for p in sys.pypy_version_info[:3])
        if sys.pypy_version_info[3:]:
            v += '-' + ''.join(str(p) for p in sys.pypy_version_info[3:])
        return 'PyPy ' + v
    else:
        return 'CPython'


def maybe_fileno(f):
    """Get object fileno, or :const:`None` if not defined."""
    if isinstance(f, numbers.Integral):
        return f
    try:
        return f.fileno()
    except FILENO_ERRORS:
        pass

def get_fdmax(default=None):
    """Return the maximum number of open file descriptors
    on this system.

    :keyword default: Value returned if there's no file
                      descriptor limit.

    """
    try:
        return os.sysconf('SC_OPEN_MAX')
    except:
        pass
    if resource is None:  # Windows
        return default
    fdmax = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if fdmax == resource.RLIM_INFINITY:
        return default
    return fdmax


def close_open_fds(keep=None):  # noqa
    keep = [maybe_fileno(f) for f in (keep or []) if maybe_fileno(f) is not None]
    for fd in reversed(range(get_fdmax(default=2048))):
        if fd not in keep:
            try:
                os.close(fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise

def fd_by_path(paths):
    """Return a list of file descriptors.

    This method returns list of file descriptors corresponding to
    file paths passed in paths variable.

    Arguments:
        paths: List[str]: List of file paths.

    Returns:
        List[int]: List of file descriptors.

    Example:
        >>> keep = fd_by_path(['/dev/urandom', '/my/precious/'])
    """
    stats = set()
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            stats.add(os.fstat(fd)[1:3])
        finally:
            os.close(fd)

    def fd_in_stats(fd):
        try:
            return os.fstat(fd)[1:3] in stats
        except OSError:
            return False

    return [_fd for _fd in range(get_fdmax(2048)) if fd_in_stats(_fd)]


def parse_uid(uid):
    """Parse user id.

    Arguments:
        uid (str, int): Actual uid, or the username of a user.
    Returns:
        int: The actual uid.
    """
    try:
        return int(uid)
    except ValueError:
        try:
            return pwd.getpwnam(uid).pw_uid
        except (AttributeError, KeyError):
            raise KeyError('User does not exist: {0}'.format(uid))


def parse_gid(gid):
    """Parse group id.

    Arguments:
        gid (str, int): Actual gid, or the name of a group.
    Returns:
        int: The actual gid of the group.
    """
    try:
        return int(gid)
    except ValueError:
        try:
            return grp.getgrnam(gid).gr_gid
        except (AttributeError, KeyError):
            raise KeyError('Group does not exist: {0}'.format(gid))


def _setgroups_hack(groups):
    # :fun:`setgroups` may have a platform-dependent limit,
    # and it's not always possible to know in advance what this limit
    # is, so we use this ugly hack stolen from glibc.
    groups = groups[:]

    while 1:
        try:
            return os.setgroups(groups)
        except ValueError:   # error from Python's check.
            if len(groups) <= 1:
                raise
            groups[:] = groups[:-1]
        except OSError as exc:  # error from the OS.
            if exc.errno != errno.EINVAL or len(groups) <= 1:
                raise
            groups[:] = groups[:-1]


def setgroups(groups):
    """Set active groups from a list of group ids."""
    max_groups = None
    try:
        max_groups = os.sysconf('SC_NGROUPS_MAX')
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        return _setgroups_hack(groups[:max_groups])
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise
        if any(group not in groups for group in os.getgroups()):
            # we shouldn't be allowed to change to this group.
            raise


def initgroups(uid, gid):
    """Init process group permissions.

    Compat version of :func:`os.initgroups` that was first
    added to Python 2.7.
    """
    if not pwd:  # pragma: no cover
        return
    username = pwd.getpwuid(uid)[0]
    if hasattr(os, 'initgroups'):  # Python 2.7+
        return os.initgroups(username, gid)
    groups = [gr.gr_gid for gr in grp.getgrall()
              if username in gr.gr_mem]
    setgroups(groups)


def setgid(gid):
    """Version of :func:`os.setgid` supporting group names."""
    os.setgid(parse_gid(gid))


def setuid(uid):
    """Version of :func:`os.setuid` supporting usernames."""
    os.setuid(parse_uid(uid))


def maybe_drop_privileges(uid=None, gid=None):
    """Change process privileges to new user/group.

    If UID and GID is specified, the real user/group is changed.

    If only UID is specified, the real user is changed, and the group is
    changed to the users primary group.

    If only GID is specified, only the group is changed.
    """
    if sys.platform == 'win32':
        return
    if os.geteuid():
        # no point trying to setuid unless we're root.
        if not os.getuid():
            raise SecurityError('contact support')
    uid = uid and parse_uid(uid)
    gid = gid and parse_gid(gid)

    if uid:
        _setuid(uid, gid)
    else:
        gid and setgid(gid)

    if uid and not os.getuid() and not os.geteuid():
        raise SecurityError('Still root uid after drop privileges!')
    if gid and not os.getgid() and not os.getegid():
        raise SecurityError('Still root gid after drop privileges!')


def _setuid(uid, gid):
    # If GID isn't defined, get the primary GID of the user.
    if not gid and pwd:
        gid = pwd.getpwuid(uid).pw_gid
    # Must set the GID before initgroups(), as setgid()
    # is known to zap the group list on some platforms.

    # setgid must happen before setuid (otherwise the setgid operation
    # may fail because of insufficient privileges and possibly stay
    # in a privileged group).
    setgid(gid)
    initgroups(uid, gid)

    # at last:
    setuid(uid)
    # ... and make sure privileges cannot be restored:
    try:
        setuid(0)
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise
        # we should get here: cannot restore privileges,
        # everything was fine.
    else:
        raise SecurityError(
            'non-root user able to restore privileges after setuid.')


class Signals(object):
    """Convenience interface to :mod:`signals`.

    If the requested signal isn't supported on the current platform,
    the operation will be ignored.

    Example:
        >>> from celery.platforms import signals

        >>> from proj.handlers import my_handler
        >>> signals['INT'] = my_handler

        >>> signals['INT']
        my_handler

        >>> signals.supported('INT')
        True

        >>> signals.signum('INT')
        2

        >>> signals.ignore('USR1')
        >>> signals['USR1'] == signals.ignored
        True

        >>> signals.reset('USR1')
        >>> signals['USR1'] == signals.default
        True

        >>> from proj.handlers import exit_handler, hup_handler
        >>> signals.update(INT=exit_handler,
        ...                TERM=exit_handler,
        ...                HUP=hup_handler)
    """

    ignored = _signal.SIG_IGN
    default = _signal.SIG_DFL

    if hasattr(_signal, 'setitimer'):

        def arm_alarm(self, seconds):
            _signal.setitimer(_signal.ITIMER_REAL, seconds)
    else:  # pragma: no cover
        try:
            from itimer import alarm as _itimer_alarm  # noqa
        except ImportError:

            def arm_alarm(self, seconds):  # noqa
                _signal.alarm(math.ceil(seconds))
        else:  # pragma: no cover

            def arm_alarm(self, seconds):      # noqa
                return _itimer_alarm(seconds)  # noqa

    def reset_alarm(self):
        return _signal.alarm(0)

    def supported(self, name):
        """Return true value if signal by ``name`` exists on this platform."""
        try:
            self.signum(name)
        except AttributeError:
            return False
        else:
            return True

    def signum(self, name):
        """Get signal number by name."""
        if isinstance(name, numbers.Integral):
            return name
        if not isinstance(name, string_t) \
                or not name.isupper():
            raise TypeError('signal name must be uppercase string.')
        if not name.startswith('SIG'):
            name = 'SIG' + name
        return getattr(_signal, name)

    def reset(self, *signal_names):
        """Reset signals to the default signal handler.

        Does nothing if the platform has no support for signals,
        or the specified signal in particular.
        """
        self.update((sig, self.default) for sig in signal_names)

    def ignore(self, *names):
        """Ignore signal using :const:`SIG_IGN`.

        Does nothing if the platform has no support for signals,
        or the specified signal in particular.
        """
        self.update((sig, self.ignored) for sig in names)

    def __getitem__(self, name):
        return _signal.getsignal(self.signum(name))

    def __setitem__(self, name, handler):
        """Install signal handler.

        Does nothing if the current platform has no support for signals,
        or the specified signal in particular.
        """
        try:
            _signal.signal(self.signum(name), handler)
        except (AttributeError, ValueError):
            pass

    def update(self, _d_=None, **sigmap):
        """Set signal handlers from a mapping."""
        for name, handler in items(dict(_d_ or {}, **sigmap)):
            self[name] = handler


signals = Signals()
get_signal = signals.signum                   # compat
install_signal_handler = signals.__setitem__  # compat
reset_signal = signals.reset                  # compat
ignore_signal = signals.ignore                # compat


def signal_name(signum):
    """Return name of signal from signal number."""
    return SIGMAP[signum][3:]


def strargv(argv):
    arg_start = 2 if 'manage' in argv[0] else 1
    if len(argv) > arg_start:
        return ' '.join(argv[arg_start:])
    return ''


def set_process_title(progname, info=None):
    """Set the :command:`ps` name for the currently running process.

    Only works if :pypi:`setproctitle` is installed.
    """
    proctitle = '[{0}]'.format(progname)
    proctitle = '{0} {1}'.format(proctitle, info) if info else proctitle
    if _setproctitle:
        _setproctitle.setproctitle(safe_str(proctitle))
    return proctitle


if os.environ.get('NOSETPS'):  # pragma: no cover

    def set_mp_process_title(*a, **k):
        """Disabled feature."""
else:

    def set_mp_process_title(progname, info=None, hostname=None):  # noqa
        """Set the :command:`ps` name from the current process name.

        Only works if :pypi:`setproctitle` is installed.
        """
        if hostname:
            progname = '{0}: {1}'.format(progname, hostname)
        name = current_process().name if current_process else 'MainProcess'
        return set_process_title('{0}:{1}'.format(progname, name), info=info)


def get_errno_name(n):
    """Get errno for string (e.g., ``ENOENT``)."""
    if isinstance(n, string_t):
        return getattr(errno, n)
    return n


@contextmanager
def ignore_errno(*errnos, **kwargs):
    """Context manager to ignore specific POSIX error codes.

    Takes a list of error codes to ignore: this can be either
    the name of the code, or the code integer itself::

        >>> with ignore_errno('ENOENT'):
        ...     with open('foo', 'r') as fh:
        ...         return fh.read()

        >>> with ignore_errno(errno.ENOENT, errno.EPERM):
        ...    pass

    Arguments:
        types (Tuple[Exception]): A tuple of exceptions to ignore
            (when the errno matches).  Defaults to :exc:`Exception`.
    """
    types = kwargs.get('types') or (Exception,)
    errnos = [get_errno_name(errno) for errno in errnos]
    try:
        yield
    except types as exc:
        if not hasattr(exc, 'errno'):
            raise
        if exc.errno not in errnos:
            raise


def check_privileges(accept_content):
    uid = os.getuid() if hasattr(os, 'getuid') else 65535
    gid = os.getgid() if hasattr(os, 'getgid') else 65535
    euid = os.geteuid() if hasattr(os, 'geteuid') else 65535
    egid = os.getegid() if hasattr(os, 'getegid') else 65535

    if hasattr(os, 'fchown'):
        if not all(hasattr(os, attr)
                   for attr in ['getuid', 'getgid', 'geteuid', 'getegid']):
            raise SecurityError('suspicious platform, contact support')

    if not uid or not gid or not euid or not egid:
        if ('pickle' in accept_content or
                'application/x-python-serialize' in accept_content):
            if not C_FORCE_ROOT:
                try:
                    print(ROOT_DISALLOWED.format(
                        uid=uid, euid=euid, gid=gid, egid=egid,
                    ), file=sys.stderr)
                finally:
                    sys.stderr.flush()
                    os._exit(1)
        warnings.warn(RuntimeWarning(ROOT_DISCOURAGED.format(
            uid=uid, euid=euid, gid=gid, egid=egid,
        )))


if sys.version_info < (2, 7, 7):  # pragma: no cover
    import functools

    def _to_bytes_arg(fun):
        @functools.wraps(fun)
        def _inner(s, *args, **kwargs):
            return fun(s.encode(), *args, **kwargs)
        return _inner

    pack = _to_bytes_arg(struct.pack)
    unpack = _to_bytes_arg(struct.unpack)
    unpack_from = _to_bytes_arg(struct.unpack_from)
else:
    pack = struct.pack
    unpack = struct.unpack
    unpack_from = struct.unpack_from
