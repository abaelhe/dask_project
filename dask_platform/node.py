# -*- coding: utf-8 -*-
"""Worker command-line program.

This module is the 'program-version' of :mod:`celery.worker`.

It does everything necessary to run that module
as an actual application, like installing signal handlers,
platform tweaks, and so on.
"""
from __future__ import absolute_import, print_function, unicode_literals

import logging, os, sys
import platform as _platform
from datetime import datetime
from functools import partial
from contextlib import contextmanager


signals = {}
_process_aware = False
_in_sighandler = False
is_pypy = hasattr(sys, 'pypy_version_info')


logger = get_logger(__name__)


def set_in_sighandler(value):
    """Set flag signifiying that we're inside a signal handler."""
    global _in_sighandler
    _in_sighandler = value


@contextmanager
def in_sighandler():
    """Context that records that we are in a signal handler."""
    set_in_sighandler(True)
    try:
        yield
    finally:
        set_in_sighandler(False)




def active_thread_count():
    from threading import enumerate
    return sum(1 for t in enumerate()
               if not t.name.startswith('Dummy-'))


def safe_say(msg):
    print('\n{0}'.format(msg), file=sys.__stderr__)


def install_platform_tweaks(self, worker):
    """Install platform specific tweaks and workarounds."""
    if self.app.IS_macOS:
        os.environ.setdefault('celery_dummy_proxy', 'set_by_celeryd')

    # Install signal handler so SIGHUP restarts the worker.
    if not self._isatty:
        # only install HUP handler if detached from terminal,
        # so closing the terminal window doesn't restart the worker
        # into the background.
        if self.app.IS_macOS:
            # macOS can't exec from a process using threads.
            # See https://github.com/celery/celery/issues#issue/152
            install_HUP_not_supported_handler(worker)
        else:
            install_worker_restart_handler(worker)
    install_worker_term_handler(worker)
    install_worker_term_hard_handler(worker)
    install_worker_int_handler(worker)
    install_cry_handler()
    install_rdb_handler()


def shutdown_handler(sig, frame):
    pass

def _shutdown_handler(worker, sig='TERM', how='Warm', exc=WorkerShutdown, callback=None, exitcode=EX_OK):
    def _handle_request(*args):
        with in_sighandler():
            if current_process()._name == 'MainProcess':
                if callback:
                    callback(worker)
                safe_say('worker: {0} shutdown (MainProcess)'.format(how))
                signals.worker_shutting_down.send( sender=worker.hostname, sig=sig, how=how, exitcode=exitcode,)
            if active_thread_count() > 1:
                setattr(state, {'Warm': 'should_stop', 'Cold': 'should_terminate'}[how], exitcode)
            else:
                raise exc(exitcode)
    platforms.signals[sig] = _handle_request


install_worker_term_handler = partial(_shutdown_handler, sig='SIGTERM', how='Warm', exc=WorkerShutdown,)
install_worker_term_hard_handler = partial(_shutdown_handler, sig='SIGQUIT', how='Cold', exc=WorkerTerminate, exitcode=EX_FAILURE,)


def on_SIGINT(worker):
    safe_say('Dask: Hitting Ctrl+C again will terminate all running tasks!')
    install_worker_term_hard_handler(worker, sig='SIGINT')


install_worker_int_handler = partial( _shutdown_handler, sig='SIGINT', callback=on_SIGINT, exitcode=EX_FAILURE,)


def _reload_current_worker():
    platforms.close_open_fds([
        sys.__stdin__, sys.__stdout__, sys.__stderr__,
    ])
    os.execv(sys.executable, [sys.executable] + sys.argv)


def install_worker_restart_handler(worker, sig='SIGHUP'):

    def restart_worker_sig_handler(*args):
        """Signal handler restarting the current python program."""
        set_in_sighandler(True)
        safe_say('Restarting celery worker ({0})'.format(' '.join(sys.argv)))
        import atexit
        atexit.register(_reload_current_worker)
        from celery.worker import state
        state.should_stop = EX_OK
    platforms.signals[sig] = restart_worker_sig_handler


def install_cry_handler(sig='SIGUSR1'):
    # Jython/PyPy does not have sys._current_frames
    if is_jython or is_pypy:  # pragma: no cover
        return

    def cry_handler(*args):
        """Signal handler logging the stack-trace of all active threads."""
        with in_sighandler():
            safe_say(cry())
    platforms.signals[sig] = cry_handler


def install_rdb_handler(envvar='CELERY_RDBSIG',
                        sig='SIGUSR2'):  # pragma: no cover

    def rdb_handler(*args):
        """Signal handler setting a rdb breakpoint at the current frame."""
        with in_sighandler():
            from celery.contrib.rdb import set_trace, _frame
            # gevent does not pass standard signal handler args
            frame = args[1] if args else _frame().f_back
            set_trace(frame)
    if os.environ.get(envvar):
        platforms.signals[sig] = rdb_handler


def install_HUP_not_supported_handler(worker, sig='SIGHUP'):

    def warn_on_HUP_handler(signum, frame):
        with in_sighandler():
            safe_say('{sig} not supported: Restarting with {sig} is unstable on this platform!'.format(sig=sig))
    platforms.signals[sig] = warn_on_HUP_handler
