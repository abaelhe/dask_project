#!/usr/bin/env python3
#coding:utf-8


"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_global.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


# Global Python Path

import sys,os,six,time,threading,socket,signal,atexit,pwd


# within context mode `forkserver` ( dask.distributed using `multiprocessing` for subprocess management. )
# HERE we use an `dask.distributed` internal environment variable to distinct inner most processes.
IN_DASK = int(os.getenv('DASK_PARENT') or 0)


SIG_OS = {'KILL', 'STOP'}
SIG_MAP_NAME_ALL = {getattr(signal, a):a[3:] for a in signal.__dict__ if a.startswith('SIG') and a.find('_') < 0}
SIG_MAP_NAME = {k:v for k,v in SIG_MAP_NAME_ALL.items() if v != 'KILL' and v != 'STOP'}
SIG_FROM_NAME= {v:k for k,v in SIG_MAP_NAME.items()}


#INT: CTRL+C
SIG_TERM_DEFAULT = ('CHLD', 'PIPE', 'INT', 'TERM',  'HUP', 'QUIT', 'USR1', 'USR2', 'ALRM', 'XCPU', 'XFSZ', 'VTALRM', 'PROF')


import logging,errno
from tornado.ioloop import IOLoop,PeriodicCallback
from tornado.process import Subprocess
from tornado.util import errno_from_exception
from functools import partial

GLOBAL_IOLOOP = IOLoop.current()
GLOBAL_SIGNAL_REGISTER = {}
_GLOBAL_CTRLC_TIME = 0


logger = logging.getLogger('distributed.preloading')


@atexit.register
def global_safe_release(*args, **kwargs):
    # Here we ensure all things are protected release.
    logger.info("Global Safe Release Check, tid:%s, pid:%s, ppid:%s, pgrp:%s, args:%s, kwargs:%s",
                threading.get_ident(), os.getpid(), os.getppid(), os.getpgrp(), args, kwargs)
    for pid, subproc in Subprocess._waiting.items():
        logger.info('Releasing tornado.process.Subprocess [%s]%s', pid, subproc.proc.args)
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass

    from distributed import Nanny, Worker, Client, Scheduler

    for cls in (Nanny, Worker, Client, Scheduler):
        while len(cls._instances) > 0:
            obj = cls._instances.pop()
            obj.loop.add_callback(obj.close)
            if cls is Client:
                logger.warning('Releasing %s:%s:%s', type(obj), obj.id, obj._start_arg)
            elif cls is Scheduler:
                logger.warning('Releasing %s:%s:%s', type(obj), obj._start_address, obj.address)
            else:
                logger.warning('Releasing %s:%s:%s', type(obj), obj.name, obj.address)
    sys.exit()


def global_signal_handler(sig, frame):
    global _GLOBAL_CTRLC_TIME
    global GLOBAL_IOLOOP
    global logger

    signame = SIG_MAP_NAME_ALL[sig]

    def log_siginfo(signame, frame):
        global logger
        f_code = frame.f_code if frame and hasattr(frame, 'f_code') else None
        f_line = frame.f_lineno if frame and hasattr(frame, 'f_lineno') else None
        logfmt = 'Global Safe Signal Handler, sig:%s, tid:%s, pid:%s, dask:%s, frame:%s:%s'
        args = (signame, threading.get_ident(), os.getpid(), IN_DASK,  f_code, f_line)
        logger.info(logfmt%args)

    log_signal = partial(log_siginfo, signame, frame)
    if signame in ('PIPE', 'HUP'):
        return log_signal()
    elif signame == 'INT':
        log_signal()
        now = time.time()
        if (now - _GLOBAL_CTRLC_TIME) < 1: # DOUBLE CTRL+C in ONE second
            GLOBAL_IOLOOP.add_callback_from_signal(global_safe_release)
        else:
            _GLOBAL_CTRLC_TIME = now
    elif signame == 'TERM':
        log_signal()
        GLOBAL_IOLOOP.add_callback_from_signal(global_safe_release)
    elif signame == 'CHLD':
        ret_pid = 1
        while ret_pid:
            try:
                ret_pid, status = os.waitpid(-1, os.WNOHANG)
            except OSError as e:
                ret_pid = 0
            if ret_pid == 0 or ret_pid not in Subprocess._waiting:
                return
            subproc = Subprocess._waiting.pop(ret_pid)
            cb = subproc._exit_callback
            logger.info('Global Subprocess Term, pid:%s, ret:%s, dask:%s, callback:%s', ret_pid,  status, IN_DASK, cb)
            subproc.io_loop.add_callback_from_signal(subproc._set_returncode, status)
    else:
        log_signal()
        global_safe_release()


def global_signal_master(signals=SIG_TERM_DEFAULT):
    global GLOBAL_SIGNAL_REGISTER

    # `SIGCHLD` signal will handle by tornado.process.Subprocess, this make it as Supervisor of all it's instances.
    Subprocess.initialize()

    for sig_name in list(signals):
        signum = SIG_FROM_NAME[sig_name]
        GLOBAL_SIGNAL_REGISTER[sig_name] = signal.signal(signum, global_signal_handler)

    logger.info('Global Safe Signal Register, tid:%s, pid:%s, dask:%s', threading.get_ident(), os.getpid(), IN_DASK)


# make sure GLOBAL_IOLOOP actually GLOBAL
GLOBAL_IOLOOP.add_callback(global_signal_master)
