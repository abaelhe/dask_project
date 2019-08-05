#!/usr/bin/env python3
#coding:utf-8


"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_global.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


# Global Python Path

import sys,os,six,time,threading,socket,signal,atexit


SIG_OS = {'KILL', 'STOP'}
SIG_MAP_NAME_ALL = {getattr(signal, a):a[3:] for a in signal.__dict__ if a.startswith('SIG') and a.find('_') < 0}
SIG_MAP_NAME = {k:v for k,v in SIG_MAP_NAME_ALL.items() if v != 'KILL' and v != 'STOP'}
SIG_FROM_NAME= {v:k for k,v in SIG_MAP_NAME.items()}


#INT: CTRL+C
SIG_TERM_DEFAULT = ('INT', 'TERM',  'HUP', 'QUIT', 'USR1', 'USR2')
SIG_TERM_FULL = ('INT', 'TERM',  'HUP', 'QUIT', 'USR1', 'USR2', 'TRAP', 'ABRT', 'EMT', 'SYS', 'PIPE', 'ALRM', 'XCPU', 'XFSZ', 'VTALRM', 'PROF')


import logging
from tornado.ioloop import IOLoop,PeriodicCallback
from tornado.process import Subprocess
from functools import partial


GLOBAL_SIGNAL_REGISTER = {}
_GLOBAL_CTRLC_TIME = 0
_GLOBAL_SAFE_RELEASE = 0

logger = logging.getLogger(__name__)


@atexit.register
def global_safe_release(*args, **kwargs):
    global _GLOBAL_SAFE_RELEASE

    if _GLOBAL_SAFE_RELEASE > 0:
        return
    else:
        _GLOBAL_SAFE_RELEASE = 1

    # Here we ensure all things are protected release.
    print("Global Safe Release Check, args:%s, kwargs:%s", args, kwargs)
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
            else:
                logger.warning('Releasing %s:%s:%s', type(obj), obj.name, obj.address)

def global_signal_handler(sig, frame, io_loop=None):
    global _GLOBAL_CTRLC_TIME
    signame = SIG_MAP_NAME_ALL[sig]
    io_loop = io_loop or IOLoop.current()

    if signame == 'INT':
        now = time.time()
        if (now - _GLOBAL_CTRLC_TIME) < 1: # DOUBLE CTRL+C in ONE second
            io_loop.add_callback_from_signal(global_safe_release)# protected IOLoop stop, will make a `global_safe_release` call
        else:
            _GLOBAL_CTRLC_TIME = now

    elif signame == 'TERM':
        print("Stop for SIGTERM.")
        io_loop.add_callback_from_signal(global_safe_release) # protected IOLoop stop, will make a call to `global_safe_release`

    elif signame == 'CHLD':
        logger.warning('All Child process MUST forked by tornado.process.Subprocess !')

    else:
        print("Received sig:%s[%d], frame:%s:%s" % (signame,sig, frame.f_code, frame.f_lineno))
        global_safe_release()


def global_signal_master(signals=SIG_TERM_FULL):
    global GLOBAL_SIGNAL_REGISTER

    if len(GLOBAL_SIGNAL_REGISTER) > 0:
        return

    sig_msg = 'Global Signal Register: %s[%s]'
    for sig_name in signals:
        signum = SIG_FROM_NAME[sig_name]
        GLOBAL_SIGNAL_REGISTER[sig_name] = signal.signal(signum, global_signal_handler)
        logger.info(sig_msg % (sig_name, signum))

    logger.info(sig_msg % (sig_name, signum))
    Subprocess.initialize()
    # `SIGCHLD` signal will received by tornado.process.Subprocess, this make it as Supervisor of all it's instances.

