#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_cuda.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


from __future__ import print_function, division, absolute_import

import atexit, logging, socket, os, toolz, click, dask, threading
from time import time, sleep
from tornado import gen
from tornado.gen import Future, Return, coroutine
from tornado.process import Subprocess
from tornado.ioloop import IOLoop, TimeoutError


_ncores = os.cpu_count()
GPU_KEY = 'CUDA_GPU'


from distributed import LocalCluster,Nanny, Worker
from distributed.comm import get_address_host_port, CommClosedError
from distributed.worker import TOTAL_MEMORY
from distributed.config import config
from distributed.utils import get_ip_interface, parse_timedelta
from distributed.security import Security
from distributed.comm.addressing import uri_from_host_port
from distributed.cli.utils import (check_python_3, install_signal_handlers)
from distributed.preloading import validate_preload_argv
from distributed.proctitle import (enable_proctitle_on_children, enable_proctitle_on_current)


logger = logging.getLogger('distributed.preloading')


import pynvml


def gpu_rscs(w):

    def pids(handle):
        return [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]

    return {GPU_KEY: len([r for r in map(pids, w.gpu_handles) if len(r) == 0])}


def resources_thread(w, secs):
    while w.status not in (None, "closed", "closing"):
        rscs = {}
        rscs.update(gpu_rscs(w))
        with w.rscs_lock:
            w.rscs = rscs
        sleep(secs)


class ResourcedWorker(Worker):
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]

    @gen.coroutine
    def collect_resources(self):
        if not hasattr(self, 'rscs_lock'):
            self.rscs_lock = threading.Lock()
            self.rscs = {}
            self.gpu_thread = threading._start_new_thread(resources_thread, (self, 0.1))

        with self.rscs_lock:
            rscs = self.rscs
        self.available_resources.update(rscs)
        self.total_resources.update(rscs)
        raise Return(rscs)

    @gen.coroutine
    def heartbeat(self):
        if not self.heartbeat_active:
            self.heartbeat_active = True
            try:
                start = time()
                rscs = yield self.collect_resources()
                response = yield self.scheduler.heartbeat_worker(
                    address=self.contact_address, now=time(), metrics=self.get_metrics(), resources=rscs
                )
                end = time()
                middle = (start + end) / 2
                if response:
                    if response["status"] == "missing":
                        yield self._register_with_scheduler()
                        return
                    self.scheduler_delay = response["time"] - middle
                    self.periodic_callbacks["heartbeat"].callback_time = (
                        response["heartbeat-interval"] * 1000
                    )
            except CommClosedError:
                logger.warning("Heartbeat to scheduler failed")
            finally:
                self.heartbeat_active = False
        else:
            logger.debug("Heartbeat skipped: channel busy")


pem_file_option_type = click.Path(exists=True, resolve_path=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("scheduler", type=str, required=False)
@click.option("--tls-ca-file", type=pem_file_option_type, default=None, help="CA cert(s) file for TLS (in PEM format)",)
@click.option("--tls-cert", type=pem_file_option_type, default=None, help="certificate file for TLS (in PEM format)",)
@click.option("--tls-key", type=pem_file_option_type, default=None, help="private key file for TLS (in PEM format)",)
@click.option("--bokeh-port", type=int, default=0, help="Bokeh port, defaults to random port")
@click.option("--bokeh/--no-bokeh", "bokeh", default=True, show_default=True,  required=False, help="Launch Bokeh Web UI",)
@click.option("--host", type=str,  default=None,
    help="Serving host. Should be an ip address that is visible to the scheduler and other workers. "
    "See --listen-address and --contact-address if you need different listen and contact addresses. See --interface.")
@click.option("--interface", type=str, default=None, help="Network interface like 'eth0' or 'ib0'")
@click.option("--nthreads", type=int, default=os.cpu_count(), help="Number of Nanny|Worker")
@click.option("--name", type=str, default=None, help="A unique name for this worker like 'worker-1'. "
    "If used with --nprocs then the process number will be appended like name-0, name-1, name-2, ...")
@click.option("--memory-limit", default="auto",
    help="Bytes of memory per process that the worker can use. This can be an integer (bytes), "
    "float (fraction of total system memory), string (like 5GB or 5000M), 'auto', or zero for no memory management")
@click.option("--reconnect/--no-reconnect", default=True, help="Reconnect to scheduler if disconnected")
@click.option("--pid-file", type=str, default="", help="File to write the process PID")
@click.option("--local-directory", default="", type=str, help="Directory to place worker files")
@click.option("--resources", type=str, default="",
    help='Resources for task constraints like "GPU=2 MEM=10e9". Resources are applied separately to each worker process'
    "(only relevant when starting multiple worker processes with '--nprocs').")
@click.option("--scheduler-file", type=str, default="", help="Filename to JSON encoded scheduler information. Use with dask-scheduler --scheduler-file")
@click.option("--death-timeout", type=str, default=None, help="Seconds to wait for a scheduler before closing")
@click.option("--bokeh-prefix", type=str, default=None, help="Prefix for the bokeh app")
@click.option("--preload", type=str, multiple=True, is_eager=True, help='Module that should be loaded by each worker process like "foo.bar" or "/path/to/foo.py"')
@click.argument("preload_argv", nargs=-1, type=click.UNPROCESSED, callback=validate_preload_argv)
def main(
    scheduler,
    host,
    nthreads,
    name,
    memory_limit,
    pid_file,
    reconnect,
    resources,
    bokeh,
    bokeh_port,
    local_directory,
    scheduler_file,
    interface,
    death_timeout,
    preload,
    preload_argv,
    bokeh_prefix,
    tls_ca_file,
    tls_cert,
    tls_key,
):
    enable_proctitle_on_current()
    enable_proctitle_on_children()
    sec = Security( tls_ca_file=tls_ca_file, tls_worker_cert=tls_cert, tls_worker_key=tls_key )

    if pid_file:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        def del_pid_file():
            if os.path.exists(pid_file):
                os.remove(pid_file)

        atexit.register(del_pid_file)

    services = {}

    if bokeh:
        try:
            from distributed.bokeh.worker import BokehWorker
        except ImportError:
            pass
        else:
            result = (BokehWorker, {"prefix": bokeh_prefix}) if bokeh_prefix else BokehWorker
            services[("bokeh", bokeh_port)] = result

    rscs = gpu_rscs(ResourcedWorker)
    if resources:
        resources = resources.replace(",", " ").split()
        resources = dict(pair.split("=") for pair in resources)
        resources ={k: (float(v) if v else 0.0) for k,v in resources.items()}
        rscs.update(resources)

    loop = IOLoop.current()

    if not scheduler and not scheduler_file and "scheduler-address" not in config:
        raise ValueError("Need to provide scheduler address like\ndask-worker SCHEDULER_ADDRESS:8786")

    if interface:
        if host:
            raise ValueError("Can not specify both interface and host")
        else:
            host = get_ip_interface(interface)

    addr = uri_from_host_port(host, 0, 0) if host else None
    kwargs = {"worker_port": None, "listen_address": addr}
    if death_timeout is not None:
        death_timeout = parse_timedelta(death_timeout, "s")

    name = name or dask.config.get('client-name') or socket.gethostname()
    nannies = [Nanny(
            scheduler,
            scheduler_file=scheduler_file,
            nthreads=nthreads,
            services=services,
            loop=loop,
            resources=rscs,
            memory_limit=memory_limit,
            reconnect=reconnect,
            local_directory=local_directory,
            death_timeout=death_timeout,
            preload=preload,
            preload_argv=preload_argv,
            security=sec,
            env={},
            worker_class=ResourcedWorker,
            contact_address=None,
            name=name if nthreads == 1 else name + "-" + str(i),
            **kwargs
        )
        for i in range(1)]

    @gen.coroutine
    def run():
        yield [n.start() for n in nannies]
        while all(n.status != "closed" for n in nannies):
            yield gen.sleep(0.1)

    # dask_global.py:global_signal_master()  will receive all signal.

    try:
        loop.run_sync(run)
    except (KeyboardInterrupt, TimeoutError):
        pass
    finally:
        logger.info("End worker")


def go():
    check_python_3()
    main()


if __name__ == "__main__":
    go()
