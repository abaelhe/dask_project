#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_cuda.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


from __future__ import print_function, division, absolute_import

import atexit, logging, socket, os, toolz, click, dask
from tornado import gen
from tornado.process import Subprocess
from tornado.ioloop import IOLoop, TimeoutError


_ncores = os.cpu_count()


from distributed import LocalCluster,Nanny, Worker
from distributed.comm import get_address_host_port
from distributed.worker import TOTAL_MEMORY
from distributed.config import config
from distributed.utils import get_ip_interface, parse_timedelta
from distributed.security import Security
from distributed.comm.addressing import uri_from_host_port
from distributed.cli.utils import (check_python_3, install_signal_handlers)
from distributed.preloading import validate_preload_argv
from distributed.proctitle import (enable_proctitle_on_children, enable_proctitle_on_current)


logger = logging.getLogger(__file__)


def get_n_gpus():
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        return _n_gpus_from_nvidia_smi()


nvidia_ngpus = None
def _n_gpus_from_nvidia_smi():
    global nvidia_ngpus
    if nvidia_ngpus is None:
        nvidia_ngpus = len(os.popen("nvidia-smi -L").read().strip().split("\n"))
    return nvidia_ngpus


def cuda_visible_devices(i, visible=None):
    """ Cycling values for CUDA_VISIBLE_DEVICES environment variable
    Examples
    --------
    >>> cuda_visible_devices(0, range(4))
    '0,1,2,3'
    >>> cuda_visible_devices(3, range(8))
    '3,4,5,6,7,0,1,2'
    """
    if visible is None:
        try:
            visible = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        except KeyError:
            visible = list(range(get_n_gpus()))

    L = visible[i:] + visible[:i]
    return ",".join(map(str, L))


class LocalCUDACluster(LocalCluster):
    def __init__(self, n_workers=None, threads_per_worker=1, memory_limit=None, **kwargs):
        if n_workers > get_n_gpus():
            raise ValueError("Can not specify more processes than GPUs")
        n_workers = n_workers or get_n_gpus()
        memory_limit = memory_limit or (TOTAL_MEMORY / n_workers)
        LocalCluster.__init__(self, n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit, **kwargs)

    @gen.coroutine
    def _start(self, ip=None, n_workers=0):
        """ Start all cluster services. """
        if self.status == "running":
            return
        if (ip is None) and (not self.scheduler_port) and (not self.processes):
            scheduler_address = "inproc://"
        elif ip is not None and ip.startswith("tls://"):
            scheduler_address = "%s:%d" % (ip, self.scheduler_port)
        else:
            if ip is None:
                ip = "127.0.0.1"
            scheduler_address = (ip, self.scheduler_port)
        self.scheduler.start(scheduler_address)
        yield [self._start_worker(**self.worker_kwargs, env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)}) for i in range(n_workers)]
        self.status = "running"
        raise gen.Return(self)


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

    if resources:
        resources = resources.replace(",", " ").split()
        resources = dict(pair.split("=") for pair in resources)
        resources ={k:(float(v) if v else 0.0) for k,v in resources.items()}
    else:
        resources = None

    loop = IOLoop.current()

    kwargs = {"worker_port": None, "listen_address": None}

    if not scheduler and not scheduler_file and "scheduler-address" not in config:
        raise ValueError("Need to provide scheduler address like\ndask-worker SCHEDULER_ADDRESS:8786")

    if interface:
        if host:
            raise ValueError("Can not specify both interface and host")
        else:
            host = get_ip_interface(interface)

    addr = uri_from_host_port(host, 0, 0) if host else None
    if death_timeout is not None:
        death_timeout = parse_timedelta(death_timeout, "s")

    name = name or dask.config.get('client-name') or socket.gethostname()
    nannies = [Nanny(
            scheduler,
            scheduler_file=scheduler_file,
            nthreads=nthreads,
            services=services,
            loop=loop,
            resources=resources,
            memory_limit=memory_limit,
            reconnect=reconnect,
            local_dir=local_directory,
            death_timeout=death_timeout,
            preload=preload,
            preload_argv=preload_argv,
            security=sec,
            contact_address=None,
            env={"CUDA_VISIBLE_DEVICES": cuda_visible_devices(i)},
            name=name if nthreads == 1 else name + "-" + str(i),
            **kwargs
        )
        for i in range(1)]

    @gen.coroutine
    def run():
        yield [n._start(addr) for n in nannies]
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
