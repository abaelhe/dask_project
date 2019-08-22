#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_master.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""

from __future__ import print_function, division, absolute_import

import atexit
import dask
import logging
import gc
import os
import re
import shutil
import sys
import tempfile
import warnings

import click, pickle, collections

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.process import Subprocess
from distributed import Scheduler
from distributed.scheduler import log_errors, get_address_host, ignoring,CommClosedError, KilledWorker, parse_timedelta
from distributed.security import Security
from distributed.cli.utils import check_python_3, install_signal_handlers
from distributed.preloading import preload_modules, validate_preload_argv
from distributed.proctitle import (
    enable_proctitle_on_children,
    enable_proctitle_on_current,
)

logger = logging.getLogger('distributed.preloading')


class AScheduler(Scheduler):
    def remove_worker(self, comm=None, address=None, safe=False, close=True):
        logger.debug("Removing worker %s ", address)

        if self.status == "closed" or not address:
            return

        address = self.coerce_address(address)
        if address not in self.workers:
            return "already-removed"

        ws = self.workers[address]

        with log_errors():
            try:

                self.log_event(
                    ["all", address],
                    {
                        "action": "remove-worker",
                        "worker": address,
                        "processing-tasks": dict(ws.processing),
                    },
                )
                logger.info("Remove worker %s", address)

                recommendations = collections.OrderedDict()
                for ts in list(ws.processing):
                    k = ts.key
                    recommendations[k] = "released"
                    if not safe:
                        ts.suspicious += 1
                        if ts.suspicious > self.allowed_failures:
                            del recommendations[k]
                            e = pickle.dumps(
                                KilledWorker(task=k, last_worker=ws.clean()), -1
                            )
                            r = self.transition(k, "erred", exception=e, cause=k)
                            recommendations.update(r)

                for ts in ws.has_what:
                    ts.who_has.remove(ws)
                    if not ts.who_has:
                        if ts.run_spec:
                            recommendations[ts.key] = "released"
                        else:  # pure data
                            recommendations[ts.key] = "forgotten"
                ws.has_what.clear()

                self.transitions(recommendations)

                for plugin in self.plugins[:]:
                    try:
                        plugin.remove_worker(scheduler=self, worker=address)
                    except Exception as e:
                        logger.exception(e)

                if close:
                    with ignoring(AttributeError, KeyError, CommClosedError):
                        self.stream_comms[address].send({"op": "close", "report": False})

            finally:
                if address in self.workers:
                    self.remove_resources(address)

                host = get_address_host(address)
                if host in self.host_info:
                    self.host_info[host]["nthreads"] -= ws.nthreads
                    self.host_info[host]["addresses"].remove(address)
                    if not self.host_info[host]["addresses"]:
                        del self.host_info[host]

                self.total_nthreads -= ws.nthreads
                self.rpc.remove(address)
                if address in self.stream_comms:
                    del self.stream_comms[address]
                if ws.name in self.aliases:
                    del self.aliases[ws.name]
                self.idle.discard(ws)
                self.saturated.discard(ws)
                del self.workers[address]
                ws.status = "closed"
                self.total_occupancy -= ws.occupancy

            if not self.workers:
                logger.info("Lost all workers")

            @gen.coroutine
            def remove_worker_from_events():
                # If the worker isn't registered anymore after the delay, remove from events
                if address not in self.workers and address in self.events:
                    del self.events[address]

            cleanup_delay = parse_timedelta(
                dask.config.get("distributed.scheduler.events-cleanup-delay")
            )
            self.loop.call_later(cleanup_delay, remove_worker_from_events)
            logger.debug("Removed worker %s", address)

        return "OK"




pem_file_option_type = click.Path(exists=True, resolve_path=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--host", type=str, default="", help="URI, IP or hostname of this server")
@click.option("--port", type=int, default=None, help="Serving port")
@click.option("--interface", type=str, default=None,help="Preferred network interface like 'eth0' or 'ib0'")
@click.option("--protocol", type=str, default=None, help="Protocol like tcp, tls, or ucx")
@click.option("--tls-ca-file", type=pem_file_option_type, default=None, help="CA cert(s) file for TLS (in PEM format)")
@click.option("--tls-cert", type=pem_file_option_type, default=None, help="certificate file for TLS (in PEM format)")
@click.option("--tls-key", type=pem_file_option_type, default=None, help="private key file for TLS (in PEM format)")
# XXX default port (or URI) values should be centralized somewhere
@click.option("--bokeh-port", type=int, default=None, help="Deprecated.  See --dashboard-address")
@click.option("--dashboard-address", type=str, default=":8787", help="Address on which to listen for diagnostics dashboard")
@click.option("--dashboard/--no-dashboard", "dashboard", default=True, show_default=True, required=False, help="Launch the Dashboard")
@click.option("--bokeh/--no-bokeh", "bokeh", default=None, required=False, help="Deprecated.  See --dashboard/--no-dashboard.")
@click.option("--show/--no-show", default=False, help="Show web UI")
@click.option("--dashboard-prefix", type=str, default=None, help="Prefix for the dashboard app")
@click.option("--use-xheaders", type=bool, default=False, show_default=True, help="User xheaders in dashboard app for ssl termination in header",)
@click.option("--pid-file", type=str, default="", help="File to write the process PID")
@click.option("--scheduler-file", type=str, default="", help="File to write connection information. This may be a good way to share connection information if your cluster is on a shared network file system.")
@click.option("--local-directory", default="", type=str, help="Directory to place scheduler files")
@click.option("--preload", type=str, multiple=True, is_eager=True, default="", help='Module that should be loaded by the scheduler process like "foo.bar" or "/path/to/foo.py".')
@click.argument("preload_argv", nargs=-1, type=click.UNPROCESSED, callback=validate_preload_argv)
@click.version_option()
def main(
    host,
    port,
    bokeh_port,
    show,
    dashboard,
    bokeh,
    dashboard_prefix,
    use_xheaders,
    pid_file,
    scheduler_file,
    interface,
    protocol,
    local_directory,
    preload,
    preload_argv,
    tls_ca_file,
    tls_cert,
    tls_key,
    dashboard_address,
):
    g0, g1, g2 = gc.get_threshold()  # https://github.com/dask/distributed/issues/1653
    gc.set_threshold(g0 * 3, g1 * 3, g2 * 3)

    enable_proctitle_on_current()
    enable_proctitle_on_children()

    if bokeh_port is not None:
        warnings.warn(
            "The --bokeh-port flag has been renamed to --dashboard-address. "
            "Consider adding ``--dashboard-address :%d`` " % bokeh_port
        )
        dashboard_address = bokeh_port
    if bokeh is not None:
        warnings.warn(
            "The --bokeh/--no-bokeh flag has been renamed to --dashboard/--no-dashboard. "
        )
        dashboard = bokeh

    if port is None and (not host or not re.search(r":\d", host)):
        port = 8786

    sec = Security(
        tls_ca_file=tls_ca_file, tls_scheduler_cert=tls_cert, tls_scheduler_key=tls_key
    )

    if not host and (tls_ca_file or tls_cert or tls_key):
        host = "tls://"

    if pid_file:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        def del_pid_file():
            if os.path.exists(pid_file):
                os.remove(pid_file)

        atexit.register(del_pid_file)

    local_directory_created = False
    if local_directory:
        if not os.path.exists(local_directory):
            os.mkdir(local_directory)
            local_directory_created = True
    else:
        local_directory = tempfile.mkdtemp(prefix="scheduler-")
        local_directory_created = True
    if local_directory not in sys.path:
        sys.path.insert(0, local_directory)

    if sys.platform.startswith("linux"):
        import resource  # module fails importing on Windows

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit = max(soft, hard // 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

    loop = IOLoop.current()
    logger.info("-" * 47)

    scheduler = AScheduler(
        loop=loop,
        scheduler_file=scheduler_file,
        security=sec,
        host=host,
        port=port,
        interface=interface,
        protocol=protocol,
        dashboard_address=dashboard_address if dashboard else None,
        service_kwargs={"dashboard": {"prefix": dashboard_prefix}},
    )

    loop.add_callback(scheduler.start)
    if not preload:
        preload = dask.config.get("distributed.scheduler.preload")
    if not preload_argv:
        preload_argv = dask.config.get("distributed.scheduler.preload-argv")
    preload_modules(
        preload, parameter=scheduler, file_dir=local_directory, argv=preload_argv
    )

    logger.info("Local Directory: %26s", local_directory)
    logger.info("-" * 47)

    # dask_global.py:global_signal_master()  will receive all signal.

    try:
        loop.start()
        loop.close()
    finally:
        scheduler.stop()
        if local_directory_created:
            shutil.rmtree(local_directory)

        logger.info("End scheduler at %r", scheduler.address)


def go():
    check_python_3()
    main()


if __name__ == "__main__":
    go()
