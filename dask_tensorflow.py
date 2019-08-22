#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_tensorflow.py
@time: 7/26/19 12:05 PM
@license: All Rights Reserved, Abael.com
"""

import imp
import six, os, sys, time, collections, socket, json, datetime, traceback, signal, zipimport, threading, tornado
from functools import partial,wraps,lru_cache
from concurrent.futures import ThreadPoolExecutor


from tornado import gen, locks
from tornado import process
from tornado.gen import coroutine, Return, Future
from tornado.locks import Event, Condition, Lock, Semaphore, BoundedSemaphore

from distributed import (Client, Worker, get_worker, secede as worker_secede, get_client, Reschedule, Pub, Sub, Lock,
                         Variable, worker_client)
from distributed.protocol import to_serialize
from distributed.threadpoolexecutor import secede as thread_pool_secede
from distributed.worker import thread_state
import logging


logger = logging.getLogger('distributed.preloading')


from dask_usage import USAGE_INFO
from dask_signal import IN_DASK, GLOBAL_IOLOOP
from dask_global import (cuda_free_indexes, global_cluster, urlparse, safe_cmd, model_cleanup, async_send,
    DASK_PYTHONHASHSEED, DASK_PYTHON_INTERPRETER,
    DASK_READ_CHUNK_SIZE, DASK_WORKSPACE, DASK_DATA_POOL_DIR, DASK_MODEL_POOL_DIR)


TF_PORT = 11111
TENSORFLOW_KEYS = ['chief', 'master', 'service', 'session', 'ps', 'worker']

TF_CONFIG_ENV = 'TF_CONFIG'
TF_TASK_ENV_KEY = 'task'
TF_TASK_TYPE_KEY = 'type'
TF_TASK_ID_KEY = 'index'

TF_CLUSTER_KEY = 'cluster'
TF_SERVICE_KEY = 'service'
TF_SESSION_MASTER_KEY = 'session_master'
TF_EVAL_SESSION_MASTER_KEY = 'eval_session_master'
TF_MODEL_DIR_KEY = 'model_dir'
TF_GRPC_SCHEME = 'grpc://'

# Tensor TF_CONFIG environment:
# Master | Chief: require 'cluster' key in TF_CONFIG;
# Worker: 'cluster' key MUST NOT in TF_CONFIG
TF_CONFIG = {
    TF_TASK_ENV_KEY: {TF_TASK_TYPE_KEY: 'chief', TF_TASK_ID_KEY: 0},
    TF_CLUSTER_KEY: {
        'evaluator': ['192.168.1.1:2224'], # ONLY ONE `evaluator` !
        'chief': ['192.168.1.1:2222'],  'master': ['192.168.1.1:2223'], # This TWO is for Super ONE
        'ps': ['192.168.1.104:2222', '192.168.1.105:2222'],
        'worker': ['192.168.1.100:2222', '192.168.1.101:2222',
                   '192.168.1.102:2222', '192.168.1.103:2222'],
    },
    TF_SERVICE_KEY: {},
    TF_SESSION_MASTER_KEY: {},
    TF_EVAL_SESSION_MASTER_KEY: {},
}


class TFTaskType(object):
    MASTER = 'master'
    PS = 'ps'
    WORKER = 'worker'
    CHIEF = 'chief'
    EVALUATOR = 'evaluator' # 'For distributed training, there can only be one `evaluator` task,  means: `task_id=0`'
    dic = {'chief':0, 'master':1, 'ps':2, 'worker':4, 'evaluator':8}
    def __init__(self, task_type): # TaskType And Device Filters association rules
        self.task_type = task_type
        if self._task_type == TFTaskType.MASTER:
            device_filters = ['/job:ps', '/job:master']
        elif self._task_type == TFTaskType.CHIEF:
            device_filters = ['/job:ps', '/job:chief']
        elif self._task_type == TFTaskType.WORKER:
            device_filters = ['/job:ps', '/job:worker/task:%d' % self._task_id]
        elif self._task_type == TFTaskType.PS:
            device_filters = ['/job:ps', '/job:worker', '/job:chief', '/job:master']
        else:
            # If the task_type is `EVALUATOR` or something other than the ones in
            # TaskType then don't set any device filters.
            device_filters = None
        self.device_filters  = device_filters

# ref.: distributed:worker.py:run(server, comm, function, args=(), kwargs={}, is_coro=None, wait=True):
# support builtin special kwargs:  dask_worker:server, dask_scheduler:server, ONLY available: distributed.client._run()

#################################
def gpu_filter(_devices, gpu_flag):
    return [x for x in _devices if x['name'].find('GPU') > 0] if gpu_flag is True else _devices


def tensorflow_devices():
    from tensorflow.python.client import device_lib
    device_list = device_lib.list_local_devices()
    for device in device_list:
        locality = device.locality
        yield (device.name, {
            'name': device.name,
            'memory_limit': device.memory_limit,
            'device_type': device.device_type,
            'byte_size': device.ByteSize(),
            'physical_device_desc': device.physical_device_desc,
            'incarnation': device.incarnation,
            'locality': {
               'bus_id': locality.bus_id,
               'numa_node': locality.numa_node,
               'links': str(locality.links)
            },
         })


def tensorflow_options(gpu_mem_fraction=0.95):
    import tensorflow as tf
    Config = tf.compat.v1.ConfigProto
    print('TENSORFLOW JIT STATUS: %s' % tf.config.optimizer.get_jit())
    tf.config.optimizer.set_jit(1)
    tf_option = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True,
                                            gpu_options=tf.compat.v1.GPUOptions(
                                                per_process_gpu_memory_fraction=gpu_mem_fraction,
                                                force_gpu_compatible=True,
                                                allow_growth=True,
                                            ),
                                          )
    tf_option.graph_options.optimizer_options.opt_level = tf.compat.v1.OptimizerOptions.L1
    tf_option.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    tf_option.graph_options.optimizer_options.do_function_inlining = True
    return tf_option
    #s=tf_option.SerializeToString();tf.compat.v1.ConfigProto.FromString(s)


class TFActor(object):
    def __del__(self):

        self.recovery()

    def __str__(self):
        return "<%s %s>" %(self.__class__.__name__, self.key)

    def __init__(self, key, *args, tf_config=None, tf_option=None, scheduler_info=None, **kwargs):
        # here we made this thread an OWNER of this task by secede from it's ThreadPoolExecutor.
        # NOTE: `thread_pool_secede` ONLY works in NON-coroutine actor_exectutor, ref:worker.actor_execute()
        self.dask_worker = get_worker()
        self.thrid = threading.get_ident()

        thread_pool_secede(adjust=True)
        self.dask_worker.loop.add_callback(self.dask_worker.transition, thread_state.key, "long-running")

        self.key = key
        self.name = self.dask_worker.name
        self.hostname = socket.gethostname()
        self.address = self.dask_worker.address
        self.scheduler_info = scheduler_info

        model_name = self.key.partition(':')[0]
        self.model_name = model_name[:-4] if model_name.endswith('.zip') else model_name
        self.tf_option = json.loads(tf_option) if isinstance(tf_option, str) else tf_option
        self.tf_config = json.loads(tf_config) if isinstance(tf_config, str) else tf_config

        self.dask_cwd = os.path.abspath(os.getcwd())
        self.tf_model_pool_dir = os.path.abspath(DASK_MODEL_POOL_DIR)
        self.tf_data_pool_dir = os.path.abspath(DASK_DATA_POOL_DIR)
        self.tf_data_dir = os.path.join(self.tf_data_pool_dir, self.model_name)
        self.tf_config_dir = os.path.join(self.tf_data_dir, 'config')
        self.tf_save_dir = os.path.join(self.tf_data_dir, 'ckpt')
        self.tf_log_dir = os.path.join(self.tf_data_dir, 'log')
        os.system('mkdir -p %r; rm -rf %r; mkdir -p %r %r %r %r' % (self.tf_save_dir, self.tf_save_dir,
                  self.tf_data_dir, self.tf_config_dir, self.tf_save_dir, self.tf_log_dir))
        os.chdir(self.tf_data_dir)

        self.sys_stdio = (sys.__stdin__, sys.__stdout__, sys.__stderr__, sys.stdin, sys.stdout, sys.stderr)
        self.stdout = open(os.path.join(self.tf_log_dir, '%s.log' % self.key.partition(':')[-1]), 'a+', encoding=sys.stdout.encoding)
        sys.__stdout__ = sys.__stderr__ = sys.stdout = sys.stderr = self.stdout
        self.stdin = sys.stdin
        self.sys_path, self.sys_argv = sys.path[:], sys.argv[:]

        logger.info('Accepted Tensorflow Key:%s, Job:%s, Options:%s, Scheduler:%s', key, tf_config, tf_option, scheduler_info)
        self.devices = dict(tensorflow_devices())
        self.future_chunk_size = DASK_READ_CHUNK_SIZE
        self.args = args
        self.kwargs = kwargs

        self.sub = Sub(self.key, worker=self.dask_worker)
        self.result = Pub(model_name, worker=self.dask_worker)
        self.exe = self.preflight()
        self.dask_worker.loop.add_callback(self.flight, self.exe)

    def device_info(self, xla=None, gpu=True):
        if xla is None:
            return gpu_filter([v for (x, v) in self.devices.items() if v['name'].find('GPU') >= 0], gpu_flag=gpu)
        elif xla is True:
            return gpu_filter([v for (x, v) in self.devices.items() if v['name'].find('XLA') >= 0], gpu_flag=gpu)
        else:
            return gpu_filter([v for (x, v) in self.devices.items() if v['name'].find('XLA') < 0], gpu_flag=gpu)

    def tensorflow_env(self, tf_option, tf_config, dask_context, cuda_indexes=None):
        model_entrypoint = os.path.join(self.tf_model_pool_dir, self.model_name)
        zip_ep, pkg_ep =model_entrypoint + '.zip', os.path.join(model_entrypoint, '__main__.py')
        if os.path.exists(pkg_ep) and os.path.isfile(pkg_ep):
            model_entrypoint = pkg_ep
        elif os.path.exists(zip_ep) and os.path.isfile(zip_ep):
            model_entrypoint = zip_ep
        else:
            raise Exception(USAGE_INFO)

        env_dict = {}

        for key in ('LANG', 'PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH',
            'USER', 'HOME', 'HOSTNAME', 'SHELL', 'TERM', 'SHLVL', 'MAIL', 'SSH_CONNECTION', 'SSH_TTY', 'SSH_CLIENT'):
            val = os.getenv(key)
            if val is not None:
                env_dict[key] = val

        env_dict.update(
            XLA_FLAGS='--xla_hlo_profile',
            TF_DASK_PID=str(os.getpid()),
            RF_DASK_PGRP=str(os.getpgrp()),
            TF_XLA_FLAGS=("--tf_xla_cpu_global_jit " + os.environ.get("TF_XLA_FLAGS", "")),
            TF_MODEL=self.model_name,
            TF_CONTEXT=json.dumps(dask_context),
            TF_CONFIG=json.dumps(tf_config),
            TF_MODEL_POOL_DIR=self.tf_model_pool_dir,
            TF_DATA_POOL_DIR=self.tf_data_pool_dir,
            TF_MODEL_ENTRYPOINT=model_entrypoint,
            TF_CONFIG_DIR=self.tf_config_dir,
            TF_DATA_DIR=self.tf_data_dir,
            TF_LOG_DIR=self.tf_log_dir,
            TF_SAVE_DIR=self.tf_save_dir,
            PYTHONPATH=':'.join([self.tf_model_pool_dir, self.tf_data_dir, self.dask_cwd]),
            PYTHONHASHSEED=str(int(DASK_PYTHONHASHSEED)),
            PYTHONIOENCODING=sys.getdefaultencoding(),
            PYTHONUNBUFFERED='True',
        )

        if cuda_indexes:  # we explicitly assign GPU indexes to use; let tensorflow aware of ONLY these indexes
            env_dict['CUDA_VISIBLE_DEVICES'] = cuda_indexes

        return env_dict

    def log(self, msg, *args, flush=True):
        self.stdout.write((msg % args) if args else msg)
        if flush:
            self.stdout.flush()

    def run_model(self, stdin, stdout, *args, **kwargs):
        import sys
        sys.stdin = stdin
        self.stdout = sys.stdout = sys.stderr = sys.__stdout__ = sys.__stderr__ = stdout

        self.log('HERE IN ASYNC SUBPROCESS: %s' % os.getpid())

        model_name = os.getenv('TF_MODEL')
        model_entry =os.getenv('TF_MODEL_ENTRYPOINT')

        if model_entry.endswith('.zip'):
            model_root, modname = model_entry, '__main__'
        elif model_entry.endswith('.py'):
            model_root, modname = os.path.dirname(model_entry), os.path.basename(model_entry).rsplit('.',1)[0]

        self.log('HERE IN ASYNC MODEL START, %s, %s' % (modname, model_root))
        sys.path.insert(0, model_root)
        __import__(modname)

    def preflight(self):
        # this NODE is selected for this task
        node_name, node_port, cuda_indexes, dask_url = self.tf_config.pop('dask').split(':', 3)
        job_name, task_index = self.tf_config['task']['type'], self.tf_config['task']['index']
        tensorflow_addr = self.tf_config['cluster'][job_name][task_index]

        using_xla_gpu_devices = self.device_info(xla=True, gpu=True)
        using_xla_gpu_device_names = sorted([x['name'] for x in using_xla_gpu_devices])

        if isinstance(self.tf_option, (str, bytes)):
            import tensorflow as tf
            tf_option = tf.compat.v1.ConfigProto.FromString(self.tf_option)
        elif self.tf_option is not None:
            tf_option = self.tf_option
        else:
            tf_option = tensorflow_options()

        dask_context = {
            'model_task': '%s, %s' % (self.key, ','.join(using_xla_gpu_device_names)),
            'model_addr': tensorflow_addr,
            'worker_addr': self.address,
            'schduler_addr': self.scheduler_info,
            'workspace': DASK_WORKSPACE,
            'local_dir': self.dask_cwd,
            'pid': os.getpid(),
            'thread_id': self.thrid,
            'code': 0,
        }

        env_dict = self.tensorflow_env(tf_option, self.tf_config, dask_context, cuda_indexes=cuda_indexes)
        cmd = [sys.executable, r'-u', env_dict['TF_MODEL_ENTRYPOINT'], self.key]
        fmt = 'Model Start, key:%s,\n  cmd:%s\n  dask_context:%s\n  sys.path:%s\n  tf_option:%s\n  tf_config:%s\n\n'
        self.log(fmt % (self.key, cmd, dask_context, self.sys_path, tf_option, self.tf_config))

        for k, v in env_dict.items():
            if not isinstance(k, str) or not isinstance(v, str):
                self.log('Error env k:%s, v:%s\n' % (k, v))

        exe_package = partial(process.Subprocess, cmd, executable=DASK_PYTHON_INTERPRETER,
            cwd=env_dict['TF_DATA_DIR'], env=env_dict, preexec_fn=None,
            stdin=self.stdin, stdout=self.stdout, stderr=self.stdout, encoding=sys.getdefaultencoding(),
            pass_fds=(self.stdin.fileno(), self.stdout.fileno()), universal_newlines=False, bufsize=0,
            restore_signals=False, start_new_session=False)

        return exe_package

    def flight(self, exe_package):
        # flighting in main thread, since `SIGCHLD` MUST received in it; and then correctly call exit callback.
        self.exe = exe_package()
        self.exe.set_exit_callback(self.landed)
        msg = '\n ***** Tensorflow Task   Inited, key:%s, sub:%s, pid:%s ***** ' %(self.key, self.exe.pid, os.getpid())
        self.log(msg)

    def landed(self, retval=0):
        self.log("worker pub msg: %s", {self.key: retval})
        self.result.put({self.key: retval})
        ident = yield self.dask_worker.scheduler.identity()
        msg = yield self.sub._get(timeout=10)
        self.log('Tensorflow Push Message Received, sub:%s, msg:%s, ident:%s' % (self.key, msg, ident))
        msg = '\n ***** Tensorflow Task Finished, key:%s, ret:%s, tid:%s, pid:%s, ***** \n\n' % (
            self.key, retval, threading.get_ident(), os.getpid())
        self.log(msg)
        self.recovery()

    def recovery(self):
        if self.sys_stdio is None:
            return

        self.log('State Recovery:%s', self.key)
        os.chdir(self.dask_cwd)
        sys.__stdin__, sys.__stdout__, sys.__stderr__, sys.stdin, sys.stdout, sys.stderr = self.sys_stdio
        sys.path, sys.argv = self.sys_path, self.sys_argv

        if self.stdin:
            if self.stdin != sys.__stdin__:
                self.stdin.close()
            else:
                self.stdin.flush()

        if self.stdout:
            if self.stdout != sys.__stdout__:
                self.stdout.close()
            else:
                self.stdout.flush()

        self.stdin = self.stdout = self.sys_stdio = self.sys_path = self.sys_argv = self.dask_worker = None
        del self.result
        del self.exe
        del self.sub


def tensorflow_gen_config(free_node_name_map=None, parallel=1, save='~/tf_configs.json', **tf_cluster_spec):
    if free_node_name_map:
        r = [(url, name, indexes) for url, (name, indexes) in free_node_name_map.items() if len(indexes) > 0]
    else:
        client = global_cluster(asynchronous=False)
        r =[(url, name, indexes) for url, (name, indexes) in client.run(cuda_free_indexes).items() if len(indexes) > 0]

    free_node_urls = sorted(r, key=lambda x: -len(x[-1]))
    free_gpu_devices = collections.deque([(url, name, x) for (url, name, indexes) in free_node_urls for x in indexes])
    total_gpu_cores = len(free_gpu_devices)
    if parallel > 1:
        total_gpu_cores = max(1, int(total_gpu_cores // parallel))
    #  tf_cluster_spec['worker'] = max(1, (len(free_gpu_devices)-2) // 10)# 预计同时会有10个同事跑GPU任务; 公平调度策略, 预留10个;

    if total_gpu_cores < 1:
        raise Exception('All Machines is busy, Total Available:%s' % len(free_node_urls))
    if len(tf_cluster_spec) < 1:
        if total_gpu_cores == 1:
            tf_cluster_spec ={'chief': 1}
        elif total_gpu_cores == 2:
            tf_cluster_spec ={'chief': 1, 'worker':1}
        elif total_gpu_cores == 3:
            tf_cluster_spec ={'chief': 1, 'worker':2}
        else:
            tf_cluster_spec ={'chief': 1, 'ps': 1, 'worker':  max(1, len(free_gpu_devices)-2)}

    print('tf_cluster_spec: %s' % tf_cluster_spec)
    tf_cluster = collections.defaultdict(list)
    port_allocation = collections.defaultdict(lambda : (TF_PORT - 1))
    chief_allocation = tf_cluster_spec.pop('chief', 1)
    ps_allocation = tf_cluster_spec.pop('ps', 0)
    tf_configs = []
    if chief_allocation:
        job_name = 'chief'
        job_task_index = 0
        chief_node_url, chief_node_name, cuda_index = free_gpu_devices.pop()
        chief_node_host, _ = urlparse(chief_node_url).netloc.rsplit(':', 1)
        port_allocation[chief_node_host] += 1
        chief_node_port = port_allocation[chief_node_host]
        tf_cluster[job_name].append('%s:%s' % (chief_node_host, chief_node_port))
        tf_configs.append({
            'cluster': tf_cluster,
            'task': {'type': job_name, 'index': job_task_index},
            'dask':'%s:%s:%s:%s' %(chief_node_name, chief_node_port, cuda_index, chief_node_url)
        })

    if ps_allocation:
        job_name = 'ps'
        job_task_index = 0
        ps_node_url, ps_node_name, cuda_index = free_gpu_devices.pop()
        ps_node_host, _ = urlparse(ps_node_url).netloc.rsplit(':', 1)
        port_allocation[ps_node_host] += 1
        ps_node_port = port_allocation[ps_node_host]
        tf_cluster[job_name].append('%s:%s' % (ps_node_host, ps_node_port))
        tf_configs.append({
            'cluster': tf_cluster,
            'task': {'type': job_name, 'index': job_task_index},
            'dask': '%s:%s:%s:%s' % (ps_node_name, ps_node_port, cuda_index, ps_node_url)
        })

    for job_name, machine_total in tf_cluster_spec.items():
        for job_task_index in range(machine_total):
            dask_url, node_name, cuda_index = free_gpu_devices.popleft()
            dask_host, _ = urlparse(dask_url).netloc.rsplit(':', 1)
            port_allocation[dask_host] += 1
            node_port = port_allocation[dask_host]
            tf_cluster[job_name].append('%s:%s' % (dask_host, node_port))
            tf_configs.append({
                'cluster': tf_cluster,
                'task': {'type': job_name, 'index': job_task_index},
                'dask': '%s:%s:%s:%s' % (node_name, node_port, cuda_index, dask_url)
            })

    if save:
        with open(os.path.expanduser(save), 'wb') as jsf:
            jsf.write(json.dumps(tf_configs).encode())
            jsf.flush()

    return tf_configs


def dask_sork_key(task_key):
    m, _, n = task_key.partition('@')
    (model_name, job_name, job_index), (node_name, cuda_index) = m.split(':'), n.split(':')
    return model_name, TFTaskType.dic[job_name], int(job_index), node_name, cuda_index


@coroutine
def startup_actors(scheduler_info, client, model_name, tf_option, tf_configs, future):
    rsc = {'CUDA_GPU': 1}
    def gen_func(tf_config):
        job_name, job_index = tf_config['task']['type'], tf_config['task']['index']
        node_name, task_port, cuda_index, node_url = tf_config['dask'].split(':', 3)
        task_key = '%s:%s:%s@%s:%s' % (model_name, job_name, job_index, node_name, cuda_index)

        actor_startup = partial(client.submit, TFActor, task_key,
                tf_config =tf_config, tf_option =tf_option, scheduler_info=scheduler_info,
                key=task_key, workers=[node_url], resources=rsc,
                fifo_timeout="100 ms",
                retries=10,
                priority=100,
                allow_other_workers=False,
                actor=True)
        return task_key, actor_startup

    startups =[gen_func(tf_config) for tf_config in tf_configs]
    logger.info('Submitting: %s, Resources:%s',[x[0] for x in startups], rsc)
    actors = yield {k:s() for (k, s) in startups}
    future.set_result(actors)


@coroutine # eg.: job_counts={'ps':10, 'workers':100}, ParameterServers:10, CUDAworkers:100
def tensorflow_scheduler(global_future, model_name, client=None, tf_option=None, tf_port=None, **tf_cluster_spec):
    scheduler_info =yield client.scheduler.identity()
    cuda_free_map =yield client.run(cuda_free_indexes)
    tf_configs = tensorflow_gen_config(free_node_name_map=cuda_free_map, **tf_cluster_spec)

    logger.info('Model Schedule %s: \n  tf_configs:%s\n\n', model_name, tf_configs)

    tf_option = tf_option if isinstance(tf_option, (str, bytes)) else (tf_option.SerializeToString() if tf_option else tf_option)

    chief_configs, ps_configs, other_configs = [], [], []
    for tf_config in tf_configs:
        task_type = tf_config['task']['type']
        task_index = tf_config['task']['index']
        if task_type in ('chief', 'master'):
            chief_configs.append(tf_config)
        elif task_type in ('ps',):
            ps_configs.append(tf_config)
        else:
            other_configs.append(tf_config)

    s_time = time.time()
    dt = datetime.datetime.now()

    chief_configs.sort(key=lambda cfg: cfg['task']['index'])
    ps_configs.sort(key=lambda cfg: cfg['task']['index'])
    other_configs.sort(key=lambda cfg: cfg['task']['index'])

    client.loop.set_default_executor(ThreadPoolExecutor(max_workers=len(tf_configs)))

    result_future = Future()
    result_future.tf_configs = tf_configs
    result_future.tf_option = tf_option
    result_future.cuda_map = cuda_free_map

    chief_future = Future()
    client.loop.add_callback(startup_actors, scheduler_info, client, model_name, tf_option, tf_configs, chief_future)
    chief_actors = yield chief_future

    sorted_task_keys = list(sorted(chief_actors.keys(), key=lambda x:dask_sork_key(x)))

    sub = Sub(model_name, client=client)
    pubs ={k: Pub(model_name, client=client) for k in sorted_task_keys}
    scheduler_info =yield client.scheduler.identity() # data flush sync between this client and scheduler

    def chief_finish(task_key, actor, fu):
        value = fu.result()
        logger.info('Tensorflow Finished[%s/%s], key:%s, val:%s', len(chief_actors), len(tf_configs), task_key, value)
        chief_actors[task_key] = actor
        if len(chief_actors) == len(tf_configs):
            logger.info('Tensorflow Cluster All Finished: %s', chief_actors.keys())

    # Chief First.
    msgs = {}
    chief_key_actor = sorted_task_keys[0]

    while (len(msgs) + 1) < len(chief_actors):
        msg = yield sub._get()
        logger.info('Sub Rcv %s:%s', type(msg), msg)
        msgs.update(msg)

    import pdb;pdb.set_trace()
    #    A = yield chief_actor.get_result()
    assert chief_key_actor in msgs, 'Tensorflow Chief Task Required: %s' % chief_key_actor
    time.sleep(1)
    future = yield model_cleanup(client, model_name)
    import pdb;pdb.set_trace()
    logger.info("Tensorflow Task clean, %s", chief_actors)
    global_future.set_result(chief_actors)


def start_tensorflow(model_name, client=None, options=None, port=TF_PORT, **kwargs):
    """ Start Tensorflow on Dask Cluster

    This launches Tensorflow Servers alongside Dask workers in-process

    Examples
    --------
    >>> client = Client('dask-scheduler-address:8786')
    >>> tf_spec, dask_spec = start_tensorflow(client)
    >>> tf_spec.as_dict()
    {'worker': ['192.168.1.100:2222', '192.168.1.101:2222']}

    Specify desired number of jobs types as keyword args
    >>> tf_config = tf.compat.v1.OptimizerOptions()
    >>> tf_config.GlobalJitLevel = tf_config.OFF
    >>> tf_config.do_function_inlining = True
    >> tf_config.opt_level
    >>> tf_config.gpu_options.force_gpu_compatible = True


    >>> tf_spec, dask_spec = start_tensorflow(client, tf_config=tf_config, chief=1, master=1, ps=2, worker=30)
    >>> tf_spec.as_dict()
    {
     'chief': ['192.168.1.1:2222'],
     'master': ['192.168.1.1:2223'],
        'ps': ['192.168.1.104:2222', '192.168.1.105:2222'],
    'worker': ['192.168.1.100:2222', '192.168.1.101:2222',
                '192.168.1.102:2222', '192.168.1.103:2222']
    }
    """
    client = client if client is not None else global_cluster(asynchronous=True)
    global_future = Future()
    tensorflow_scheduler_wrap = partial(tensorflow_scheduler, global_future, model_name, client=client, tf_port=port, **kwargs)
    if client.asynchronous:
        global_future.add_done_callback(lambda fu: client.loop.stop())
        client.loop.add_callback(tensorflow_scheduler_wrap)
        client.loop.start()
        result = global_future.result()
    else:
        result = client.sync(client.loop, tensorflow_scheduler_wrap)
    return result


