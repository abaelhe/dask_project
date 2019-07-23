#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_global.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


# Global Python Path

import sys,os,threading,socket
DASK_REPO = os.path.expanduser('~/.dask')
DASK_WORKSPACE = os.path.expanduser('~/dask-workspace')
DASK_PRIORITY = {'chief': 10,
                 'master': 10,
                 'ps': 50,
                 'worker':100,
                 'evaluator':100
                 }
sys.path.insert(0, DASK_REPO)
LD_LIBRARY_PATH = ['/usr/local/cuda/lib64', '/usr/local/cuda/extras/CUPTI/lib64']
list([LD_LIBRARY_PATH.append(x) for x in os.environ.get('LD_LIBRARY_PATH', '').split(':') if x not in LD_LIBRARY_PATH])
os.environ['LD_LIBRARY_PATH']=':'.join(LD_LIBRARY_PATH)

from tornado import autoreload
autoreload.start()

import numpy as np, pandas as pd, tensorflow as tf
from multiprocessing import get_all_start_methods


# place early to ensure dask configuration initialize early;
from distributed import Client, worker_client, get_worker, get_client, get_task_stream
from distributed.worker import logger
from distributed.utils import sync
from distributed.security import Security
from distributed.compatibility import Queue
from distributed.deploy.ssh import SSHCluster


import six, time,itertools,collections, traceback, dask, distributed,json, importlib
from toolz import merge
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.concurrent import (
    Future,
    is_future,
    chain_future,
    future_set_exc_info,
    future_add_done_callback,
    future_set_result_unless_cancelled,
)
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


urllib_parse = six.moves.urllib_parse
urlparse = urllib_parse.urlparse


log_dir = os.path.expanduser('~/dask/logs/') # ATTENTION: this is the remote location !
# TLS 模式注意事项: 1.路径为绝对路径, '~' 和通配符 必须先展开;    2.Client的 addr指定为tls协议格式"tls://host:port"
TLS_CA_FILE=os.path.expanduser('~/.dask/ca.crt')
TLS_CA_CERT=os.path.expanduser('~/.dask/ca.crt')
TLS_CA_KEY=os.path.expanduser('~/.dask/ca.key')

SECURITY = SEC = Security(tls_ca_file=TLS_CA_FILE, tls_scheduler_cert=TLS_CA_CERT, tls_scheduler_key=TLS_CA_KEY)
SECURITY_WORKER = Security(tls_ca_file=TLS_CA_FILE, tls_worker_cert=TLS_CA_CERT, tls_worker_key=TLS_CA_KEY)
SECURITY_CLIENT = Security(tls_ca_file=TLS_CA_FILE, tls_client_cert=TLS_CA_CERT, tls_client_key=TLS_CA_KEY, require_encryption=True)

SCHEDULER_PORT = 8786
MASTERS ='''gpu01.ops.zzyc.360es.cn'''.strip().splitlines()
#gpu08.ops.zzyc.360es.cn
GLOBAL_CLUSTER = list(map(lambda m:'tls://%s:%s'%(m, SCHEDULER_PORT), MASTERS))[0]
MACHINES = '''
gpu02.ops.zzyc.360es.cn
gpu05.ops.zzyc.360es.cn
gpu06.ops.zzyc.360es.cn
gpu07.ops.zzyc.360es.cn
gpu08.ops.zzyc.360es.cn
gpu09.ops.zzyc.360es.cn
gpu10.ops.zzyc.360es.cn
'''.strip().splitlines()
#gpu04.ops.zzyc.360es.cn : Driver Update Required.
#gpu02.ops.zzyc.360es.cn
#gpu10.ops.zzyc.360es.cn


SSH_USER = 'heyijun'
SSH_PKEY = os.path.expanduser('~/.ssh/id_rsa')
SSH_PUB  = os.path.expanduser('~/.ssh/id_rsa.pub')
SSH_PORT = 22
SSH_MASTER_ip = 'gpu01.ops.zzyc.360es.cn'
SSH_MASTER_port = 11111
SSH_NANNY_port = 22222
SSH_WORKER_port = 33333
SSH_WORKER_nprocs = 1
SSH_WORKER_nthreads = 10
SSH_WORKER_python = '/usr/bin/python3.6'


TF_PORT = 2222
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


def cuda_running_pids():
    processes_memory_usage = os.popen('nvidia-smi pmon -c 1 -s m').read().splitlines()[2:]
    pids = list(map(lambda x: x.split()[1], processes_memory_usage))
    return [int(pid_str) for pid_str in pids if pid_str.isdigit()]


def cuda_free_machines(client=None):
    client = client or GLOBAL_CLUSTER
    r = client.run(cuda_running_pids)
    return [node_url for node_url, pids in r.items() if len(pids) == 0]


def global_cluster(addr=GLOBAL_CLUSTER, timeout=100000):
    global GLOBAL_CLUSTER
    if isinstance(GLOBAL_CLUSTER, str):
        GLOBAL_CLUSTER = Client(address=addr, name=dask.config.get('client-name') or socket.gethostname(), security=SECURITY_CLIENT)
    return GLOBAL_CLUSTER


def tensorflow_devices(xla=None, gpu=True):
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    devices = []
    for device in device_lib.list_local_devices():
        devices.append({'name': device.name,
                        'memory_limit':device.memory_limit,
                        'device_type': device.device_type,
                        'byte_size': device.ByteSize(),
                        'locality': device.locality,
                        'physical_device_desc': device.physical_device_desc,
                        'incarnation': device.incarnation
                        })

    def gpu_filter(_devices, gpu_flag):
        return [x for x in _devices if x['name'].find('GPU')>0] if gpu_flag is True else _devices

    if xla is None:
        return gpu_filter([x for x in devices if x['name'].find('GPU')>0], gpu)
    elif xla is True:
        return gpu_filter([x for x in devices if x['name'].find('XLA') > 0], gpu)
    else:
        return gpu_filter([x for x in devices if x['name'].find('XLA') < 0], gpu)


def tensorflow_options(gpu_mem_fraction=0.9):
    import tensorflow as tf
    Config = tf.compat.v1.ConfigProto
    print('TENSORFLOW JIT STATUS: %s' %  tf.config.optimizer.get_jit())
    tf.config.optimizer.set_jit(1)
    tf_options = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True,
                                            gpu_options=tf.compat.v1.GPUOptions(
                                                per_process_gpu_memory_fraction=gpu_mem_fraction,
                                                force_gpu_compatible=True,
                                                allow_growth=True,
                                            ),
                                          )
    tf_options.graph_options.optimizer_options.opt_level = tf.compat.v1.OptimizerOptions.L1
    tf_options.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    tf_options.graph_options.optimizer_options.do_function_inlining = True
    return tf_options
    #s=tf_options.SerializeToString()
    #tf.compat.v1.ConfigProto.FromString(s)

# ref.: distributed:worker.py:run(server, comm, function, args=(), kwargs={}, is_coro=None, wait=True):
# support builtin special kwargs:  dask_worker:server, dask_scheduler:server, ONLY available: distributed.client._run()
def tensorflow_manager(model_name, tf_config=None, tf_options=None, node_url=None, scheduler_info=None):
    logger.info('dask manager start:\n');

    os.environ["TF_XLA_FLAGS"] = ("--tf_xla_cpu_global_jit " + os.environ.get("TF_XLA_FLAGS", ""))
    os.environ['XLA_FLAGS']='--xla_hlo_profile'
    import tensorflow as tf

    using_gpu_devices = tensorflow_devices(xla=True, gpu=True)
    using_device_names = sorted([x['name'] for x in using_gpu_devices])

    from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
    GlobalConfigProto = tf.compat.v1.ConfigProto.FromString(tf_options) if isinstance(tf_options, (str, bytes)) else TENSORFLOW_GLOBAL_OPTIONS
    cluster_spec, tf_task = tf_config['cluster'], tf_config['task']
    job_name, task_index = tf_task['type'], tf_task['index']
    tensorflow_addr = tf_config['cluster'][job_name][task_index]

    GlobalConfigProto.intra_op_parallelism_threads=len(using_gpu_devices)
    GlobalConfigProto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1


    #    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index) # 设定使用哪几块显卡

    dask_context = {
         'model_name': model_name,
         'tensor_task':'%s:%s' %(node_url, ','.join(using_device_names)),
         'worker_addr': node_url,
         'schduler_addr': scheduler_info,
         'workspace': DASK_WORKSPACE,
         'local_dir': os.getcwd(),
         'pid': os.getpid(),
          'thread_id': threading.get_ident(),
         'code': 0,
         'msg': ''
        }

    tf_config_json_str = json.dumps(tf_config)
    tf_context_json_str = json.dumps(dask_context)
    os.environ['TF_CONTEXT'] = tf_context_json_str
    os.environ['TF_CONFIG'] = tf_config_json_str
    model_path = os.path.join(DASK_REPO, 'model_pool', model_name)
    import_path = 'model_pool.%s.model' % model_name
    logger.info('dask worker, model_import:%s\n    tf_configProto:%s\n    tf_config:%s\n    dask_context:%s\n    sys.path:%s\n',
                import_path,
        repr(GlobalConfigProto),
        tf_config,
        dask_context,
        sys.path)

    cwd = os.getcwd()
    sys_path = sys.path[:]
    try:
        assert os.path.exists(model_path) and os.path.isfile(os.path.join(model_path, '__init__.py')) , 'Ensure model:"%s" upload to MasterNode:MODEL_POOL_DIR, and included a "__init__.py"'
        sys.path = [model_path, DASK_WORKSPACE] + sys_path
        os.chdir(model_path)
        # device_spec = tf.DeviceSpec(job=job_name, device_type="GPU", device_index=gpu_index, task=task_index, replica=1)
        #    device_spec_str = device_spec.to_string()
        # with tf.device(tf.compat.v1.train.replica_device_setter(cluster=cluster_spec)):
        #    tf_session = tf.compat.v1.Session(config=GlobalConfigProto)
        model = importlib.import_module(import_path)
        model.main(GlobalConfigProto)
    except Exception as e:
        os.chdir(cwd)
        fio = six.StringIO()
        traceback.print_exc(file=fio)
        logger.info('dask worker error:\ntf_configProto:%s\ndask_context:%s\nexception:%s\n', repr(GlobalConfigProto), dask_context, fio.getvalue())
        return (-1, model_name, fio.getvalue())
    else:
        logger.info('dask worker done:\ntf_configProto:%s\ndask_context:%s', repr(GlobalConfigProto), dask_context)
    finally:
        sys.path = sys_path
        os.chdir(cwd)

    return (0, model_name, '')

@gen.coroutine # eg.: job_counts={'ps':10, 'workers':100}, ParameterServers:10, CUDAworkers:100
def tensorflow_scheduler(model_name, client=None, tf_options=None, tf_port=None, **tf_cluster_spec):
    active_machines_dict = collections.defaultdict(list)

    for node_url, w in client.scheduler_info()['workers'].items():
        node_host, node_port = urlparse(node_url).netloc.rsplit(':', 1)
        active_machines_dict[node_host].append(node_url)

    for node_addrs in active_machines_dict.values():
        node_addrs.sort()

    active_machines =sorted(set(active_machines_dict.keys()))
    if sum(tf_cluster_spec.values()) > len(active_machines):
        raise ValueError("Dask cluster Need %d machines, have %d live" % (sum(tf_cluster_spec.values()), len(active_machines)))

    cluster_spec_json = {job_name: [] for job_name in tf_cluster_spec}
    dask_spec = {job_name: [] for job_name in tf_cluster_spec}
    jobs = {}
    iter_active_machines = iter(sorted(active_machines_dict.items(), key=lambda x:x[0]))
    for job_name, machine_total in tf_cluster_spec.items():
        for job_machine_index in range(machine_total):
            (node_host, node_urls) = next(iter_active_machines)
            first_selected = node_urls[0]
            host, dask_port = urlparse(first_selected).netloc.rsplit(':',1)
            dask_spec[job_name].append({'url':first_selected, 'job':job_name,  'task_index':job_machine_index})
            cluster_spec_json[job_name].append('%s:%d' % (host, int(TF_PORT))) # DASK PORT => TF PORT
            jobs[first_selected] = (job_name, job_machine_index)

    tf_configs ={node_url:{'cluster':cluster_spec_json, 'task':{'type':task[0], 'index':task[1]}} for node_url, task in jobs.items()}
    logger.info('Model Schedule %s: \n  tf_configs:%s\n  dask_spec:%s\n  jobs:%s', model_name, tf_configs, dask_spec, jobs)

    tf_options_bytes=tf_options.SerializeToString()
    result = {}
    thread_pool_executor = ThreadPoolExecutor(max_workers=len(tf_configs))

    chief_or_master = [(node_url, tf_config) for node_url, tf_config  in tf_configs.items() if tf_config['task']['type'] in ('chief', 'master')]
    pses = [(node_url, tf_config) for node_url, tf_config in tf_configs.items() if tf_config['task']['type'] in ('ps',)]
    workers = [(node_url, tf_config) for node_url, tf_config in tf_configs.items() if tf_config['task']['type'] in ('worker',)]

    with open('./tf_configs.json', 'wb') as jsf:
        jsf.write(json.dumps(chief_or_master).encode())
        jsf.write(json.dumps(pses).encode())
        jsf.write(json.dumps(workers).encode())

    for node_url, tf_config in tf_configs.items():
        func = partial(client.submit,tensorflow_manager, model_name,
                               tf_config=tf_config,
                               tf_options=tf_options_bytes,
                               node_url = node_url,
                               scheduler_info=client.scheduler_info(),
                               priority = DASK_PRIORITY[tf_config['task']['type']],
                      workers=[node_url], key='%s:%s'%(tf_config['task']['type'], tf_config['task']['index']))
        result[node_url] = yield client.loop.run_in_executor(thread_pool_executor, func)
    result = yield result
    result = merge(result.values())
    logger.info('RET EXECUTION:\n')
    for k, v in result.items():
        logger.info('    %s: %s', k, v)
    raise gen.Return((result, cluster_spec_json, dask_spec))


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
    client = global_cluster() if client is None else client
    options = options or TENSORFLOW_GLOBAL_OPTIONS
    tensorflow_scheduler_wrap = partial(tensorflow_scheduler, client=client, tf_options=options, tf_port=port, **kwargs)
    result, tf_cluster_spec, dask_spec = sync(client.loop, tensorflow_scheduler_wrap, model_name)
    # errs = ['Error Tensorflow CUDA:']
    # for dask_node_url, dask_ctx in result.items():
    #     if dask_ctx['code'] != 0:
    #         errs.append('    %s, %s'%(dask_node_url, dask_ctx['msg']))
    # raise Exception('\n'.join(errs))
    return (tf_cluster_spec, dask_spec, result)


TENSORFLOW_GLOBAL_OPTIONS = options = tensorflow_options()
