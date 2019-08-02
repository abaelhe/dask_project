#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_tensorflow.py
@time: 7/26/19 12:05 PM
@license: All Rights Reserved, Abael.com
"""

import six, os, sys, time, threading, itertools, collections, socket,subprocess, json, datetime
from functools import partial,wraps,lru_cache

from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.process import Subprocess
from tornado.gen import coroutine, Return, \
    Future, future_add_done_callback, future_set_result_unless_cancelled, future_set_exc_info

from distributed import Client, get_worker
import logging


logger = logging.getLogger(__name__)


from dask_global import (cuda_free_indexes, global_cluster, urlparse,
    DASK_PYTHONHASHSEED, DASK_PYTHON_INTERPRETER,
    DASK_READ_CHUNK_SIZE, DASK_WORKSPACE, DASK_DATA_POOL_DIR, DASK_MODEL_POOL_DIR)


MODEL_POOL_USAGE_INFO = '''
0. 名词解释:
    MODEL_POOL: 源代码池目录, 当前为SSH URI: "gpu01:/data01/model_pool/"
    DATA_POOL: 数据池和配置池目录, 当前为 SSH URI: "gpu01:/data01/model_pool/"
    MODEL_NAME: 模型名字;

1. 首先上传 MODEL 到 目录里:
    1.1. MODEL 的 源代码 部分:
        首选打包成ZIP Python Package 方式上传: "MODEL_NAME.zip", 请保证可执行入口文件 "__main__.py" 在 "MODEL_NAME.zip" 顶层;

    1.2. MODEL 的 配置/数据 部分:
        1.2.1 将配置文件目录传成:
            gpu01:/data01/data_pool/MODEL_NAME/config/
        1.2.2 将数据文件目录传成:
            gpu01:/data01/data_pool/MODEL_NAME/data/

    1.3. 关于 CWD (当前工作目录), 运行时CWD(模型平台用 python3.6 调模型可执行入口文件后)为:
            gpu01:/data01/data_pool/MODEL_NAME/

2. 以 模型 pageclassify 为例, 整个目录:
     $ cd pageclassify/
     $ $ ls __main__.py 2> /dev/null || echo '创建可执行入口文件 "__main__.py" 先'
     $ zip ../pageclassify.zip  -r *     -x 'data/*' -x 'config/*'  -x 'ckpt/*' -x 'log/*' -x '*.pyc'
     $ scp ../pageclassify.zip  gpu01:/data01/model_pool/
     $ echo "以上 模型源码打包成 Zip Python Package 上传完成."
     $ scp -r  ./config/   gpu01:/data01/data_pool/pageclassify/config/
     $ scp -r  ./data/   gpu01:/data01/data_pool/pageclassify/data/
     $ echo "以上将 配置文件目录 ./config 和 数据目录 ./data 传到对应位置完成. "


3. 环境变量( 已预先设定, 可在MODEL模型源代码内通过 `os.environ["env_key"]` 方式提取( 为json的先json.loads(env_val) )  ):
    TF_CONTEXT:        模型运行上下文, json;
    TF_CONFIG:         TF_CONFIG 集群配置, json;
    TF_MODEL_POOL_DIR:        模型池的位置,
    TF_DATA_POOL_DIR:         数据/配置池的位置
    TF_MODEL_ENTRY: 模型入口位置, 将会被 python 解析器直接调用, 例: "python3.6  model_pool/credit.zip" 或 "python3.6  model_pool/credit"   
    TF_LOG_DIR:          日志文件目录,
    TF_SAVE_DIR:         模型CheckPoint数据保存目录,
    CUDA_VISIBLE_DEVICES: 当前机器上可用的GPU设备序列号, 例如 "0,1"
'''


TF_PORT = 1111
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


def gpu_filter(_devices, gpu_flag):
    return [x for x in _devices if x['name'].find('GPU') > 0] if gpu_flag is True else _devices


class TFActor(object):
    def __del__(self):
        self.dask_worker = None

    def __name__(self):
        return '%s@%s'%(self.model_name, self.name)

    def __init__(self, model_name, *args, tf_config=None, tf_option=None, scheduler_info=None, **kwargs):
        logger.info('Accepted Tensorflow Job:%s, Options:%s, Scheduler:%s', tf_config, tf_option, scheduler_info)
        self.dask_worker = get_worker()
        self.name = self.dask_worker.name
        self.hostname = socket.gethostname()
        self.address = self.dask_worker.address

        self.scheduler_info = scheduler_info
        self.devices = dict(tensorflow_devices())

        self.model_name = model_name[:-4] if model_name.endswith('.zip') else model_name
        self.tf_option = json.loads(tf_option) if isinstance(tf_option, str) else tf_option
        self.tf_config = json.loads(tf_config) if isinstance(tf_config, str) else tf_config

        self.dask_cwd = os.path.abspath(os.getcwd())
        self.tf_model_pool_dir = os.path.abspath(DASK_MODEL_POOL_DIR)
        self.tf_data_pool_dir = os.path.abspath(DASK_DATA_POOL_DIR)
        self.tf_data_dir = os.path.join(self.tf_data_pool_dir, self.model_name)
        self.tf_config_dir = os.path.join(self.tf_data_dir, 'config')
        self.tf_save_dir = os.path.join(self.tf_data_dir, 'save')
        self.tf_log_dir = os.path.join(self.tf_data_dir, 'log')

        self.future_chunk_size = DASK_READ_CHUNK_SIZE
        self.proc = self.stdout = self.stderr = self._stderr_buf = self._stdout_buf = None
        self.args = args
        self.kwargs = kwargs
        logger.info('TFNode:(Tensorflow Node) inited at hostname:%s, addr:%s, model:%s', self.hostname, self.address, self.model_name)

    def device_info(self, xla=None, gpu=True):
        if xla is None:
            return gpu_filter([v for (x,v) in self.devices.items() if v['name'].find('GPU') >= 0], gpu_flag=gpu)
        elif xla is True:
            return gpu_filter([v for (x,v) in self.devices.items() if v['name'].find('XLA') >= 0], gpu_flag=gpu)
        else:
            return gpu_filter([v for (x,v) in self.devices.items() if v['name'].find('XLA') <  0], gpu_flag=gpu)

    def tensorflow_env(self, tf_option, tf_config, dask_context, cuda_indexes=None):
        model_entrypoint = os.path.join(self.tf_model_pool_dir, self.model_name)
        zip_ep, pkg_ep = model_entrypoint + '.zip', os.path.join(model_entrypoint, '__main__.py')
        if os.path.exists(zip_ep) and os.path.isfile(zip_ep):
            model_entrypoint = zip_ep
        elif os.path.exists(pkg_ep) and os.path.isfile(pkg_ep):
            model_entrypoint = pkg_ep
        else:
            raise Exception(MODEL_POOL_USAGE_INFO)

        env_dict = dict(
            XLA_FLAGS='--xla_hlo_profile',
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
            PYTHONUNBUFFERED='True',
        )

        if cuda_indexes:  # we explicitly assign GPU indexes to use; let tensorflow aware of ONLY these indexes
            env_dict['CUDA_VISIBLE_DEVICES'] = cuda_indexes

        logger.info('dask model, entrypoint:%s\n  tf_option:%s\n  tf_config:%s\n  dask_context:%s\n  sys.path:%s\n',
            model_entrypoint, repr(tf_option), tf_config, dask_context, sys.path)

        return env_dict

    def future_log_read(self, data, source=None):
        #log Async Push: 实现异步的日志推送, 相当于 "tail -f" 效果
        logger.info('%s' % data)
        print(data)

    @coroutine
    def tensorflow_model(self, env_dict):
        model_cwd = env_dict['TF_DATA_DIR']
        model_entry = env_dict['TF_MODEL_ENTRYPOINT']

        cwd = os.getcwd()
        sys_path = sys.path[:]
        loop = self.dask_worker.loop

        try:
            os.chdir(model_cwd)

            os.system('mkdir -p "%s" "%s" "%s" "%s"' % (self.tf_data_dir, self.tf_config_dir, self.tf_save_dir, self.tf_log_dir))
            stdout_path = os.path.join(self.tf_log_dir, '%s.%s.log' % (self.model_name, self.hostname))
            stderr_path = os.path.join(self.tf_log_dir, '%s.%s.err.log' % (self.model_name, self.hostname))
            self.stdout = open(stdout_path, 'ab+')
            self.stderr = open(stderr_path, 'ab+')
            self._stdout_buf = bytearray(self.future_chunk_size)
            self._stderr_buf = bytearray(self.future_chunk_size)
            def auto_close_subproc():
                if self.stdout:
                    self.stdout.close()
                if self.stderr:
                    self.stderr.close()
                self._stdout_buf = self._stderr_buf = None

            cmd = [sys.executable, r'-u', model_entry]
            logger.info('Subprocess: %r', cmd)
            self.proc = Subprocess(cmd, executable=DASK_PYTHON_INTERPRETER,
                #bufsize=0, #encoding=sys.getdefaultencoding(),
                pass_fds=(0, 1, 2), stdin=sys.stdin, stdout=self.stdout, stderr=self.stderr,
#                stdout=subprocess.STDOUT,
#                stderr=subprocess.STDOUT,
                #stdin=Subprocess.STREAM, stdout=Subprocess.STREAM, stderr=Subprocess.STREAM,
                 preexec_fn=None, shell=True, cwd=model_cwd, env=env_dict, universal_newlines=False,
                 restore_signals=True, start_new_session=False,
            )

#            self.proc.stdout.set_close_callback(lambda :self.future_log_read('\nstdout CLOSED.\n', source='stdout'))
#            self.proc.stderr.set_close_callback(lambda :self.future_log_read('\nstderr CLOSED.\n', source='stderr'))

            @coroutine
            def _future_stdio_read(self, io_type=None):
                try:
                    if io_type == 'stdout':
                        num_bytes = yield self.proc.stdout.read_into(self._stdout_buf, partial=True)
                        if num_bytes is not None and num_bytes > 0:
                            data = memoryview(self._stdout_buf)[:num_bytes] if num_bytes > 0 else None
                            self.future_log_read(data, io_type)
                    elif io_type == 'stderr':
                        num_bytes = yield self.proc.stderr.read_into(self._stderr_buf, partial=True)
                        if num_bytes is not None and num_bytes > 0:
                            data = memoryview(self._stderr_buf)[:num_bytes] if num_bytes > 0 else None
                            self.future_log_read(data, io_type)
                except StreamClosedError:
                    return
                loop.add_callback(partial(self._future_stdio_read, io_type=io_type))
                # this 'read_into() DO MUST call after self.future_packet_read(data)! '

#            loop.add_callback(partial(_future_stdio_read, io_type='stdout'))
#            loop.add_callback(partial(_future_stdio_read, io_type='stderr'))
            retval = yield self.proc.wait_for_exit(raise_error=True)
            auto_close_subproc()
            raise Return(retval)
        finally:
            sys.path = sys_path
            os.chdir(cwd)
        raise Return(-1)

    @coroutine
    def start(self):
        # If this NODE selected for this task
        node_name, node_port, cuda_indexes, dask_url = self.tf_config.pop('dask').split(':', 3)
        job_name, task_index = self.tf_config['task']['type'], self.tf_config['task']['index']
        tensorflow_addr = self.tf_config['cluster'][job_name][task_index]
        logger.info('Dask Manager matched: %s:%s, %s:%s\n', job_name, task_index, self.name, self.address)

        using_xla_gpu_devices = self.device_info(xla=True, gpu=True)
        using_xla_gpu_device_names = sorted([x['name'] for x in using_xla_gpu_devices])

        import tensorflow as tf
        if isinstance(self.tf_option, (str, bytes)):
            GlobalConfigProto = tf.compat.v1.ConfigProto.FromString(self.tf_option)
        elif self.tf_option is not None:
            GlobalConfigProto = self.tf_option
        else:
            GlobalConfigProto = tensorflow_options()

        dask_context = {
             'model_name': self.model_name,
             'tensor_task':'%s:%s' %(self.name, ','.join(using_xla_gpu_device_names)),
             'worker_addr': self.address,
             'schduler_addr': self.scheduler_info,
             'workspace': DASK_WORKSPACE,
             'local_dir': os.getcwd(),
             'pid': os.getpid(),
              'thread_id': threading.get_ident(),
             'code': 0,
             'msg': ''
            }

        env_dict = self.tensorflow_env(GlobalConfigProto, self.tf_config, dask_context, cuda_indexes=cuda_indexes)
        retval = yield self.tensorflow_model(env_dict)
        raise Return(retval)

# ref.: distributed:worker.py:run(server, comm, function, args=(), kwargs={}, is_coro=None, wait=True):
# support builtin special kwargs:  dask_worker:server, dask_scheduler:server, ONLY available: distributed.client._run()


def tensorflow_gen_config(free_node_name_map=None, save='~/tf_configs.json', **tf_cluster_spec):
    if free_node_name_map:
        r = [(url, name, indexes) for url, (name, indexes) in free_node_name_map.items() if len(indexes) > 0]
    else:
        client = global_cluster(asynchronous=False)
        r =[(url, name, indexes) for url, (name, indexes) in client.run(cuda_free_indexes).items() if len(indexes) > 0]

    free_node_urls = sorted(r, key=lambda x: -len(x[-1]))
    free_gpu_devices = collections.deque()
    for (url, name, indexes) in free_node_urls:
        free_gpu_devices.extend((url, name, idx) for idx in indexes)

    if len(free_gpu_devices) < 3:
        raise Exception('All Machines is busy, Total Available:%s' % len(free_node_urls))

    if len(tf_cluster_spec) == 0:
        tf_cluster_spec['chief'] = 1
        tf_cluster_spec['ps'] = 1
        tf_cluster_spec['worker'] = max(1, len(free_gpu_devices)-2)
#        tf_cluster_spec['worker'] = max(1, (len(free_gpu_devices)-2) // 10)# 预计同时会有10个同事跑GPU任务; 公平调度策略, 预留10个;

    tf_cluster = collections.defaultdict(list)
    port_allocation = collections.defaultdict(lambda : (TF_PORT - 1))
    chief_allocation = tf_cluster_spec.pop('chief', 1)
    ps_allocation = tf_cluster_spec.pop('ps', 1)
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
        with open(save, 'wb') as jsf:
            jsf.write(json.dumps(tf_configs).encode())

    return tf_configs


@coroutine # eg.: job_counts={'ps':10, 'workers':100}, ParameterServers:10, CUDAworkers:100
def tensorflow_scheduler(model_name, client=None, tf_option=None, tf_port=None, **tf_cluster_spec):
    cuda_free_map =yield client.run(cuda_free_indexes)
    tf_configs = tensorflow_gen_config(free_node_name_map=cuda_free_map, **tf_cluster_spec)
    logger.info('Model Schedule %s: \n  tf_configs:%s\n\n', model_name, tf_configs)

    tf_option = tf_option if isinstance(tf_option, (str, bytes)) else (tf_option.SerializeToString() if tf_option else tf_option)
    scheduler_info =yield client.scheduler.identity()

    s_time = time.time()
    dt = datetime.datetime.now()

    result_future = Future()
    result_future.tf_configs = tf_configs
    result_future.tf_option = tf_option
    result_future.cuda_map = cuda_free_map

    results = {}

    @coroutine
    def submit_task(tf_config, tf_option=None, scheduler_info=None, client=None):
        job_name, job_index = tf_config['task']['type'], tf_config['task']['index']
        node_name, task_port, cuda_index, node_url = tf_config['dask'].split(':', 3)
        task_key = '%s:%s:%s@%s:%s' % (model_name, job_name, job_index, node_name, cuda_index)
        actor = yield client.submit(TFActor,
            model_name, tf_config=tf_config, tf_option=tf_option, scheduler_info=scheduler_info,
            key=task_key,
            workers=[node_url],
            retries=10,
            priority=0,
            allow_other_workers=False,
            actor=True,
        )
        logger.info('Start: %s' % task_key)
        startup = yield actor.start()
        actor.result = startup
        logger.info('Ready: %s' % task_key)
        raise Return((task_key.partition('@')[0].split(':', 1)[1], actor))

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

    chief_configs.sort(key=lambda cfg: cfg['task']['index'])
    ps_configs.sort(key=lambda cfg: cfg['task']['index'])
    other_configs.sort(key=lambda cfg: cfg['task']['index'])
    
    submit_task_wrap = partial(submit_task, tf_option=tf_option, client=client, scheduler_info=scheduler_info)
    chief_actors = yield {submit_task_wrap(tf_cfg) for tf_cfg in chief_configs}
    import pdb;pdb.set_trace()
    ps_actors = yield {submit_task_wrap(tf_cfg) for tf_cfg in ps_configs}
    other_actors = yield {submit_task_wrap(tf_cfg) for tf_cfg in other_configs}
    chief_actors.update(ps_actors)
    chief_actors.update(other_actors)

    raise Return(chief_actors)


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
    client = global_cluster(asynchronous=True) if client is None else client
    import tensorflow as tf
    options =(tf.compat.v1.ConfigProto.FromString(options) if isinstance(options, (str,bytes)) else (options if options else tensorflow_options())).SerializeToString()
    tensorflow_scheduler_wrap = partial(tensorflow_scheduler, model_name, client=client, tf_option=options, tf_port=port, **kwargs)
    if client.asynchronous:
        result = [None]

        def future_result(fut):
            client.loop.stop()
            result[0] = fut.result()

        future = tensorflow_scheduler_wrap()
        client.loop.add_future(future, lambda fut: future_result)
        client.loop.start()
        result = result[0]
    else:
        result = client.sync(client.loop, tensorflow_scheduler_wrap)
    return result


