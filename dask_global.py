#!/usr/bin/env python3
#coding:utf-8


"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_global.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


# Global Python Path
# client = global_cluster(addr='tls://abael.com:8786', asynchronous=False)
#MASTERS ='''gpu01.ops.zzyc.360es.cn'''.strip().splitlines()
MASTERS ='''abael.com'''.strip().splitlines()
#gpu01.ops.zzyc.360es.cn
MACHINES = '''
gpu02.ops.zzyc.360es.cn
gpu05.ops.zzyc.360es.cn
gpu06.ops.zzyc.360es.cn
gpu07.ops.zzyc.360es.cn
gpu08.ops.zzyc.360es.cn
gpu10.ops.zzyc.360es.cn
'''.strip().splitlines()
#gpu04.ops.zzyc.360es.cn : Driver Update Required.
#gpu02.ops.zzyc.360es.cn
#gpu10.ops.zzyc.360es.cn


from functools import partial
from tornado import autoreload, gen
if not getattr(autoreload, 'main_started', None):
    setattr(autoreload, 'main_started', True)
    autoreload.start()


import sys,os,six,threading,socket,logging,json,base64, struct, tornado


iostream_allows_memoryview = tornado.version_info >= (4, 5)

DASK_PYTHONHASHSEED = 0
DASK_PYTHON_INTERPRETER = '/usr/bin/python3.6'
DASK_READ_CHUNK_SIZE = 100000
DASK_REPO = os.path.expanduser('~/.dask')
DASK_DATA_POOL_DIR = os.path.abspath(os.path.expanduser('~/.dask/data_pool'))
DASK_MODEL_POOL_DIR = os.path.abspath(os.path.expanduser('~/.dask/model_pool'))
DASK_WORKSPACE = os.path.expanduser('~/dask-workspace')
DASK_PRIORITY = {
    'chief': 10,
    'master': 10,
    'ps': 50,
    'worker':100,
    'evaluator':100
}


sys.path.insert(0, DASK_REPO)
LD_LIBRARY_PATH = ['/usr/local/cuda/lib64', '/usr/local/cuda/extras/CUPTI/lib64']
list([LD_LIBRARY_PATH.append(x) for x in os.environ.get('LD_LIBRARY_PATH', '').split(':') if x not in LD_LIBRARY_PATH])
os.environ['LD_LIBRARY_PATH']=':'.join(LD_LIBRARY_PATH)


# place early to ensure dask configuration initialize early;
from distributed import Client,get_worker,get_client
from distributed.utils import thread_state, set_thread_state
from distributed.security import Security
from distributed.comm.tcp import ensure_bytes, nbytes, to_frames, from_frames, get_tcp_server_address


urllib_parse = six.moves.urllib_parse
urlparse = urllib_parse.urlparse
logger = logging.getLogger('distributed.preloading')


from dask_signal import *
from dask_usage import USAGE_INFO


log_dir = os.path.expanduser('~/dask/logs/') # ATTENTION: this is the remote location !
# TLS 模式注意事项: 1.路径为绝对路径, '~' 和通配符 必须先展开;    2.Client的 addr指定为tls协议格式"tls://host:port"
TLS_CA_FILE=os.path.expanduser('~/.dask/ca.crt')
TLS_CA_CERT=os.path.expanduser('~/.dask/ca.crt')
TLS_CA_KEY=os.path.expanduser('~/.dask/ca.key')

SECURITY = SEC = Security(tls_ca_file=TLS_CA_FILE, tls_scheduler_cert=TLS_CA_CERT, tls_scheduler_key=TLS_CA_KEY)
SECURITY_WORKER = Security(tls_ca_file=TLS_CA_FILE, tls_worker_cert=TLS_CA_CERT, tls_worker_key=TLS_CA_KEY)
SECURITY_CLIENT = Security(tls_ca_file=TLS_CA_FILE, tls_client_cert=TLS_CA_CERT, tls_client_key=TLS_CA_KEY, require_encryption=True)

SCHEDULER_PORT = 8786
#gpu09.ops.zzyc.360es.cn
GLOBAL_CLUSTER = list(map(lambda m:'tls://%s:%s'%(m, SCHEDULER_PORT), MASTERS))[0]


SSH_USER = 'heyijun'
SSH_MASTER_ip = 'gpu01.ops.zzyc.360es.cn'
SSH_PKEY = os.path.expanduser('~/.ssh/id_rsa')
SSH_PUB = os.path.expanduser('~/.ssh/id_rsa.pub')
SSH_WORKER_python = '/usr/bin/python3.6'
SSH_PORT = 22
SSH_MASTER_port = 11111
SSH_NANNY_port = 22222
SSH_WORKER_port = 33333
SSH_WORKER_nprocs = 1
SSH_WORKER_nthreads = 10


def cuda_free_indexes(dask_worker=None, dask_scheduler=None):
    name = dask_worker.name if dask_worker else socket.gethostname()
    gpu_indexes = []
    for line in os.popen('nvidia-smi pmon -c 1 -s m').read().splitlines()[2:]:
        line = line.split()
        if len(line) != 5:
            continue
        gpu_index, pid, cpu_gpu_type, fb, command = line
        if gpu_index.isdigit() and not pid.isdigit() and not fb.isdigit():
            gpu_indexes.append(int(gpu_index))
    return (name, gpu_indexes)


def node_thread(thrid, depth=10):
    frames = sys._current_frames()
    frame = frames[thrid]
    call_stack = []
    for i in range(depth):
        call_stack.append('%s:%s' % (frame.f_code.co_filename, frame.f_lineno))
        if frame.f_back:
            frame = frame.f_back
        else:
            break
    return call_stack


def node_threads():
    return [(thrid, '%s:%s'%(frame.f_code.co_filename, frame.f_lineno))  for thrid, frame in sys._current_frames().items()]


def global_cluster(addr=GLOBAL_CLUSTER, asynchronous=True, direct_to_workers=True):
    global GLOBAL_CLUSTER

    if isinstance(GLOBAL_CLUSTER, str):
        GLOBAL_CLUSTER = Client(address=addr, name='global.abael.com', security=SECURITY_CLIENT,
            asynchronous=asynchronous, direct_to_workers=False,
        )

    return Client(address=addr, name=socket.gethostname(), security=SECURITY_CLIENT,
            asynchronous=asynchronous, direct_to_workers=False,
            # serializers=['msgpack', 'dask', 'error'], deserializers=['msgpack','error', 'dask'],
        )

# `yield async(client, msg)` when return, that will ensure the msg data has been sync flushed into Kernel Cache.
# bypass the client's Batched Send Communication, give us best choice based on reality.
#             1. client.scheduler_comm: <class 'distributed.batched.BatchedSend'>     # periodically send.
#        2. client.scheduler_comm.comm: <class 'distributed.comm.tcp.TLS'>            # provide sync `async write`
# 3. client.scheduler_comm.comm.stream: <class 'tornado.iostream.SSLIOStream'>        # provide sync `async write`
# client.scheduler_comm.comm.stream.

@gen.coroutine
def async_send(client, msg):
    batch_send_com = client.scheduler_comm
    abst_comm = batch_send_com.comm
    iostream = abst_comm.stream

    msgs, batch_send_com.buffer = batch_send_com.buffer, []
    msgs.append(msg)
    context = {"sender": abst_comm._local_addr, "recipient": abst_comm._peer_addr},
    frames = yield to_frames(msg, serializers=batch_send_com.serializers, on_error="raise", context=context)

    lengths = [nbytes(frame) for frame in frames]
    length_bytes = [struct.pack("Q", len(frames))] + [struct.pack("Q", x) for x in lengths]

    futures = []

    if six.PY3 and sum(lengths) < 2 ** 27:  # 128MB
        futures.append(iostream.write(b"".join(length_bytes + frames)))  # send in one pass
    else:
        iostream.write(b"".join(length_bytes))  # avoid large memcpy, send in many
        bytes_since_last_yield = 0
        for frame in frames:
            if not iostream_allows_memoryview:
                frame = ensure_bytes(frame)
            future = iostream.write(frame)
            bytes_since_last_yield += nbytes(frame)
            futures.append(future)
            if bytes_since_last_yield > 2**28: # 256MB
                yield futures
                bytes_since_last_yield = 0
                futures.clear()

    if futures:
        yield futures

    raise gen.Return(sum(lengths))


def addr_name_map(client=None, asynchronous=True):
    client = client if client else global_cluster(asynchronous=asynchronous)

    def scheduler_info_to_addr_name_map(scheduler_info):
        return [(k, v['name']) for k, v in scheduler_info['workers'].items()]

    if client.asynchronous:
        result_future = gen.Future()
        scheduler_info_future= gen.Future()

        @gen.coroutine
        def async_scheduler_info(fu):
            sched_info = yield client.scheduler.identity()
            fu.set_result(sched_info)

        client.loop.add_callback(async_scheduler_info, scheduler_info_future)
        scheduler_info_future.add_done_callback(lambda fu: result_future.set_result(scheduler_info_to_addr_name_map(fu.result())))
        return result_future
    else:
        scheduler_info = client.sync(client.scheduler.identity)
        return scheduler_info_to_addr_name_map(scheduler_info)


def escape_cmd(cmd):
    b64cmd = base64.b64encode(cmd.encode() if isinstance(cmd, str) else cmd)
    b64exe = "echo '%s'  | base64 -d | bash" % b64cmd.decode()
    return b64exe


def safe_cmd(cmd, output=False):
    if output:
        return os.popen(escape_cmd(cmd)).read()
    else:
        return os.system(escape_cmd(cmd))


def cluster_safe_cmd(client, cmd, workers=None):
    exe = partial(safe_cmd, escape_cmd(cmd))
    if client.asynchronous:
        result_future = gen.Future()

        @gen.coroutine
        def async_exe(x, fu):
            r = yield client.run(exe, workers=workers)
            fu.set_result(r)
        client.loop.add_callback(partial(async_exe, exe, result_future))

        return result_future

    else:
        r = client.run(exe)
        return r


def model_cleanup(client, model_name, workers=None):
    m_key = '%s:' % model_name
    cmd = r"pkill -KILL -f %r " % m_key
    print('CLUSTER CMD: %r' % cmd)
    return cluster_safe_cmd(client, cmd, workers=workers)

