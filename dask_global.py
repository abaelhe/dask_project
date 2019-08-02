#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_global.py
@time: 7/4/19 10:03 AM
@license: All Rights Reserved, Abael.com
"""


# Global Python Path

from tornado import autoreload
autoreload.start()

import sys,os,six,threading,socket
DASK_REPO = os.path.expanduser('~/.dask')
DASK_PYTHONHASHSEED = 0
DASK_PYTHON_INTERPRETER = '/usr/bin/python3.6'
DASK_READ_CHUNK_SIZE = 100000
DASK_DATA_POOL_DIR = '/home/heyijun/.dask/data_pool'
DASK_MODEL_POOL_DIR = '/home/heyijun/.dask/model_pool'
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


# place early to ensure dask configuration initialize early;
from distributed import Client
from distributed.utils import sync
from distributed.security import Security

from tornado.process import Subprocess
Subprocess.initialize()
urllib_parse = six.moves.urllib_parse
urlparse = urllib_parse.urlparse


import logging
logger = logging.getLogger(__name__)


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
#gpu09.ops.zzyc.360es.cn
GLOBAL_CLUSTER = list(map(lambda m:'tls://%s:%s'%(m, SCHEDULER_PORT), MASTERS))[0]
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


def cuda_free_machines(client=None):
    client = client or GLOBAL_CLUSTER
    r = sync(client.loop, client.run, cuda_free_indexes)
    return sorted([(node_url,name, indexes) for node_url, (name, indexes) in r.items() if len(indexes) > 0], key=lambda x: -len(x[-1]))


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
