#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: debug.sh.py
@time: 8/11/19 11:34 PM
@license: All Rights Reserved, Abael.com
"""


import os, sys, itertools, collections


import asyncio
from dask.distributed import Scheduler, Worker, Client
from dask_tensorflow import start_tensorflow, startup_actors


async def A():
    async with Scheduler(host='tcp://127.0.0.1', port=8786, protocol='tcp') as s:
        async with Worker(s.address, resources={'CPU':8})              as w0, \
                   Worker(s.address, resources={'GPU':128, 'CPU':128}) as w1:

            async with Client(s.address, asynchronous=True) as client:
                future = client.submit(lambda x: x + 1, 10, resources={'GPU': 16, 'CPU':16})
                result = await future
                print(result)


asyncio.get_event_loop().run_until_complete(A())


