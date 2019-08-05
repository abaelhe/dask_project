#!/usr/bin/env python3
#coding:utf-8

"""
@Author: Abael He<abaelhe@icloud.com>
@file: dask_usage.py
@time: 8/5/19 11:33 AM
@license: All Rights Reserved, Abael.com
"""

import os, sys, itertools, collections



USAGE_INFO = '''
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

