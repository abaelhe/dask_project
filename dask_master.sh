pgrep -f dask_master.py && sudo pkill -f dask_master.py

rpm  -q fuse-sshfs --quiet || sudo yum -y install   fuse-sshfs
sudo  sed -ri 's/^.*MaxSessions.*$/MaxSessions 10000/' /etc/ssh/sshd_config && sudo systemctl restart sshd
SSHFS_MOUNTPOINT=/home/heyijun/.dask/model_pool
mkdir -p  ${SSHFS_MOUNTPOINT}
sudo rm -rf  /home/heyijun/dask-workspace/*


# Hosts mapping
#[ -e /etc/hosts ] && sudo grep 'ops.zzyc.360es.cn' -q -v /etc/hosts && sudo cat /home/heyijun/.dask/hosts >> /etc/hosts

# SSH
rm -rf /home/heyijun/.ssh/ && cp -pr   /home/heyijun/.dask/ssh  /home/heyijun/.ssh

function dask_env(){
    sudo rm -rf /home/heyijun/.config/dask/
    mkdir -p    /home/heyijun/.config/dask/
    chmod a+r /home/heyijun/.dask/dask.yaml
}

function cuda_env(){
    # CUDA Install
    sudo rpm -ivh ~/.dask/cuda-repo-rhel7-10.1.168-1.x86_64.rpm;
    sudo yum -y update freetype mesa-libGL cuda
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


    # CUDA Library
    sudo ln -svf  /usr/lib64/libnvidia-gtk3.so.410.72 /usr/local/cuda/lib64/libnvidia-gtk3.so
    sudo ln -svf  /usr/lib64/libnvidia-gtk2.so.418.67  /usr/local/cuda/lib64/libnvidia-gtk2.so

    echo 'libcublas,libcublasLt,libnvblas'| tr ','  '\n' |while read lib; do
        sudo ln -svf "/usr/lib64/${lib}.so.10"  "/usr/local/cuda/lib64/${lib}.so.10.0"
    done
    echo 'libcudart,libcufft,libcurand,libcusolver,libcusparse,libcudnn'| tr ','  '\n' |while read lib; do
        sudo ln -svf "/usr/local/cuda-10.1/targets/x86_64-linux/lib/${lib}.so" "/usr/local/cuda/lib64/${lib}.so.10.0"
    done

    lsmod | grep -q nvidia_drm     && sudo rmmod nvidia_drm
    lsmod | grep -q nvidia_modeset && sudo rmmod nvidia_modeset
    lsmod | grep -q nvidia_uvm     && sudo rmmod nvidia_uvm
    lsmod | grep -q nvidia         && sudo rmmod nvidia
}


# DASK Config
dask_env

# CUDA Kernel, reload driver without restart online machine !
sudo nvidia-smi || cuda_env


# Tensorflow with GPU Enable
/usr/local/bin/pip3.6  show tensorflow-gpu > /dev/null || sudo /usr/local/bin/pip3.6 install --index-url=https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check tensorflow-gpu django lz4


# Join DASK Cluster, NOTE: if specified `--scheduler-file`, then make sure it exists with right permissions and valid!!!
# Multi-Workers:   --nthreads $( python3.6 -c "import os;print(os.cpu_count())" )


pgrep -f 'dask_master.py' && sudo pkill -f 'dask_master.py'
mkdir -p  ~/dask-workspace/ && \
   PYTHONPATH=$( [ -z "${PYTHONPATH}" ] && echo "/home/heyijun/.dask" || echo "/home/heyijun/.dask:${PYTHONPATH}" )  \
   nohup python3.6  ~/.dask/dask_master.py --host 0.0.0.0 --port 8786 --dashboard-address 0.0.0.0:8787 \
    --protocol tls  --tls-ca-file ~/.dask/ca.crt --tls-cert ~/.dask/ca.crt --tls-key ~/.dask/ca.key \
    --local-directory  ~/dask-workspace \
    --scheduler-file  ~/.dask/dask_scheduler.yaml \
    --preload ~/.dask/dask_global.py \
    1> ~/dask-workspace/dask-master.log  2>> ~/dask-workspace/dask-master.log  &





pgrep -f 'dask_master.py' && sudo pkill -f 'dask_master.py'
mkdir -p  ~/dask-workspace/ && \
   PYTHONPATH=$( [ -z "${PYTHONPATH}" ] && echo "/home/heyijun/.dask" || echo "/home/heyijun/.dask:${PYTHONPATH}" )  \
   python3.6  ~/.dask/dask_master.py --host 0.0.0.0 --port 8786 --dashboard-address 0.0.0.0:8787 \
    --protocol tls  --tls-ca-file ~/.dask/ca.crt --tls-cert ~/.dask/ca.crt --tls-key ~/.dask/ca.key \
    --local-directory  ~/dask-workspace \
    --scheduler-file  ~/.dask/dask_scheduler.yaml \
    --preload ~/.dask/dask_global.py \
    1> ~/dask-workspace/dask-master.log  2>> ~/dask-workspace/dask-master.log  &

