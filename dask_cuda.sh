pgrep -f dask_cuda.py && sudo pkill -f dask_cuda.py
sudo rm -rf /home/heyijun/dask-workspace ; mkdir -p  /home/heyijun/dask-workspace

rpm  -q fuse-sshfs --quiet || sudo yum -y install   fuse-sshfs snappy-devel lz4-devel zlib-devel blosc-devel python36-blosc
# IN SSH MASTER SERVER:  sudo  sed -ri 's/^.*MaxSessions.*$/MaxSessions 10000/' /etc/ssh/sshd_config && sudo systemctl restart sshd
SSHFS_MOUNTPOINT=/home/heyijun/.dask
#mount | grep -q "${SSHFS_MOUNTPOINT}"  && fusermount -u "${SSHFS_MOUNTPOINT}"
#mkdir -p  ${SSHFS_MOUNTPOINT}
#mount | grep -q "${SSHFS_MOUNTPOINT}" || sshfs gpu08:${SSHFS_MOUNTPOINT}  ${SSHFS_MOUNTPOINT}   -o reconnect,sshfs_sync,no_readahead,sync_readdir,cache=no,disable_hardlink,follow_symlinks,no_check_root,large_read

sudo rm -rf /home/heyijun/.config/dask/
mkdir -p    /home/heyijun/.config/dask/

# Hosts mapping
#[ -e /etc/hosts ] && sudo grep 'ops.zzyc.360es.cn' -q -v /etc/hosts && sudo cat /home/heyijun/.dask/hosts >> /etc/hosts

# SSH
rm -rf /home/heyijun/.ssh/ && cp -pr   /home/heyijun/.dask/ssh  /home/heyijun/.ssh

#function dask_env(){
#    cp -f   /home/heyijun/.dask/dask.yaml   /home/heyijun/.config/dask/dask.yaml
#    chmod a+r /home/heyijun/.config/dask/dask.yaml
#}

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
#dask_env

# CUDA Kernel, reload driver without restart online machine !
sudo nvidia-smi -L || cuda_env


# Tensorflow with GPU Enable
/usr/local/bin/pip3.6  show tensorflow-gpu > /dev/null || sudo /usr/local/bin/pip3.6 install --index-url=https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check tensorflow-gpu blosc snappy


# Join DASK Cluster, NOTE: if specified `--scheduler-file`, then make sure it exists with right permissions and valid!!!
# Multi-Workers:   --nthreads $( python3.6 -c "import multiprocessing as mp;print(mp.cpu_count())" )

# tls://gpu08.ops.zzyc.360es.cn:8786
pgrep -f 'dask_cuda.py' && sudo pkill -f 'dask_cuda.py'
rm -rf ~/dask-workspace/*

PYTHONPATH=$([ -z "${PYTHONPATH}" ] && echo '/home/heyijun/.dask' || echo "/home/heyijun/.dask:${PYTHONPATH}") nohup \
  python3.6  ~/.dask/dask_cuda.py --name $(sudo hostname | sed "s/.ops.zzyc.360es.cn//") --reconnect --nthreads 1 \
    --tls-ca-file ~/.dask/ca.crt --tls-cert ~/.dask/ca.crt --tls-key ~/.dask/ca.key \
    --local-directory  ~/dask-workspace \
    --preload ~/.dask/dask_global.py \
    --scheduler-file  ~/.dask/dask_scheduler.yaml \
    1>~/dask-workspace/dask-cuda.log 2>>~/dask-workspace/dask-cuda.log &




