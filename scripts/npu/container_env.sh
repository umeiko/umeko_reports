# Source this inside the mzy-mindspeed container before any torch_npu work.
# Never modifies CANN/driver install; only sets env vars.
source /usr/local/Ascend/driver/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh
export PYTHONPATH=/mnt/models/CODE/MindSpeed-LLM-v2.3.0:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/ms/bin/python
