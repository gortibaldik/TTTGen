set -o xtrace
echo "Starting training with cluster_script.py"
export LD_LIBRARY_PATH="/lnet/aic/opt/cuda/cuda-10.1/cudnn/7.6/lib64:/lnet/aic/opt/cuda/cuda-10.1/lib64:/lnet/aic/opt/cuda/cuda-10.1/extras/CUPTI/lib64"
python3 ~/seq2seq_structure_aware/cluster_script.py --task_dir="/home/trebuna/seq2seq_structure_aware"
echo "cluster_script.py ended!"
