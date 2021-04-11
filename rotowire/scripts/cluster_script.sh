set -o xtrace
echo "Starting training with cluster_script.py"
export LD_LIBRARY_PATH="/lnet/aic/opt/cuda/cuda-10.1/cudnn/7.6/lib64:/lnet/aic/opt/cuda/cuda-10.1/lib64:/lnet/aic/opt/cuda/cuda-10.1/extras/CUPTI/lib64"
python3 ~/rotowire/cs_baseline.py --path=ni_tfrecord
echo "cluster_script.py ended!"