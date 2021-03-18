set -o xtrace
echo "Starting training with cluster_script.py"
python3 cluster_script.py --task_dir="."
echo "cluster_script.py ended!"
