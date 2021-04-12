rm ~/rotowire/logs/gpu_*

qsub -q gpu.q \
     -cwd \
     -o ~/rotowire/logs/gpu_output.txt \
     -e ~/rotowire/logs/gpu_errors.txt \
     -pe smp 1 -l \
     gpu=1,mem_free=16G,act_mem_free=16G,h_data=32G \
     ~/rotowire/scripts/cluster_script.sh
