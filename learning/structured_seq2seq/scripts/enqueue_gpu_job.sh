rm ~/seq2seq_structure_aware/logs/scr_gpu_*

qsub -q gpu.q -cwd -o ~/seq2seq_structure_aware/logs/scr_gpu_output.txt -e ~/seq2seq_structure_aware/logs/scr_gpu_errors.txt -pe smp 1 -l gpu=1,mem_free=16G,act_mem_free=16G,h_data=32G ~/seq2seq_structure_aware/scripts/cluster_script.sh
