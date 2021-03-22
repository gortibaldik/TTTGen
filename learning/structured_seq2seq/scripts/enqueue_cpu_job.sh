rm ~/seq2seq_structure_aware/logs/scr_cpu*

qsub -q cpu.q -cwd -o ~/seq2seq_structure_aware/logs/scr_cpu_output.txt -e ~/seq2seq_structure_aware/logs/scr_cpu_errors.txt -pe smp 1 -l mem_free=32G,act_mem_free=32G,h_data=32G ~/seq2seq_structure_aware/scripts/cluster_script.sh
