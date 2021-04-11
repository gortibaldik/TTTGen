rm ~/rotowire/logs/cpu*

qsub -q cpu.q \
     -cwd \
     -o ~/rotowire/logs/cpu_output.txt \
     -e ~/rotowire/logs/cpu_errors.txt \
     -pe smp 1 -l \
     mem_free=32G,act_mem_free=32G,h_data=32G \
     ~/rotowire/scripts/cluster_script.sh
