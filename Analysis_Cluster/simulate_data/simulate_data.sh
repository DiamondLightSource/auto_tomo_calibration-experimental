module load global/cluster
qsub -pe smp 4 -j y -t 1 ./generator.sh
