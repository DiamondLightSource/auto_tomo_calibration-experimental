module load global/cluster
cd ~/auto_tomo_calibration-experimental/Cluster_Data/logs
qsub -pe smp 2 -j y -t 1-2159:10 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh ../../dls/science/groups/das/norine/data/recon_%05i.tif ~/auto_tomo_calibration-experimental/Cluster_Data/results/out%05i.dat
