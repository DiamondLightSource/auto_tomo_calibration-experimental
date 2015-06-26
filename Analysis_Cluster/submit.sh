module load global/cluster
cd /dls/science/groups/das/norine/logs
qsub -pe smp 2 -j y -t 1-2159:10 -tc 10 /dls/science/groups/das/norine/run.sh /dls/science/groups/das/norine/data/recon_%05i.tif /dls/science/groups/das/norine/results/out%05i.dat
