module load global/cluster
cd /dls/tmp/jjl36382/logs
qsub -pe smp 2 -j y -t 1-2159:10 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif /dls/tmp/jjl36382/results/out%05i.dat

