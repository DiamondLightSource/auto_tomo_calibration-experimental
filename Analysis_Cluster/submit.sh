module load global/cluster
cd /dls/tmp/jjl36382/logs

#detector
#qsub -pe smp 2 -j y -t 1:2159 -tc 15 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif /dls/tmp/jjl36382/results2/out%05i.dat

#selector
#qsub -pe smp 2 -j y -t 2 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere%02i.npy

#filter
qsub -pe smp 2 -j y -t 2 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere%02i.npy /dls/tmp/jjl36382/spheres/sphere_gauss%02i.npy

#get radii
#qsub -pe smp 2 -j y -t 1-360:10 -tc 15 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere_gaussian02.npy /dls/tmp/jjl36382/radii/radii%03i.npy

#plot
#qsub -pe smp 2 -j y -t 1 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh
