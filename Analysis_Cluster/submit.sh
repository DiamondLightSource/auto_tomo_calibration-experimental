module load global/cluster
cd /dls/science/groups/das/norine/logs
qsub -pe smp 2 -j y -t 1-360:10 -tc 10 /dls/science/groups/das/norine/run.sh /dls/science/groups/das/norine/spheres/sphere_f03.npy /dls/science/groups/das/norine/radii/radii%03i.npy 
