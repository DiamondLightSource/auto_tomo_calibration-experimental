module load python/ana
cd /dls/tmp/jjl36382/logs
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a 1 -b 360 -c 10 /dls/tmp/jjl36382/complicated_data/spheres/radii6/radii%03i.npy 6 /dls/tmp/jjl36382/complicated_data/analysis /dls/tmp/jjl36382/complicated_data/spheres/radii6/contact%03i.npy
