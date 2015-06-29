module load python/ana
cd /dls/tmp/jjl36382/logs

#python ~/auto_tomo_calibration-experimental/Analysis_Cluster/detector.py $@
#python ~/auto_tomo_calibration-experimental/Analysis_Cluster/analyse.py
#python ~/auto_tomo_calibration-experimental/Analysis_Cluster/selector.py -x 1105 -y 1002 -z 1470 -r 380 $@
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/filter_sphere.py $@
#python ~/auto_tomo_calibration-experimental/Analysis_Cluster/get_radii.py -x 456 -y 456 -z 456 $@
#python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py
