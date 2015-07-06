module load python/ana
cd /dls/tmp/jjl36382/logs
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/get_radii.py -x $1 -y $1 -z $1 $2 $3
