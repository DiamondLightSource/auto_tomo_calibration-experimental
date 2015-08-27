module load python/ana
cd /dls/tmp/jjl36382/logs
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/analyse.py -a $1 -b $2 -c $3 $4
