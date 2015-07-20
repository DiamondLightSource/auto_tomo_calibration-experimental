module load python/ana
cd /dls/tmp/jjl36382/logs
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/selector.py -x $1 -y $2 -z $3 -r $4 $5 $6 $7
