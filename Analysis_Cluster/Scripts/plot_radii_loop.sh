module load global/cluster
cd /dls/tmp/jjl36382/logs

start=$1
final=$2
step=$3
resultspath=$4
spherepath=$5

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	qsub -pe smp 2 -j y -t $i ~/auto_tomo_calibration-experimental/Analysis_Cluster/Scripts/plot_radii.sh $start $final $step $spherepath/radii$i/radii%03i.npy $i
	
done
