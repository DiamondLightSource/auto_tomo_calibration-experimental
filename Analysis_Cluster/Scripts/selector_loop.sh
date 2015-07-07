module load global/cluster
cd /dls/tmp/jjl36382/logs

spherepath=$1
datapath=$2
start=$3
resultspath=$4
homepath=$5

nb_spheres=`cat /dls/tmp/jjl36382/results/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	X=`awk "NR=="$i $resultspath/centresX.txt`
	Y=`awk "NR=="$i $resultspath/centresY.txt`
	Z=`awk "NR=="$i $resultspath/centresZ.txt`
	R=`awk "NR=="$i $resultspath/radii_max.txt`
	
	qsub -pe smp 2 -j y -t $i -tc 20 $homepath/selector.sh $X $Y $Z $R $spherepath $datapath $start
done
