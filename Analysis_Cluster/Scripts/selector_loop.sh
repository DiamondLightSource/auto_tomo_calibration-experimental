module load global/cluster
cd /dls/tmp/jjl36382/logs

nb_spheres=`cat /dls/tmp/jjl36382/results/nb_spheres.txt`
for i in `$nb_spheres`;
do
	X=`awk "NR=="$i $resultspath/centresX.txt`
	Y=`awk "NR=="$i $resultspath/centresY.txt`
	Z=`awk "NR=="$i $resultspath/centresZ.txt`
	R=`awk "NR=="$i $resultspath/radii_max.txt`
	
	qsub -pe smp 2 -j y -t $i $homepath/selector.sh $X $Y $Z $R $1 $2
	
done
