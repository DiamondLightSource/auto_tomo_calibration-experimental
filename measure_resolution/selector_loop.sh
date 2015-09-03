module load global/cluster

spherepath=$1
datapath=$2
resultspath=$3
homepath=$4

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	X=`awk "NR=="$i $resultspath/centresX.txt`
	Y=`awk "NR=="$i $resultspath/centresY.txt`
	Z=`awk "NR=="$i $resultspath/centresZ.txt`
	R=`awk "NR=="$i $resultspath/radii.txt`
	
	qsub -N job3 -pe smp 2 -j y -t $i -tc 10 $homepath/selector.sh $X $Y $Z $R $spherepath/sphere$i $datapath $homepath $resultspath
done
