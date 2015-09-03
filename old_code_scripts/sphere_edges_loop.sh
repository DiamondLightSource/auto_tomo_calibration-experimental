module load global/cluster

resultspath=$1
homepath=$2
spherepath=$3

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do

	X=`awk "NR=="$i $resultspath/centresX.txt`
	Y=`awk "NR=="$i $resultspath/centresY.txt`
	Z=`awk "NR=="$i $resultspath/centresZ.txt`
	R=`awk "NR=="$i $resultspath/radii.txt`
	
	qsub -hold_jid job4 -N job5 -pe smp 2 -j y -t $i -tc 20 $homepath/sphere_edges.sh $spherepath/sphere_hessian$i $X $Y $Z $R $homepath $spherepath
	
done
