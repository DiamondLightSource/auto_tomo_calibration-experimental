module load global/cluster
cd /dls/tmp/jjl36382/logs

resultspath=$1
homepath=$2
spherepath=$3
sigma=$4

nb_spheres=`cat $4/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	R=`awk "NR=="$i $resultspath/radii_max.txt`
	prev=$(($i-1))
	holder="-hold_jid job0$prev -N job0$i"
	
	if [ $i -eq 1 ]; then
		holder="-N job01"
	fi
	
	qsub $holder -pe smp 2 -j y -t $i -tc 20 $homepath/filter_sphere.sh  $spherepath/sphere$i.mhd $spherepath/sphereEdge$i.mhd $sigma
done
