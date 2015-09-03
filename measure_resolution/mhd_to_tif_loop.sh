module load global/cluster

resultspath=$1
homepath=$2
spherepath=$3

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	mkdir $spherepath/sphere_tif$i
	qsub -hold_jid job4 -N job5 -pe smp 2 -j y -t $i -tc 20 $homepath/mhd_to_tif.sh $spherepath $homepath $spherepath/sphere_tif$i
done
