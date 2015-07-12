module load global/cluster
cd /dls/tmp/jjl36382/logs

startang=$1
stopang=$2
stepang=$3
resultspath=$4
homepath=$5
spherepath=$6

nb_spheres=`cat /dls/tmp/jjl36382/results/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	R=`awk "NR=="$i $resultspath/radii_max.txt`
	mkdir /dls/tmp/jjl36382/radii$i
	prev=$(($i-1))
	holder="-hold_jid job0$prev -N job0$i"
	
	if [ $i -eq 1 ]; then
		holder="-N job01"
	fi
	
	qsub $holder -pe smp 2 -j y -t $startang-$stopang:$stepang -tc 20 $homepath/get_radii.sh $R $spherepath/sphere$i.npy /dls/tmp/jjl36382/radii$i/radii%03i.npy
done
