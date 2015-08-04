module load global/cluster
cd /dls/tmp/jjl36382/resolution/logs

tolerance=$1
resultspath=$2
homepath=$3

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	centres=`awk "NR=="$i $resultspath/centres.txt`
	radii=`awk "NR=="$i $resultspath/radii.txt`
	
	qsub -pe smp 2 -j y -t $i -tc 10 $homepath/find_contacts.sh $centres $radii $tolerance $resultspath
done
