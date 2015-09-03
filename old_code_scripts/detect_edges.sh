module load python/ana

resultspath=$1 
datapath=$2
homepath=$3
spherepath=$4

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
    mkdir $datapath/sphere$i
    
	python $homepath/detector_edges.py $spherepath/sphere_tif$i $datapath/sphere$i
	
done
