module load global/cluster

resultspath=$1
homepath=$2
spherepath=$3
sigma=$4

nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq $nb_spheres`;
do
	# create a directory to store the files
	mkdir $spherepath/sphere_hessian$i
	
	# copy the hessian filter to the directory
	cp $homepath/itk_hes_rca $spherepath/sphere_hessian$i/
	
	prev=$(($i-1))
	holder="-hold_jid job0$prev -N job0$i"
	
	if [ $i -eq 1 ]; then
		holder="-N job01"
	fi
	
	qsub -pe smp 2 -j y -t $i -tc 20 $homepath/hessian_filter.sh $spherepath/sphere_hessian$i $spherepath/sphere$i.mhd $sigma
done
