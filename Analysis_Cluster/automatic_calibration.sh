# Detector -----------------------------------------------------------------
module load global/cluster
module load python/ana

cd /dls/tmp/jjl36382/logs

read -p "enter path together with the file name and format > " path
read -p "enter starting file number > " start
read -p "enter final file number > " final
read -p "enter step > " step

nb_spheres=`cat /dls/tmp/jjl36382/results/nb_spheres.txt`

# Detect circles ------------------------------------------------------------------------------------------------------
echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/detector.py $@" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/test_run.sh
echo qsub -pe smp 2 -j y -t $start:$final:$step -tc 15 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh $path"recon_%05i.tif" /dls/tmp/jjl36382/results/out%05i.dat

# CHANGE THE RANGE IN ANALYSE.PY. THE RANGE CORRESPONDS TO THE NUMBER OF FILES IN
# Analyse areas ------------------------------------------------------------------------------------------------------
echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/analyse.py -a $start -b $final -c $step $path" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/test_run.sh
#qsub -pe smp 2 -j y -t 1 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh 

# Select areas ------------------------------------------------------------------------------------------------------
for i in `seq $nb_spheres`;
do
	X=`awk "NR=="$i /dls/tmp/jjl36382/results/centresX.txt`
	Y=`awk "NR=="$i /dls/tmp/jjl36382/results/centresY.txt`
	Z=`awk "NR=="$i /dls/tmp/jjl36382/results/centresZ.txt`
	R=`awk "NR=="$i /dls/tmp/jjl36382/results/radii_max.txt`
	
	echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/selector.py -x $X -y $Y -z $Z -r $R $@" >  ~/auto_tomo_calibration-experimental/Analysis_Cluster/test_run.sh 
	#echo qsub -pe smp 2 -j y -t $i ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere%02i.npy $path"recon_%05i.tif"
done

# Filter spheres ------------------------------------------------------------------------------------------------------
for i in `seq $nb_spheres`;
do	
	echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/filter_sphere.py $@" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/test_run.sh
	echo qsub -pe smp 2 -j y -t 2 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere0$i.npy /dls/tmp/jjl36382/spheres/sphere_filt$i.npy
done

# Get radii ------------------------------------------------------------------------------------------------------
read -p "enter starting radii (use 1 if not sure) > " start
read -p "enter final file number (use 360 if not sure) > " final
read -p "enter step > " step

for i in `seq $nb_spheres`;
do
	R=`awk "NR=="$i /dls/tmp/jjl36382/results/radii_max.txt`
	
	echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/get_radii.py -x $R -y $R -z $R $@" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/test_run.sh
	#echo qsub -pe smp 2 -j y -t $start-$final:$step -tc 15 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere_median02.npy /dls/tmp/jjl36382/radii/radii%03i.npy
done 


# Plot radii ------------------------------------------------------------------------------------------------------
echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/test_run.sh
#qsub -pe smp 2 -j y -t 1 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh
