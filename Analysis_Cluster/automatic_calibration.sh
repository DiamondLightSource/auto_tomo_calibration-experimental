# Detector -----------------------------------------------------------------
module load global/cluster
#clear the logs directory for the new data set
rm -r /dls/tmp/jjl36382/logs
mkdir /dls/tmp/jjl36382/logs
cd /dls/tmp/jjl36382/logs


#read -p "enter path together with the file name and format > " path
#read -p "enter starting file number > " start
#read -p "enter final file number > " final
#read -p "enter step > " step

path="/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif"
start=322
stop=1558
step=10

# Detect circles ------------------------------------------------------------------------------------------------------
holder="-N job1"
echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/detector.py \$@" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
qsub $holder -pe smp 2 -j y -t $start:$stop:$step -tc 20 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh $path /dls/tmp/jjl36382/results/out%05i.dat


# CHANGE THE RANGE IN ANALYSE.PY. THE RANGE CORRESPONDS TO THE NUMBER OF FILES IN
# Analyse areas ------------------------------------------------------------------------------------------------------
#holder="-hold_jid job1 -N job2"
#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/analyse.py -a $start -b $stop -c $step /dls/tmp/jjl36382/results/out%05i.dat" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#qsub $holder -pe smp 2 -j y -t 1 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 


#nb_spheres=`cat /dls/tmp/jjl36382/results/nb_spheres.txt`
# Select areas ------------------------------------------------------------------------------------------------------
#holder="-hold_jid job1 job2 -N job3"
for i in `$nb_spheres`;
do
	#X=`awk "NR=="$i /dls/tmp/jjl36382/results/centresX.txt`
	#Y=`awk "NR=="$i /dls/tmp/jjl36382/results/centresY.txt`
	#Z=`awk "NR=="$i /dls/tmp/jjl36382/results/centresZ.txt`
	#R=`awk "NR=="$i /dls/tmp/jjl36382/results/radii_max.txt`
	
	#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/selector.py -x $X -y $Y -z $Z -r $R \$@" >>  ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 
	#qsub $holder -pe smp 2 -j y -t $i ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh /dls/tmp/jjl36382/spheres/sphere%02i.npy $path
done


# Filter spheres ------------------------------------------------------------------------------------------------------
#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/filter_sphere.py \$@" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#qsub -pe smp 2 -j y -t 2 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh /dls/tmp/jjl36382/spheres/sphere%02i.npy /dls/tmp/jjl36382/spheres/sphere_5gaus%02i.npy



# Get radii ------------------------------------------------------------------------------------------------------
# Change the range
#read -p "enter starting radii (use 1 if not sure) > " start
#read -p "enter final file number (use 360 if not sure) > " final
#read -p "enter step > " step

#startang=0
#stopang=360
#stepang=10

for i in `$nb_spheres`;
do
	#R=`awk "NR=="$i /dls/tmp/jjl36382/results/radii_max.txt`
	#next=$(($i+1))
	#holder="-hold_jid job3 job4$i -N job4$next"
	
	#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/get_radii.py -x $R -y $R -z $R \$@" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#rm -r /dls/tmp/jjl36382/radii$i
	#mkdir /dls/tmp/jjl36382/radii$i
	#qsub $holder -pe smp 2 -j y -t $startang-$stopang:$stepang -tc 20 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh /dls/tmp/jjl36382/spheres/sphere0$i.npy /dls/tmp/jjl36382/radii$i/radii%03i.npy
done 


# Plot radii ------------------------------------------------------------------------------------------------------
for i in `seq 2`;
do
	#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a $start -b $final -c $step /dls/tmp/jjl36382/radii$i/radii%03i.npy $i" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#qsub -pe smp 2 -j y -t $i ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh /dls/tmp/jjl36382/radii$i/radii%03i.npy
	#~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
done
