module load global/cluster
cd /dls/tmp/jjl36382/logs

#read -p "enter path together with the file name and format > " datapath
#read -p "enter starting file number > " start
#read -p "enter final file number > " stop
#read -p "enter step > " step

homepath="${HOME}/auto_tomo_calibration-experimental/Analysis_Cluster/Scripts"
datapath="/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif"
resultspath="/dls/tmp/jjl36382/results"
spherepath="/dls/tmp/jjl36382/spheres"
start=322
stop=1558
step=10

# Detect circles ------------------------------------------------------------------------------------------------------
holder="-N job1"
#qsub $holder -pe smp 2 -j y -t $start:$stop:$step -tc 20 $homepath/detector.sh $datapath $resultspath/out%05i.dat


# Analyse areas ------------------------------------------------------------------------------------------------------
holder="-hold_jid job1 -N job2"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/analyse.sh $start $stop $step $resultspath/out%05i.dat
#$homepath/analyse.sh $start $stop $step $resultspath/out%05i.dat


nb_spheres=`cat /dls/tmp/jjl36382/results/nb_spheres.txt`
holder="-hold_jid job1 job2 -N job3"
for i in `seq $nb_spheres`;
do
	X=`awk "NR=="$i $resultspath/centresX.txt`
	Y=`awk "NR=="$i $resultspath/centresY.txt`
	Z=`awk "NR=="$i $resultspath/centresZ.txt`
	R=`awk "NR=="$i $resultspath/radii_max.txt`
	
	#qsub -pe smp 2 -j y -t $i -tc 20 $homepath/selector.sh $X $Y $Z $R $spherepath/sphere%02i.npy $datapath $start
done
#qsub $holder -pe smp 2 -j y -t 1 $homepath/selector_loop.sh $spherepath/sphere%02i.npy $datapath

# Get radii ------------------------------------------------------------------------------------------------------
# Change the range
#read -p "enter starting radii (use 1 if not sure) > " startang
#read -p "enter final file number (use 360 if not sure) > " finalang
#read -p "enter step > " stepang

startang=1
stopang=360
stepang=10
#OUTPUT FORMAT FOR FOLDERS DOES NOT MATCH WITH %02i FORMAT
for i in `seq 1`;
do	
	i=18
	R=`awk "NR=="$i $resultspath/radii_max.txt`
	#next=$(($i+1))
	#holder="-hold_jid job1 job2 job3 job0$i -N job0$next"
	#mkdir /dls/tmp/jjl36382/radii$i
	#qsub -pe smp 2 -j y -t $startang-$stopang:$stepang -tc 20 $homepath/get_radii.sh $R $spherepath/sphere$i.npy /dls/tmp/jjl36382/radii$i/radii%03i.npy
done 


# Plot radii ------------------------------------------------------------------------------------------------------
for i in `seq 1`;
do
	#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a $start -b $final -c $step /dls/tmp/jjl36382/radii$i/radii%03i.npy $i" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	#qsub -pe smp 2 -j y -t $i ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh /dls/tmp/jjl36382/radii$i/radii%03i.npy
	#~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 
	echo "ayy"
done

echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a $startang -b $stopang -c $stepang /dls/tmp/jjl36382/radii18/radii%03i.npy 18" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 

