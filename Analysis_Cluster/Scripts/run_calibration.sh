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
# enter starting folder + 1
start=322
stop=1557
step=10


# Detect circles ------------------------------------------------------------------------------------------------------
holder="-N job1"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 20 $homepath/detector.sh $datapath $resultspath/out%05i.dat


# Analyse areas ------------------------------------------------------------------------------------------------------
holder="-hold_jid job1 -N job2"
#holder="-N job2"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/analyse.sh $start $stop $step $resultspath/out%05i.dat
$homepath/analyse.sh $start $stop $step $resultspath/out%05i.dat


# Select areas ------------------------------------------------------------------------------------------------------
holder="-hold_jid job2 -N job3"
#qsub -pe smp 2 -j y -t 1 $homepath/selector_loop.sh $spherepath/sphere%i.npy $datapath $start $resultspath $homepath


# Get radii ------------------------------------------------------------------------------------------------------
#read -p "enter starting radii (use 1 if not sure) > " startang
#read -p "enter final file number (use 360 if not sure) > " finalang
#read -p "enter step > " stepang

startang=1
stopang=360
stepang=10
holder="-hold_jid job3 -N job4"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/get_radii_loop.sh $startang $stopang $stepang $resultspath $homepath $spherepath/sphere%i.npy



# Plot radii ------------------------------------------------------------------------------------------------------
holder="-hold_jid job4 -N job5"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/plot_radii_loop.sh $startang $stopang $stepang


#echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a $startang -b $stopang -c $stepang /dls/tmp/jjl36382/radii18/radii%03i.npy 18" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 

