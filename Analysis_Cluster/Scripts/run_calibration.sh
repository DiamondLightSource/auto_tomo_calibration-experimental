module load global/cluster
cd /dls/tmp/jjl36382/complicated_data/logs

#read -p "enter path together with the file name and format > " datapath
#read -p "enter starting file number > " start
#read -p "enter final file number > " stop
#read -p "enter step > " step

#homepath="${HOME}/auto_tomo_calibration-experimental/Analysis_Cluster/Scripts"
#datapath="/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif"
#resultspath="/dls/tmp/jjl36382/results"
#spherepath="/dls/tmp/jjl36382/spheres"
#start=1
#stop=2160
#step=10

homepath="${HOME}/auto_tomo_calibration-experimental/Analysis_Cluster/Scripts"
datapath="/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif"
resultspath="/dls/tmp/jjl36382/complicated_data/results"
spherepath="/dls/tmp/jjl36382/complicated_data/spheres"
analysispath="/dls/tmp/jjl36382/complicated_data/analysis"

# ENTER START FILE NUMBER + 1 AND END NUMBER +1
start=322
stop=1558
step=10


# Detect circles ------------------------------------------------------------------------------------------------------
holder="-N job1"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 20 $homepath/detector.sh $datapath $resultspath/out%05i.dat


# Analyse areas ------------------------------------------------------------------------------------------------------
holder="-hold_jid job1 -N job2"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/analyse.sh $start $stop $step $resultspath/out%05i.dat $resultspath
#$homepath/analyse.sh $start $stop $step $resultspath/out%05i.dat $resultspath


# SHIFT Z COORDS IF NAME STARTS NOT FROM 0 (???)
# Select areas ------------------------------------------------------------------------------------------------------
holder="-hold_jid job2 -N job3"
#qsub -pe smp 2 -j y -t 1 -tc 10 $homepath/selector_loop.sh $spherepath/sphere%i.npy $datapath $start $resultspath $homepath


startang=1
stopang=360
stepang=10
# Filter spheres before analysis
qsub -pe smp 2 -j y -t 1 $homepath/filter_sphere_loop.sh $startang $stopang $stepang $resultspath $homepath $spherepath

# Get radii ------------------------------------------------------------------------------------------------------
#read -p "enter starting radii (use 1 if not sure) > " startang
#read -p "enter final file number (use 360 if not sure) > " finalang
#read -p "enter step > " stepang


holder="-hold_jid job3 -N job4"
#qsub -pe smp 2 -j y -t 1 $homepath/get_radii_loop.sh $startang $stopang $stepang $resultspath $homepath $spherepath



# Plot radii ------------------------------------------------------------------------------------------------------
holder="-hold_jid job4 -N job5"
#qsub -pe smp 2 -j y -t 1 $homepath/plot_radii_loop.sh $startang $stopang $stepang $resultspath $spherepath


nb_spheres=`cat $resultspath/nb_spheres.txt`
for i in `seq 2`;
do
	echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a $startang -b $stopang -c $stepang $spherepath/radii$i/radii%03i.npy $i $analysispath" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#	~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 
done
