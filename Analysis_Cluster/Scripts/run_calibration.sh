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


# Select areas ------------------------------------------------------------------------------------------------------
# Takes in original raw data and segments out the spheres given their centres
# Stores the images in an mhd format, which is then used for processing
holder="-hold_jid job2 -N job3"
#qsub $holder -pe smp 2 -j y -t 1 -tc 10 $homepath/selector_loop.sh $spherepath $datapath $start $resultspath $homepath



# Detects the edges ussing a 3D Hessian filter
# Takes in mhd files, returns raw data files
# Sigma 4 works well, though the edges seem a bit blurred
sigma=1
holder="-hold_jid selector.sh -N job4"
#qsub $holder -pe smp 2 -j y -t 1 -tc 20 $homepath/hessian_filter_loop.sh $resultspath $homepath $spherepath $sigma


# Get radii ------------------------------------------------------------------------------------------------------
# Load up the mhd images and does the usual processing of tracing the line
# from the centre and locating the radius
holder="-hold_jid itk_hes_rca -N job5"
qsub $holder -pe smp 2 -j y -t 1 $homepath/get_radii_loop.sh $resultspath $homepath $spherepath $sigma


# Plot radii ------------------------------------------------------------------------------------------------------
holder="-hold_jid job5 -N job6"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/plot_radii_loop.sh $startang $stopang $stepang $resultspath $spherepath


nb_spheres=`cat $resultspath/nb_spheres.txt`
#for i in `seq $nb_spheres`;
for i in `seq 11`;
do
	echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py -a 1 -b 360 -c 10 $spherepath/radii$i/radii%03i.npy $i $analysispath $spherepath/radii$i/contact%03i.npy" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
	~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 
done

echo "module load python/ana" > ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
echo "cd /dls/tmp/jjl36382/logs" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
echo "python ~/auto_tomo_calibration-experimental/Analysis_Cluster/calculate_resolution.py $analysispath" >> ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh
#qsub -pe smp 2 -j y -t 1 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run_auto_calib.sh 
