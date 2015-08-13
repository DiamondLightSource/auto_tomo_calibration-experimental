module load global/cluster
cd /dls/tmp/jjl36382/resolution/logs

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

#folder_start = "./huge_contrast/"
#name = folder_start + "data/sino_%05i.tif"
#label_name = folder_start + "label/analytical%i.png"
#results = folder_start + "results/result%i.txt"
#sorted = folder_start + "sorted/"
#plots = folder_start + "plots/"

mkdir /dls/tmp/jjl36382/resolution
mkdir /dls/tmp/jjl36382/resolution/logs
mkdir /dls/tmp/jjl36382/resolution/results
mkdir /dls/tmp/jjl36382/resolution/sorted
mkdir /dls/tmp/jjl36382/resolution/plots
mkdir /dls/tmp/jjl36382/resolution/label
mkdir /dls/tmp/jjl36382/resolution/data
cd /dls/tmp/jjl36382/resolution/logs

homepath="${HOME}/auto_tomo_calibration-experimental/Analysis_Cluster/measure_resolution"
#datapath="/dls/tmp/tomas_aidukas/scans_july16/cropped/50867/image_%05i.tif"
datapath="/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif"
#datapath="/dls/tmp/jjl36382/resolution1/reconstruction/testdata1/image_%05i.tif"
sinopath="/dls/tmp/jjl36382/resolution/sinograms/sino_%05i.tif"
resultspath="/dls/tmp/jjl36382/resolution/results"
spherepath="/dls/tmp/jjl36382/resolution/spheres"
labelpath="/dls/tmp/jjl36382/resolution/label/"
plotspath="/dls/tmp/jjl36382/resolution/plots"
segmented="/dls/tmp/jjl36382/resolution/data/sphere"
#analysispath="/dls/tmp/jjl36382/complicated_data/resolution1/analysis"

# ENTER START FILE NUMBER + 1 AND END NUMBER +1
start=1
stop=2161
step=1
# Two points are considered to be in contact if their radii sum
# equals the distance between the points within the tolerance value
tolerance=5

# Create simulation data
#holder="-N merge"
#qsub $holder -pe smp 3 -j y -t $start-$stop:1 -tc 30 $homepath/merge.sh $sinopath

# Detect circles ------------------------------------------------------------------------------------------------------
holder="-N detect"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 25 $homepath/detector.sh $datapath $resultspath/out%05i.dat $labelpath

# Analyse areas ------------------------------------------------------------------------------------------------------
holder="-hold_jid detect -N analyse"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath
#$homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath

# Get resolution ------------------------------------------------------------------------------------------------------
holder="-hold_jid merge -N resolution"
#qsub $holder -pe smp 2 -j y -t 1 -tc 10 $homepath/find_contacts.sh $plotspath $resultspath/ $datapath $tolerance
$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $tolerance $start $stop
