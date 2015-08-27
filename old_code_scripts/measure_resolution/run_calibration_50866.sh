module load global/cluster


#50867 is a nice data set
#50866 is also very good - might use this in presentation
#others dont know
##########################################
datapath="/dls/tmp/tomas_aidukas/scans_july16/cropped/50867/image_%05i.tif"
folder="/dls/tmp/jjl36382/data_50867"
##########################################

mkdir $folder
mkdir $folder/logs
mkdir $folder/results
mkdir $folder/sorted
mkdir $folder/plots
mkdir $folder/label
mkdir $folder/data
mkdir $folder/logs
mkdir $folder/sinograms

cd $folder/logs

homepath="${HOME}/auto_tomo_calibration-experimental/Analysis_Cluster/measure_resolution"
#datapath="/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif"
#datapath="/dls/tmp/jjl36382/resolution1/reconstruction/testdata1/image_%05i.tif"
#datapath="/dls/tmp/jjl36382/resolution2/reconstruction/testdata/image_%05i.tif"

sinopath=$folder"/sinograms/sino_%05i.tiff"
resultspath=$folder"/results"
spherepath=$folder"/spheres"
labelpath=$folder"/label/"
plotspath=$folder"/plots"
segmented=$folder"/sphere"

# ENTER START FILE NUMBER + 1 AND END NUMBER +1
start=1
stop=2128
step=1
# Two points are considered to be in contact if their radii sum
# equals the distance between the points within the tolerance value
tolerance=5

# Create simulation data
#holder="-N merge"
#qsub $holder -pe smp 3 -j y -t $start-$stop:1 -tc 30 $homepath/merge.sh $sinopath

# Detect circles ------------------------------------------------------------------------------------------------------
#holder="-hold_jid folder50869 -N folder50871"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 30 $homepath/detector.sh $datapath $resultspath/out%05i.dat $labelpath

# Analyse areas ------------------------------------------------------------------------------------------------------
#holder="-hold_jid detect -N analyse"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath
$homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath

# Get resolution ------------------------------------------------------------------------------------------------------
#holder="-hold_jid merge -N resolution"
#qsub $holder -pe smp 2 -j y -t 1 -tc 10 $homepath/find_contacts.sh $plotspath $resultspath/ $datapath $tolerance
$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $tolerance $start $stop
