module load global/cluster

####################### USER INPUT ###########################
############# SPECIFY DATA PATH AND OUTPUT PATH ##############
#datapath="/dls/tmp/tomas_aidukas/scans_july16/cropped/50873/image_%05i.tif"
datapath="/dls/tmp/tomas_aidukas/new_recon_steel/50880/recon_noringsup/r_2015_0825_200209_images/image_%05i.tif"
outputpath="/dls/tmp/jjl36382/50880"

# ENTER STARTING FILE NAME NUMBER +1 AND END NUMBER +1
# THE IMAGE NAME FORMAT IS IMAGE_#####.TIF
start=461
stop=1731
step=1
##############################################################


############ CREATE DIRECTORIES ##############################
mkdir $outputpath
mkdir $outputpath/logs
mkdir $outputpath/results
mkdir $outputpath/sorted
mkdir $outputpath/plots
mkdir $outputpath/label
mkdir $outputpath/data
mkdir $outputpath/logs
mkdir $outputpath/sinograms
#############################################################


############ STORE THE LOGS HERE ############################
cd $outputpath/logs
#############################################################


#### VARIABLES TO STORE PATH NAMES USED IN PYTHON SCRIPTS ###
# HOMEPATH - DIRECTORY WHICH CONTAINS ALL THE PYTHON SCRIPTS
homepath="${HOME}/auto_tomo_calibration-experimental/measure_resolution"

# RESULTS STORE THE DATA OF THE CIRCLE PERIMETERS AND THEIR CENTRES
resultspath=$outputpath"/results"

# LABEL STORES THE PNG FILES OF IMAGES WITH CIRCLE PLOTTED (FOR TESTING)
labelpath=$outputpath"/label"

# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
plotspath=$outputpath"/plots"
#############################################################



############## DETECT CIRCLES ###############################
holder="-hold_jid f50867 -N fold"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 30 $homepath/detector.sh $datapath $resultspath/out%05i.dat $labelpath/ $homepath


### USE CIRCLE INFORMATION TO GET INFO ABOUT THE SPHERES ####
holder="-hold_jid detect -N analyse"
#$homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath $homepath


### FIND CONTACT POINTS AND MEASURE RESOLUTION AROUND THEM ###
$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $start $stop $homepath
