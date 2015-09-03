module load global/cluster

####################### USER INPUT ###########################
############# SPECIFY DATA PATH AND OUTPUT PATH ##############
#datapath="/dls/tmp/tomas_aidukas/scans_july16/cropped/50873/image_%05i.tif"
datapath="/dls/tmp/tomas_aidukas/new_recon_steel/50873/recon_noringsup/r_2015_0825_200208_images/image_%05i.tif"
outputpath="/dls/tmp/jjl36382/50873"

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
mkdir $outputpath/finalresults
mkdir $outputpath/plots
mkdir $outputpath/label
mkdir $outputpath/hessianedges
mkdir $outputpath/logs
mkdir $outputpath/spheres
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

# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
spherepath=$outputpath"/spheres"

# CONTAINS THE DATA OF SPHERES FRO HOUGH EDGES
hessianedges=$outputpath"/hessianedges"

# STORE THE FINAL RADIUS/CENTRE PARAMETERS HERE
finalpath=$outputpath"/finalresults"
#############################################################



############## DETECT CIRCLES ###############################
holder="-hold_jid f50867 -N fold"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 30 $homepath/detector.sh $datapath $resultspath/out%05i.dat $labelpath/ $homepath


### USE CIRCLE INFORMATION TO GET INFO ABOUT THE SPHERES ####
holder="-hold_jid detect -N job2"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath $homepath


### SEGMENT THE SPHERE AND STORE INTO AND MHD FILE ###########
holder="-hold_jid job2 -N job3"
#qsub $holder -pe smp 2 -j y -t 1 -tc 10 $homepath/selector_loop.sh $spherepath $datapath $resultspath $homepath

#sleep 10

### DETECT EDGES USING A HESSIAN AFFINE FILTER ###############
sigma=1
holder="-hold_jid job3 -N job4"
#qsub $holder -pe smp 2 -j y -t 1 -tc 20 $homepath/hessian_filter_loop.sh $resultspath $homepath $spherepath $sigma

#sleep 10

### CONVERT THE MHD EDGE IMAGE INTO TIF FILES ################
holder="-hold_jid job4 -N job5"
qsub $holder -pe smp 2 -j y -t 1 -tc 20 $homepath/mhd_to_tif_loop.sh $resultspath $homepath $spherepath

#sleep 10

### APPLY HOUGH TRANSFORM ON THE EDGE IMAGES #################
holder="-hold_jid job5 -N job6"
#qsub $holder -pe smp 2 -j y -t 1-1200 -tc 30 $homepath/detect_edges.sh $resultspath $hessianedges $homepath $spherepath


### ANALYSE DATA FROM HOUGH TO OBTAIN THE CENTROIDS ##########
holder="-hold_jid job6 -N job7"
mkdir $finalpath/sphere1
mkdir $finalpath/sphere2
#qsub $holder -pe smp 2 -j y -t 1 $homepath/sort_sphere_edges.sh 1 1200 1 $hessianedges/sphere2/slice%i.dat $finalpath/sphere2 $homepath
#$homepath/sort_sphere_edges.sh 1 1200 1 $hessianedges/sphere2/slice%i.dat $finalpath/sphere2 $homepath


# Make sphere parameter corrections
# obtained from fitting a sphere to the Hessian edges
holder="-hold_jid job7 -N job8"
#qsub $holder -pe smp 2 -j y -t 1 -tc 20 $homepath/sphere_edges_loop.sh $homepath $spherepath $resultspath

### FIND CONTACT POINTS AND MEASURE RESOLUTION AROUND THEM ###
#$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $start $stop $homepath
