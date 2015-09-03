module load global/cluster

####################### USER INPUT ###########################
# SPECIFY DATA PATH AND OUTPUT PATH 
datapath="/dls/tmp/tomas_aidukas/new_recon_steel/50880/recon_noringsup/r_2015_0825_200209_images/image_%05i.tif"
outputpath="/dls/tmp/jjl36382/50880"

# ENTER STARTING FILE NAME NUMBER +1 AND END NUMBER +1
# THE IMAGE NAME FORMAT IS IMAGE_#####.TIF
start=461
stop=1731
# STEP OF 1 GIVES THE MOST ACCURATE RESULTS, THOUGH EVEN WITH 10
# SPHERES ARE SOMEWHAT PROPERLY SEGMENTED - NEEDS MORE TESTING
step=1

# MEDIAN FILTER WINDOW SIZE
window_size=0

# TOLERANCE USED IN DETERMINING WHETHER SPHERES ARE TOUCHING OR NOT
# IF SPHERES ARE VERY CLOSE, BUT DO NOT TOUCH THIS WILL STILL FIND THEM
# ONLY USEFUL IF CENTRES WERE A BIT OFF TO GET THE REGION
tolerance=20
##############################################################


#############################################################
# CREATE DIRECTORIES 
mkdir $outputpath
mkdir $outputpath/logs
mkdir $outputpath/results
mkdir $outputpath/plots
mkdir $outputpath/label
mkdir $outputpath/logs
mkdir $outputpath/spheres

# STORE THE LOGS HERE
cd $outputpath/logs
#############################################################



#############################################################
# VARIABLES TO STORE PATH NAMES USED IN PYTHON SCRIPTS
# HOMEPATH - DIRECTORY WHICH CONTAINS ALL THE PYTHON/SHELL SCRIPTS
homepath="${HOME}/auto_tomo_calibration-experimental/measure_resolution"

# RESULTS IS THE FOLDER CONTAINING CIRCLE PERIMETERS AND THEIR CENTRES
# IF SEGMENTATION FAILS, THE TEXT FILES CAN HAVE CENTROIDS WRITTEN MANUALLY
# AND THE SLICE EXTRACTION WILL STILL BE DONE
resultspath=$outputpath"/results"

# LABEL STORES THE PNG FILES OF IMAGES WITH CIRCLE PLOTTED (FOR TESTING)
labelpath=$outputpath"/label"

# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
plotspath=$outputpath"/plots"

# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
spherepath=$outputpath"/spheres"

#############################################################



############## DETECT CIRCLES ###############################
holder="-N job1"
#qsub $holder -pe smp 2 -j y -t $start-$stop:$step -tc 30 $homepath/detector.sh $datapath $resultspath/out%05i.dat $labelpath/ $homepath


### USE CIRCLE INFORMATION TO GET INFO ABOUT THE SPHERES ####
holder="-hold_jid job1 -N job2"
#qsub $holder -pe smp 2 -j y -t 1 $homepath/sort_centres.sh $start $stop $step $resultspath/out%05i.dat $resultspath $homepath


### CAN BE USED FOR SINGLE SPHERE SEGMENTATION IF NEEDED ###
### NOT USABLE ANYMORE, SINCE THERE ARE BETTER SOLUTIONS ###
### SEGMENT THE SPHERE APPLY EDM AND STORE CENTROIDS ###########
holder="-hold_jid job2 -N job3"
#qsub $holder -pe smp 2 -j y -t 1 -tc 10 $homepath/selector_loop.sh $spherepath $datapath $resultspath $homepath 


datapath="/dls/tmp/tomas_aidukas/new_recon_steel/50873/recon_noringsup/r_2015_0825_200208_images/image_%05i.tif"
outputpath="/dls/tmp/jjl36382/50873"
# RESULTS STORE THE DATA OF THE CIRCLE PERIMETERS AND THEIR CENTRES
resultspath=$outputpath"/results"
# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
plotspath=$outputpath"/plots"

# FIND CONTACT POINTS AND MEASURE RESOLUTION AROUND THEM
$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $start $stop $homepath $tolerance $window_size


datapath="/dls/tmp/tomas_aidukas/new_recon_steel/50880/recon_noringsup/r_2015_0825_200209_images/image_%05i.tif"
outputpath="/dls/tmp/jjl36382/50880"
# RESULTS STORE THE DATA OF THE CIRCLE PERIMETERS AND THEIR CENTRES
resultspath=$outputpath"/results"
# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
plotspath=$outputpath"/plots"

# FIND CONTACT POINTS AND MEASURE RESOLUTION AROUND THEM 
$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $start $stop $homepath $tolerance $window_size



datapath="/dls/tmp/tomas_aidukas/new_recon_steel/50867/recon_noringsup/r_2015_0825_200207_images/image_%05i.tif"
outputpath="/dls/tmp/jjl36382/50867"
# RESULTS STORE THE DATA OF THE CIRCLE PERIMETERS AND THEIR CENTRES
resultspath=$outputpath"/results"
# PLOTS CONTAIN THE REGIONS OF POINTS OF CONTACT
plotspath=$outputpath"/plots"

# FIND CONTACT POINTS AND MEASURE RESOLUTION AROUND THEM
$homepath/find_contacts.sh $plotspath $resultspath/ $datapath $start $stop $homepath $tolerance $window_size



