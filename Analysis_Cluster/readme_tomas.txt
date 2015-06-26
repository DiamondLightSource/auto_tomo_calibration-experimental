# Task IDs: "start-stop:step"

# You can find below all the necessary commands to run my program.
# To use the cluster:
# Run the single file "submit.sh" after having modified the names of the files you want to run in this same file and in "run.sh".
# The file called "monitor.sh" is used to display the job array running on the cluster.


# What does 1-2159:10 -tc 1- mean? Can it be a default value for all data sets?
# Can I output my data to my alocated hard drive?
# Can I use the cluster at any time and from home using ssh? How?
# Detector -----------------------------------------------------------------
module load global/cluster
cd ~/auto_tomo_calibration-experimental/Analysis_Cluster/logs
qsub -pe smp 2 -j y -t 1-2159:10 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh ~/../../dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif ~/auto_tomo_calibration-experimental/Analysis_Cluster/results/out%05i.dat
# In run.sh
module load python/ana
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/detector.py $@


# CHANGE THE RANGE IN ANALYSE.PY. THE RANGE CORRESPONDS TO THE NUMBER OF FILES IN
# Analyse: get centres and radii -------------------------------------------
module load python/ana
cd ~/auto_tomo_calibration-experimental/Analysis_Cluster/logs # not sure about the relevance of this
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/analyse.py


# What does 1-7 -tc 10 means? 
# Area selector ------------------------------------------------------------
module load global/cluster
cd ~/auto_tomo_calibration-experimental/Analysis_Cluster/logs
qsub -pe smp 2 -j y -t 1-7 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh ~/auto_tomo_calibration-experimental/Analysis_Cluster/spheres/sphere%02i.npy
# In run.sh
module load python/ana
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/selector.py -x 456 -y 456 -z 456 -r 380 $@


# What is everything after -t?
# Filter whole spheres (numbers 3 to 6 in this data) -----------------------
module load global/cluster
cd ~/auto_tomo_calibration-experimental/Analysis_Cluster/logs
qsub -pe smp 2 -j y -t 3-6 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh ~/auto_tomo_calibration-experimental/Analysis_Cluster/spheres/sphere%02i.npy ~/auto_tomo_calibration-experimental/Analysis_Cluster/spheres/sphere_f%02i.npy
# In run.sh
module load python/ana
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/filter_sphere.py $@

# What is everything after -t?
# Get radii according to angles --------------------------------------------
module load global/cluster
cd ~/auto_tomo_calibration-experimental/Analysis_Cluster/logs
qsub -pe smp 2 -j y -t 1-360:10 -tc 10 ~/auto_tomo_calibration-experimental/Analysis_Cluster/run.sh ~/auto_tomo_calibration-experimental/Analysis_Cluster/spheres/sphere_f%02i.npy ~/auto_tomo_calibration-experimental/Analysis_Cluster/radii/radii%03i.npy
# In run.sh
module load python/ana
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/radii.py -x 456 -y 456 -z 456 $@

# Plot radii ----------------------------------------------------------------
# Not sure if matplotlib works without Dawn - otherwise run with Dawn
module load python/ana
cd ~/auto_tomo_calibration-experimental/Analysis_Cluster/logs # not sure about the relevance of this
python ~/auto_tomo_calibration-experimental/Analysis_Cluster/plot_radii.py
