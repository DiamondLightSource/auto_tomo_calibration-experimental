# Task IDs: "start-stop:step"

# You can find below all the necessary commands to run my program.
# To use the cluster:
# Run the single file "submit.sh" after having modified the names of the files you want to run in this same file and in "run.sh".
# The file called "monitor.sh" is used to display the job array running on the cluster.

# Detector -----------------------------------------------------------------
module load global/cluster
cd /dls/science/groups/das/norine/logs
qsub -pe smp 2 -j y -t 1-2159:10 -tc 10 /dls/science/groups/das/norine/run.sh /dls/science/groups/das/norine/data/recon_%05i.tif /dls/science/groups/das/norine/results/out%05i.dat
# In run.sh
module load python/ana
python /dls/science/groups/das/norine/detector.py $@

# Analyse: get centres and radii -------------------------------------------
module load python/ana
cd /dls/science/groups/das/norine/logs # not sure about the relevance of this
python /dls/science/groups/das/norine/analyse.py

# Area selector ------------------------------------------------------------
module load global/cluster
cd /dls/science/groups/das/norine/logs
qsub -pe smp 2 -j y -t 1-7 -tc 10 /dls/science/groups/das/norine/run.sh /dls/science/groups/das/norine/spheres/sphere%02i.npy
# In run.sh
module load python/ana
python /dls/science/groups/das/norine/selector.py -x 456 -y 456 -z 456 -r 380 $@

# Filter whole spheres (numbers 3 to 6 in this data) -----------------------
module load global/cluster
cd /dls/science/groups/das/norine/logs
qsub -pe smp 2 -j y -t 3-6 -tc 10 /dls/science/groups/das/norine/run.sh /dls/science/groups/das/norine/spheres/sphere%02i.npy /dls/science/groups/das/norine/spheres/sphere_f%02i.npy
# In run.sh
module load python/ana
python /dls/science/groups/das/norine/filter_sphere.py $@

# Get radii according to angles --------------------------------------------
module load global/cluster
cd /dls/science/groups/das/norine/logs
qsub -pe smp 2 -j y -t 1-360:10 -tc 10 /dls/science/groups/das/norine/run.sh /dls/science/groups/das/norine/spheres/sphere_f%02i.npy /dls/science/groups/das/norine/radii/radii%03i.npy
# In run.sh
module load python/ana
python /dls/science/groups/das/norine/radii.py -x 456 -y 456 -z 456 $@

# Plot radii ----------------------------------------------------------------
# Not sure if matplotlib works without Dawn - otherwise run with Dawn
module load python/ana
cd /dls/science/groups/das/norine/logs # not sure about the relevance of this
python /dls/science/groups/das/norine/plot_radii.py