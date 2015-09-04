# What it does?

The program uses images of a tomographic reconstruction to compute modulation transfer function and segment regions of interest around the contact point between two spheres. Each image slice must have a file name "image_00000.tif". It must satisfy the "image_%05i.tif" file naming convention as this is used in Python.

# Structure

There is one main shells script called "calculate_resolution.sh". This shell script send other shell scripts to the computer cluster using qsub. Each shells script  runs a Python script.

Inside "calculate_resolution.sh" image path must be specified as well as the output folder, where all the results will be stored. There are also some variables such as "window_size" and "tolerance". Window size specified the size of the median filter kernel used in filtering the final region of interest. 0 will apply no filtering. Tolerance specifies the maximum error tolerance used in determining whether two spheres are touching or not. "if |(radii sum) - (centre to centre distance)| < tolerance then they are touching". If there are many spheres, a value that is two small can find spheres with a very narrow gap in between them. This also helps to prevent artefacts that were reconstructed as spheres, such as part of a sphere that is cropped and only part of it is visible.

The begining, end and step size has to be specified as well to tell which part of the stack contains the spheres. For best results every slice should be used, but step size of 10 is able to locate spheres quite well, although it might not be too accurate.

Also location containing the shell/python scripts has to be specified

# How to run it

Once variables are set up in the script it can be run by simply executing. Though it has two parts. First one computers the sphere parameters such as radius and their centre postion. Secondly, the ROI is segmented and analysed.

However, qsub does not have a display, which is needed for MTF plotting. Therefore, hold_jid was unavailable and it "find_contacts" command was simply commented out while sphere segmentation was being done.

1. Comment "find_contacts.sh" 
2. Uncomment the qsub commands (two of them) and run. Circle detector will be used and when finished the spheres will be constructed.
3. Comment the qsubs, and run "find_contacts.sh"


# Output

Labels - contains .png images of every slice  with the detected centre and edges drawn on them. Was used for visual testing, whether the code ran successfully or not. Only plots every tenth slice, but could be turned off completely if it is useless (suggestion would be to always keep an eye on how segmentation is going)

Results - contains the .dat file, which store radius, edge and centre positions for every circle in a single slice. Results also stores the centres.txt and radius.txt values, which contain the centres and radii of the spheres. In case of segmentation failure, the values can be entered manually, if they were known by some other methods.

Plots - contains the MTF plots for every sphere pair. Also contains .txt and .npy, which store spacings between the spheres and the MTF values for each point, in case other software was used to analyse the data. Lines profiles close to the cointact point are also saved as well as image containing the region of interest. This allows inspection of the regions using other software, such as ImageJ.

If any issues arise, even trivial ones, I might know what is causing them, therefore, I would be much quicker in locating and fixing them, than solving the problem on your own. I spent ages on the code and know every detail of it. Also, if other information is required or changes are needed to be made to the code please do not hesitate to contact - it would be a pleasure to help.

Thank You.

Tomas


