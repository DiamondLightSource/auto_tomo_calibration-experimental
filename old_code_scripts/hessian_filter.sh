module load python/ana
cd $1

# run the c script in each of the folders passed as $1
$1/itk_hes_rca $2 $3

