module load python/ana
cd /dls/tmp/jjl36382/logs

# run the c script in each of the folders passed as $1
$1/itk_hes_rca $2 $3

