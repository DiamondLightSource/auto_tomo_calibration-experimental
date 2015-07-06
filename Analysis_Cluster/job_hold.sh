#!/bin/bash

# jobhold.sh
# JB 8/2010
# usage: 
#       jobhold.sh qsub <my_command>
#       jobhold.sh q.sh <my_command>
# For commands that are submitted to cluster, hold until jobs have completed

shopt -s expand_aliases

sleep_time=30 # seconds; don't make this too short! don't want to tax system with excessive qstat calls

me=`whoami`
alias myqstat='qstat | grep $me'
stdout=`$@` # call the command and capture the stdout
id=`echo $stdout | awk -F' ' '{print $3}'` # get the jobid
status=`myqstat | grep $id` # check to see if job is running
while [ -n "$status" ] # while $status is not empty
	do
		sleep $sleep_time
		status=`myqstat | grep $id`
	done
