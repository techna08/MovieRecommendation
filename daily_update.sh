#!/usr/bin/env bash

LOGFILE="auto_model_update_logs.out"

while true; do 
    TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
    echo -e "\n[INFO] Auto update run at: $TIMESTAMP" >> $LOGFILE
    echo -e "======================================================\n" >> $LOGFILE
    bash run_auto_model_update.sh >> $LOGFILE

    sleep 86400;
done
