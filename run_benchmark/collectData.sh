#!/bin/bash
# File  : collectData.sh
# Author: Tianyi Liu
# Email : liuty10@gmail.com

usage(){
     echo "Usage:"
     echo "*******************************************************************"
     echo "./collectData.sh GameName RecordFlag RunTime(s) BindCPU AutoOrHuman MultiMode"
     echo "	1) GameNames  : supertuxkart-1, supertuxkart, 0ad, redeclipse, dota2, inmindvr, imhotepvr, nasawebvr, javaeclipse, libreoffice"
     echo "	2) RecordFlag : 0 -- do not record metrics on server."
     echo "			1 -- record"
     echo "	3) RunTime(s) : How long to run the benchmark(seconds). Typically, stk, 0ad, dota2, imhotepvr: 900; redclipse: 600; inmindvr: 240"
     echo "	4) BindCPU    : 0 -- donot bind python thread to specific cpu core"
     echo "			1 -- bind"
     echo "	5) AutoOrHuman: 0 -- AI Bots runs benchmark."
     echo "			1 -- Human runs benchmark"
     echo "	6) MultiMode  : MultiMode means running multiple instances or not."
     echo "                        0 -- Single Instance."
     echo "                        1 -- run 1 game"
     echo "                        2 -- run 2 games..."
     echo "                        3 -- run 3 games..."
     echo "                        4 -- run 4 games..."
     echo "e.g: ./collectData.sh supertuxkart 1 100 0 0 0"
     exit 0
}
if [ $# -ne 6 ]; then
	usage
fi
APP_NAME=$1
RecordFlag=$2
RUNNING_TIME=$3
BIND_CPU=$4
HUMAN=$5
MULTIPLE_MODE=$6

AI_BOTS_DIR=$(dirname `pwd`)
Reso_Width=1920
Reso_Hight=1080
##########################################################
if [ $MULTIPLE_MODE -ne 0 ] ; then
    THEHOSTNAME=`cat /etc/hostname`
    RESULT_DIR=`pwd`/result_client_cgr_nobind_auto_multi_${MULTIPLE_MODE}_${THEHOSTNAME}/$APP_NAME
elif [ $BIND_CPU -eq 0 ] ; then
    if [ $HUMAN -eq 0 ] ; then
        RESULT_DIR=`pwd`/result_client_cgr_nobind_auto/$APP_NAME
    else
        RESULT_DIR=`pwd`/result_client_cgr_nobind_human/$APP_NAME
    fi
else
    if [ $HUMAN -eq 0 ] ; then
        RESULT_DIR=`pwd`/result_client_cgr_bind_auto/$APP_NAME
    else
        RESULT_DIR=`pwd`/result_client_cgr_bind_human/$APP_NAME
    fi
fi

[ -e `dirname $RESULT_DIR` ] || mkdir `dirname $RESULT_DIR`
[ -e $RESULT_DIR ] || mkdir $RESULT_DIR
rm $RESULT_DIR/*

case $APP_NAME in
    supertuxkart | 0ad | redeclipse | dota2 | inmindvr | imhotepvr | nasawebvr | supertuxkart-demo )
        if [ $BIND_CPU -eq 0 ] ; then
            python3 ../training_scripts/AI-play-${APP_NAME}.py $RecordFlag $RUNNING_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN $Reso_Width $Reso_Hight $MULTIPLE_MODE& echo $! > ./client.pid
        else
            taskset 0x1 python3 ../training_scripts/AI-play-${APP_NAME}.py $RecordFlag $RUNNING_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN $Reso_Width $Reso_Hight $MULTIPLE_MODE& echo $! > ./client.pid
        fi
        ;;
    supertuxkart-1 )
        if [ $BIND_CPU -eq 0 ] ; then
            python3 ../training_scripts/AI-play-supertuxkart-1.py $RecordFlag $RUNNING_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN $Reso_Width $Reso_Hight $MULTIPLE_MODE& echo $! > ./client.pid
        else
            taskset 0x1 python3 ../training_scripts/AI-play-supertuxkart-1.py $RecordFlag $RUNNING_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN $Reso_Width $Reso_Hight $MULTIPLE_MODE& echo $! > ./client.pid
        fi
        ;;

    *)
        echo "Application name NOT correct\n"
        exit 0
        ;;
esac

echo -en `ps -C java -o pid=` > ./VncClient.pid
BotPID=`cat ./client.pid`
VncClientPID=`cat ./VncClient.pid`

networkcards=`ifconfig | grep -P '^[^\s]+\s+[A-Z]' | awk '{print $1}'`
echo $networkcards > $RESULT_DIR/networkcards_name.log

COUNT=$RUNNING_TIME # number of records, RUNNING_TIME/2;
echo "Running Time: $RUNNING_TIME seconds"
echo "Count: $COUNT"
echo "running.."

num=$COUNT
while [ $num -gt 0 ]; do
    echo "Count: $num"
    cat /proc/stat >> $RESULT_DIR/proc_stat.log                      		# overall CPU
    cat /proc/meminfo >> $RESULT_DIR/proc_meminfo.log                		# overall Mem
    cat /proc/uptime >> $RESULT_DIR/proc_uptime.log                  		# For PID CPU
    cat /proc/net/dev >> $RESULT_DIR/proc_network_card.log           		# Network Card
    
    if [ $HUMAN -eq 0 ] ; then
        cat /proc/$BotPID/stat >> $RESULT_DIR/proc_Bot_stat.log         		# Bot CPU Util
        cat /proc/$BotPID/status >> $RESULT_DIR/proc_Bot_status.log     		# Bot Memory Util
    fi
    cat /proc/$VncClientPID/stat >> $RESULT_DIR/proc_VncClient_stat.log        	# VNC client CPU Util
    cat /proc/$VncClientPID/status >> $RESULT_DIR/proc_VncClient_status.log    	# VNC client Mem Util
    sleep 1
    num=`expr $num - 1`
done
kill `cat ./client.pid`
echo "Done"

