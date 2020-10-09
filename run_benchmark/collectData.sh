#!/bin/bash
# File  : collectData.sh
# Author: Tianyi Liu
# Email : liuty10@gmail.com

usage(){
     echo "Usage:"
     echo "*******************************************************************"
     echo "./collectData.sh GameName PerfFlag RunTime(s) BindCPU AutoOrHuman MultiMode"
     echo "	1) GameName   : supertuxkart-1, supertuxkart, 0ad, redeclipse, dota2, inmindvr, imhotepvr"
     echo "	2) ServerIP  : IP address of VNC server" 
     echo "	3) VncPort   : VNC port, starting from 5901" 
     echo "	4) TotalClients: How many clients to run games simultaneously: 1/2/3/4" 
     echo "	5) AutoRun    : Run game automatically? 1 or 0"
     echo "	6) Geometry  : 1920x1080 or 1280x720" 
     echo "	7) BenchTime  : How long to run the benchmark(seconds). Typically, stk, dota2, imhotepvr: 900; 0ad: 600; redclipse: 450; inmindvr: 240"
     echo "	8) PerfFlag   : 0 -- do NOT log metrics on server."
     echo "		        1 -- log performance results on server"
     echo "	9) BindCPU    : 0 -- donot bind python thread to specific cpu core"
     echo "			1 -- bind"
     echo "e.g: ./collectData.sh supertuxkart 10.100.233.197 5901 1 1 1920x1080 100 1 0"
     exit 0
}
if [ $# -ne 9 ]; then
	usage
fi
EXPERIMENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")

GAME_NAME=$1
SERVER_IP=$2
VNC_PORT=$3
TOTAL_CLIENTS=$4
AUTO_RUN=$5
GEOMETRY=$6
BENCH_TIME=$7
PERF_FLAG=$8
BIND_CPU=$9

if [ $AUTO_RUN ]
then
HUMAN_RUN=0
else
HUMAN_RUN=1
fi

WIDTH=$(echo $GEOMETRY | cut -dx -f1)
HEIGHT=$(echo $GEOMETRY | cut -dx -f2)
RESULT_DIR=`pwd`/result_client_cgr_multi_${TOTAL_CLIENTS}_${VNC_PORT}/$GAME_NAME
AI_BOTS_DIR=$(dirname `pwd`)

[ -e `dirname $RESULT_DIR` ] || mkdir `dirname $RESULT_DIR`
[ -e $RESULT_DIR ] || mkdir $RESULT_DIR

# writing metadata
echo "Date,GameName,ServerIP,VncPort,TotalClients,AutoRun,Geomrtry,BenchTime,PerfFlag,BindCPU" > $RESULT_DIR/metadata.csv 
echo "$EXPERIMENT_DATE,$GAME_NAME,$SERVER_IP,$VNC_PORT,$TOTAL_CLIENTS,$AUTO_RUN,$GEOMETRY,$BENCH_TIME,$PERF_FLAG,$BIND_CPU" >> $RESULT_DIR/metadata.csv

case $GAME_NAME in
    supertuxkart | 0ad | redeclipse | dota2 | inmindvr | imhotepvr | nasawebvr | supertuxkart-demo )
        if [ $BIND_CPU -eq 0 ] ; then
            python3 ../training_scripts/AI-play-${GAME_NAME}.py $PERF_FLAG $BENCH_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN_RUN $WIDTH $HEIGHT $TOTAL_CLIENTS& echo $! > ./.client.pid
        else
            taskset 0x1 python3 ../training_scripts/AI-play-${GAME_NAME}.py $PERF_FLAG $BENCH_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN_RUN $WIDTH $HEIGHT $TOTAL_CLIENTS& echo $! > ./.client.pid
        fi
        ;;
    supertuxkart-1 )
        if [ $BIND_CPU -eq 0 ] ; then
            python3 ../training_scripts/AI-play-supertuxkart-1.py $PERF_FLAG $BENCH_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN_RUN $WIDTH $HEIGHT $TOTAL_CLIENTS& echo $! > ./.client.pid
        else
            taskset 0x1 python3 ../training_scripts/AI-play-supertuxkart-1.py $PERF_FLAG $BENCH_TIME $AI_BOTS_DIR $RESULT_DIR $BIND_CPU $HUMAN_RUN $WIDTH $HEIGHT $TOTAL_CLIENTS& echo $! > ./.client.pid
        fi
        ;;

    *)
        echo "Application name NOT correct\n"
        exit 0
        ;;
esac
sleep 2

BotPID=`cat ./.client.pid`
VncClientPID=`cat ./.VncClient.pid`

networkcards=`ifconfig | grep -P '^[^\s]+\s+[A-Z]' | awk '{print $1}'`
echo $networkcards > $RESULT_DIR/networkcards_name.log

COUNT=$BENCH_TIME # number of records, BENCH_TIME/2;
echo "Running Time: $BENCH_TIME seconds"
echo "Count: $COUNT"
echo "running.."

num=$COUNT
while [ $num -gt 0 ]; do
    echo "Count: $num"
    cat /proc/stat >> $RESULT_DIR/proc_stat.log                      		# overall CPU
    cat /proc/meminfo >> $RESULT_DIR/proc_meminfo.log                		# overall Mem
    cat /proc/uptime >> $RESULT_DIR/proc_uptime.log                  		# For PID CPU
    cat /proc/net/dev >> $RESULT_DIR/proc_network_card.log           		# Network Card
    
    if [ $AUTO_RUN -eq 1 ] ; then
        cat /proc/$BotPID/stat >> $RESULT_DIR/proc_Bot_stat.log         		# Bot CPU Util
        cat /proc/$BotPID/status >> $RESULT_DIR/proc_Bot_status.log     		# Bot Memory Util
    fi
    cat /proc/$VncClientPID/stat >> $RESULT_DIR/proc_VncClient_stat.log        	# VNC client CPU Util
    cat /proc/$VncClientPID/status >> $RESULT_DIR/proc_VncClient_status.log    	# VNC client Mem Util
    sleep 1
    num=`expr $num - 1`
done
kill `cat ./.client.pid`
echo "Done"

