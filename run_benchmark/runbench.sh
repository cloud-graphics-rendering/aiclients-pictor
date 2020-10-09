#!/bin/bash
set -x #echo on
# ./runbench.sh 2>&1 | tee ./xxx/run.log 

SERVER_IP=10.100.233.197
TOTAL_CLIENTS=1
BIND_CPU=0

PERF_FLAG=1
AUTO_RUN=1
VNC_PORT=5901
GEOMETRY=1920x1080

sh ./startVncClient.sh supertuxkart $SERVER_IP $VNC_PORT $TOTAL_CLIENTS $AUTO_RUN $GEOMETRY
sh ./collectData.sh supertuxkart $SERVER_IP $VNC_PORT $TOTAL_CLIENTS $AUTO_RUN $GEOMETRY 100 $PERF_FLAG $BIND_CPU
sleep 30 && kill `cat ./.VncClient.pid`

sh ./startVncClient.sh 0ad $SERVER_IP $VNC_PORT $TOTAL_CLIENTS $AUTO_RUN $GEOMETRY
sh ./collectData.sh 0ad $SERVER_IP $VNC_PORT $TOTAL_CLIENTS $AUTO_RUN $GEOMETRY 100 $PERF_FLAG $BIND_CPU
sleep 30 && kill `cat ./.VncClient.pid`

sh ./startVncClient.sh redeclipse $SERVER_IP $VNC_PORT $TOTAL_CLIENTS $AUTO_RUN $GEOMETRY
sh ./collectData.sh redeclipse $SERVER_IP $VNC_PORT $TOTAL_CLIENTS $AUTO_RUN $GEOMETRY 100 $PERF_FLAG $BIND_CPU
sleep 30 && kill `cat ./.VncClient.pid`

# Kill the remote VNC session
sh ./.killVncSession.sh
