#!/bin/sh
# File	: startVncClient.sh
# Author: Tianyi Liu
# Email : liuty10@gmail.com

usage(){
     echo "Usage:"
     echo "*******************************************************************"
     echo "./startVncClient.sh GameName ServerIP VNCPort NumOfGames auto/human"
     echo "	1) GameName  : supertuxkart-1, supertuxkart, 0ad, redeclipse, dota2, inmindvr, imhotepvr"
     echo "	2) ServerIP  : IP address of VNC server" 
     echo "	3) VNCPort   : VNC port, starting from 5901" 
     echo "	4) NumOfGames: How many clients to run games simultaneously" 
     echo "	5) auto/human: How to run the games? auto or human?" 
     echo ""
     echo "e.g: ./startVncClient.sh supertuxkart 10.100.233.197 5901 single/1/2/3/4 auto"
     exit 0
}

if [ $# -le 2 ]; then
    usage
fi

APP=$1
IP=$2
PORT=$3
MODE=$4
AUTO=$5

if [ $MODE = 'single' ]
then
    if [ $AUTO = 'auto' ]
    then
        RESULT_DIR=./result_client_cgr_nobind_auto
    else
        RESULT_DIR=./result_client_cgr_nobind_human
    fi
else
    HOSTNAME=`cat /etc/hostname`
    if [ $AUTO = 'auto' ]
    then
    	RESULT_DIR=./result_client_cgr_nobind_auto_multi_${MODE}_${HOSTNAME}
    else
    	RESULT_DIR=./result_client_cgr_nobind_human_multi_${MODE}_${HOSTNAME}
    fi
fi

[ -e $RESULT_DIR ] || mkdir $RESULT_DIR
/opt/TurboVNC/bin/vncviewer $IP:$PORT > $RESULT_DIR/${APP}_RTT.log&

