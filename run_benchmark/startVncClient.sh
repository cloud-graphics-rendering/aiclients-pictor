#!/bin/sh
# File	: startVncClient.sh
# Author: Tianyi Liu
# Email : liuty10@gmail.com

usage(){
     echo "Usage:"
     echo "*******************************************************************"
     echo "./startVncClient.sh GameName ServerSERVER_IP VNCPort NumOfGames auto/human"
     echo "	1) GameName  : supertuxkart-1, supertuxkart, 0ad, redeclipse, dota2, inmindvr, imhotepvr"
     echo "	2) ServerIP  : IP address of VNC server" 
     echo "	3) VncPort   : VNC port, starting from 5901" 
     echo "	4) TotalClients: How many clients to run games simultaneously: 1/2/3/4" 
     echo "	5) AutoRun   : Run game automatically? 1 or 0" 
     echo "	6) Geometry  : 1920x1080 or 1280x720" 
     echo ""
     echo "e.g: ./startVncClient.sh supertuxkart 10.100.233.197 5901 1 1 1920x1080"
     exit 0
}

if [ $# -ne 6 ]; then
    usage
fi

GAME_NAME=$1
SERVER_IP=$2
VNC_PORT=$3
TOTAL_CLIENTS=$4
AUTO_RUN=$5
GEOMETRY=$6

MYDISPLAY=$(expr $VNC_PORT - 5900)
RESULT_DIR=`pwd`/result_client_cgr_multi_${TOTAL_CLIENTS}_${VNC_PORT}/$GAME_NAME

[ -e `dirname $RESULT_DIR` ] || mkdir `dirname $RESULT_DIR`
[ -e $RESULT_DIR ] || mkdir $RESULT_DIR
rm $RESULT_DIR/*

ssh lty@$SERVER_IP "/opt/TurboVNC/bin/vncserver -kill :$MYDISPLAY; /opt/TurboVNC/bin/vncserver -securitytypes None -geometry $GEOMETRY" && \
echo "ssh lty@$SERVER_IP "/opt/TurboVNC/bin/vncserver -kill :$MYDISPLAY"" > ./.killVncSession.sh
                                                   
/opt/TurboVNC/bin/vncviewer -DesktopSize Server $SERVER_IP:$VNC_PORT > $RESULT_DIR/RTT.log&
sleep 2
echo `ps -C java -o pid=` > ./.VncClient.pid
