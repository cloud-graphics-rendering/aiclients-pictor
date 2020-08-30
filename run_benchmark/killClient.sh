#!/bin/bash
# Author: Tianyi Liu
# Email: liuty10@gmail.com

kill `cat ./client.pid`
killall collectData.sh
