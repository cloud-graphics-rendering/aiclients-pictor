How to use:
cd run_benchmark

Run ./startVncClient.sh, and see the guidline
After the command above, you should connect to VNC server successfully.
Open a terminal by hand, and put it at the center of screen.

Run ./collectData.sh, and see the guidline
After running the above command with proper params, the games will start and run automatically.


Details about the AI scripts:
cgrAPI.gameBotParamInit(): is used to set LSTM related params.
cgrAPI.globalParamInit() : set other params after input of commandline.
cgrAPI.LSTMInit()	 : Init LSTM model.
cgrAPI.CNNInit()	 : Init CNN model.
cgrAPI.logsInit()	 : set CNN/LSTM log path.
cgrAPI.commandInit()	 : mimic keyboard and mouse to start game at server side with neccessary params.
