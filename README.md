the DataHandler script is the main script: 
First it turns forward the linear actuator until it detects the pond's water to calculate the water level.
After water presence, the linear actuator keeps going forward until the camera detects black signifies the transparency i.e optical density.
And only then, the linear actuator stops and the sensors (pH, Conductivity, Temperature) collect data and sends it from the arduino UNO to the raspberry Pi to get processed.
After collecting the data, it will be published to the LLM model via the publisher and the linear actuator turns backward to its initial form indicating the end of the loop which will be 2hours until it starts again.
While waiting for the 2hours loop, the system keeps waiting for the commands from the LLM model to act upon it.
the ModelHandler is the script respensible for loading the models in case of internet connection absence and the data collected need to be processed by the models to generate the commands enabling the activation of the actuators.
