import numpy as np
import paho.mqtt.client as mqtt
import json
import time
import serial
import RPi.GPIO as GPIO
import cv2
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from ModelHandler import ModelHandler


# Define GPIO pins
IN1 = 27
IN2 = 17
LED_PH = 22
LED_EC = 23
LED_TEMP = 24
LED_TRANSP = 25


# Define serial port and baud rate to receive data from Arduino UNO
ser = serial.Serial('/dev/ttyACM0', 115200)


def init_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(LED_PH, GPIO.OUT)
    GPIO.setup(LED_EC, GPIO.OUT)
    GPIO.setup(LED_TEMP, GPIO.OUT)
    GPIO.setup(LED_TRANSP, GPIO.OUT)


def turn_forward():
    init_gpio()
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    print("Turning forward...")
    #time.sleep(sec)
    #GPIO.cleanup()


def turn_backward(sec):
    init_gpio()
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    print("Turning backward...")
    time.sleep(sec)
    GPIO.cleanup()


def turn_motor_off():
    init_gpio()
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    print("Turning motor off...")
    #time.sleep(sec)
    #GPIO.cleanup()


class DataHandler:
    def __init__(self):
        self.broker_address = "mqtt.eclipseprojects.io"
        self.topic = "Spirulina_Edge"
        self.listen_topic = "spirulina"
        self.api_key = os.environ['OPENAI_API_KEY']
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.connected = self.connect_to_broker()


        # Load models and scalers
        self.model_handler = ModelHandler()
        self.model_handler.load_models()
        self.model_handler.load_scalers_and_encoders()


    def connect_to_broker(self):
        try:
            self.client.connect(self.broker_address)
            self.client.loop_start()
            return True
        except Exception as e:
            print("Failed to connect to broker:", e)
            return False


    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to broker")
            self.client.subscribe(self.listen_topic)
        else:
            print("Failed to connect, return code %d\n", rc)


    def on_message(self, client, userdata, msg):
        #Receive commands from the LLM model to activate actuators based on it
        print("Message received from topic {}: {}".format(msg.topic, msg.payload.decode()))
        data = json.loads(msg.payload.decode())
        print(data)
        self.control_actuators(data)


    def process_camera_level(self):
	    # Define initial coordinates and size of the ROI frame
	    roi_x, roi_y, roi_width, roi_height = 150, 120, 280, 120
	
	    def update_roi(event, x, y, flags, param):
	        nonlocal roi_x, roi_y, roi_width, roi_height
	        if event == cv2.EVENT_LBUTTONDOWN:
	            roi_x, roi_y = x, y
	        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
	            roi_width, roi_height = x - roi_x, y - roi_y
	
	    # Create a VideoCapture object
	    webcam = cv2.VideoCapture(0)
	
	    # Check if the camera opened successfully
	    if not webcam.isOpened():
	        print("Error: Could not open video device")
	        return None
	
	    # Set camera resolution
	    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	
	    Disappearance_Height = None
	
	    while True:
	        ret, frame = webcam.read()
	        if not ret:
	            print("Error: Could not read frame inside loop")
	            break
	
	        roi = frame[roi_y:roi_y + roi_height, roi_x: roi_x + roi_width]
	        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	        white_lower = np.array([0, 0, 200], np.uint8) # lower white in hsv
	        white_upper = np.array([180, 55, 255], np.uint8) # upper white in hsv
	        white_mask = cv2.inRange(hsv_roi, white_lower, white_upper)
	
	        # Check the center pixel of the ROI in the white mask
	        center_y = roi_height // 4
	        center_x = roi_width // 2
	        b, g, r = white_mask[center_y, center_x], white_mask[center_y, center_x], white_mask[center_y, center_x]
	        
	        
	        # Show the frame with ROI to ensure it's facing the target
	        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
	        cv2.imshow("Frame", white_mask)
	
	        # Check if the spot has become dark indicating black
	        if b == 0 and g == 0 and r == 0:
	            Disappearance_Height = time_ex * 0.4  # Calculate disappearance height
	            print(f"Disappearance Height: {Disappearance_Height:.2f} cm")
	            cv2.destroyAllWindows()
	            break
	
	
	        # Wait for key press and handle key events
	        key = cv2.waitKey(30) & 0xFF
	        if key == ord('q'):
	            break
	
	    webcam.release()
	    cv2.destroyAllWindows()
	    return Disappearance_Height



    def process_sensor_data(self, Water_Level, Disappearance_Height):
        num_iterations = 10
        ph_values = []
        temp_values = []
        ec_values = []
        #### 10 iterations to get the average value of each sensor in case of absence of certain sensor value
        print('Collecting data...')
        for _ in range(num_iterations):
            while True:
                data = ser.readline().decode().strip()
                if self.is_valid_sensor_data(data):
                    ArdphValue, ArdTemperatureValue, ArdConductivityValue = map(float, data.split(","))
                    ph_values.append(ArdphValue)
                    temp_values.append(ArdTemperatureValue)
                    ec_values.append(ArdConductivityValue)
                    break
            time.sleep(1)  # wait for 1 second between readings


        avg_ph = np.mean(ph_values)
        avg_temp = np.mean(temp_values)
        avg_ec = np.mean(ec_values)
        
        avg_ph = round(avg_ph, 2) 
        avg_temp = round(avg_temp, 2) 
        avg_ec = round(avg_ec, 2) 
        Water_Level = round(Water_Level, 2) 
        Disappearance_Height = round(Disappearance_Height, 2)



        print(f'Average pH: {avg_ph}, Average Temperature: {avg_temp}, Average EC: {avg_ec}')


        user_input = (f"temperature={avg_temp};Ph_value={avg_ph};water_level={Water_Level};conductivity={avg_ec};brightness={Disappearance_Height};")


        if self.connected:
            ########## if internet connection is available : sending data to LLM model
            result = self.generate_json(user_input)
            self.publish_data(result)
            print('connected')


        else:
            ########## in case of internet connection absence : generating commands with tinyML models
            predictions = self.model_handler.load_predictions(avg_ph, avg_temp, avg_ec, Disappearance_Height)
            predictions_dict = { "ph_prediction": predictions[0], "temp_prediction": predictions[1], "ec_prediction": predictions[2], "transparency_prediction": predictions[3] }
            print('not connected')
            print(f'pH_value: {avg_ph}, Temperature: {avg_temp}, EC_value: {avg_ec}, Optical density: {Disappearance_Height}, Water_level: {Water_Level}')
            self.control_actuators(predictions_dict)
   
   
    ############this method to split the data received from the arduino UNO
    def is_valid_sensor_data(self, data):
        try:
            values = data.split(",")
            if len(values) == 3:
                float(values[0])
                float(values[1])
                float(values[2])
                return True
        except ValueError:
            return False
        return False


    def generate_json(self, user_input):
        application_prompt = """Make sure that all responses are in json format and the values are float
        DESCRIPTION:
        {user_input}
        """
        llm = ChatOpenAI(
            temperature=0.7,
            max_tokens=500,
            model="gpt-3.5-turbo",
            api_key=self.api_key
        )
        prompt = PromptTemplate(
            input_variables=["user_input"],
            template=application_prompt
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"user_input": user_input})
        return result


    def publish_data(self, data):
        self.client.publish(self.topic, json.dumps(data))
        print("Published data:", data)


   
    def control_actuators(self, data):
        print('activate actuators')
        ph_values = data.get('ph_prediction', 0)####change all with that name data_value
        ec_values = data.get('ec_prediction', 0)
        temp_values = data.get('temp_prediction', 0)
        transparency_values = data.get('transparency_prediction', 0)
       
        #####control actuators###########
        print(ph_values,ec_values,temp_values,transparency_values)
       
        GPIO.cleanup()


while True:
    start_time = time.time()
    turn_forward()
    print('Waiting for water level...')
    
    # Initialize DataHandler instance
    handler = DataHandler()

    # Wait for water level sensor to detect water
    levelbassin = 25
    WLS = 2
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(WLS, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    while GPIO.input(WLS) == 0:
        time.sleep(0.1)  # Check water sensor every 0.1 seconds

    # Water detected, now start checking for black spots using camera
    print('Water detected. Checking for black spots...')
    time_ex = time.time() - start_time
    Water_Level = levelbassin - (time_ex * 0.4)
    density = None
    while density is None:
        density = handler.process_camera_level()
        time.sleep(0.1)  # Check camera every 0.1 seconds

    # Now that black is detected, proceed with sensor data collection
    print('Black spot detected. Collecting sensor data...')
    turn_motor_off()
    handler.process_sensor_data(Water_Level, density)

    # Turn backward with the same time duration as forward
    turn_backward(time_ex)
    print("Sleeping for 2 hours...")
    time.sleep(2 * 3600)  # Sleep for 2 hours
