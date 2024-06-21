import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import json
import time
import serial
import RPi.GPIO as GPIO
import cv2
import os
import joblib
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Define motor control pins
IN1 = 27
IN2 = 17

# Define Liquid Sensor pin
WLS = 2
GPIO.setmode(GPIO.BCM)
GPIO.setup(WLS, GPIO.IN)

# Define serial port and baud rate to receive data from Arduino
ser = serial.Serial('/dev/ttyACM0', 115200)

def init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)

def turn_forward(sec):
    init()
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    print("Turning forward...")
    time.sleep(sec)
    GPIO.cleanup()

def turn_backward(sec):
    init()
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    print("Turning backward...")
    time.sleep(sec)
    GPIO.cleanup()

class DataHandler:
    def __init__(self):
        self.broker_address = "mqtt.eclipseprojects.io"
        self.topic = "Spirulina_Edge"
        self.api_key = os.environ['OPENAI_API_KEY']
        self.client = mqtt.Client()
        self.connected = self.connect_to_broker()

        # Load models and scalers
        self.load_models()
        self.load_scalers_and_encoders()

    def connect_to_broker(self):
        try:
            self.client.connect(self.broker_address)
            return True
        except Exception as e:
            print("Failed to connect to broker:", e)
            return False

    def load_models(self):
        self.ph_model = tf.lite.Interpreter(model_path='tflite_model/ph_model.tflite')
        self.ph_model.allocate_tensors()
        self.ec_model = tf.lite.Interpreter(model_path='tflite_model/ec_model.tflite')
        self.ec_model.allocate_tensors()
        self.temp_model = tf.lite.Interpreter(model_path='tflite_model/temperature_model.tflite')
        self.temp_model.allocate_tensors()
        self.transparency_model = tf.lite.Interpreter(model_path='tflite_model/transparency_model.tflite')
        self.transparency_model.allocate_tensors()

    def load_scalers_and_encoders(self):
        self.ec_scaler = joblib.load('tflite_model/ec_scaler.pkl')
        self.ec_label_encoder = joblib.load('tflite_model/ec_label_encoder.pkl')
        self.ph_scaler = joblib.load('tflite_model/ph_scaler.pkl')
        self.ph_label_encoder = joblib.load('tflite_model/ph_label_encoder.pkl')
        self.temperature_scaler = joblib.load('tflite_model/temperature_scaler.pkl')
        self.temperature_label_encoder = joblib.load('tflite_model/temperature_label_encoder.pkl')
        self.transparency_scaler = joblib.load('tflite_model/transp_scaler.pkl')
        self.transparency_label_encoder = joblib.load('tflite_model/transp_label_encoder.pkl')

    def process_sensor_data(self):
        num_iterations = 5
        ph_values = []
        temp_values = []
        ec_values = []

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

        print(f'Average pH: {avg_ph}, Average Temperature: {avg_temp}, Average EC: {avg_ec}')

        print('wls data')
        levelbassin = 25
        WLS = GPIO.input(2)
        if WLS == 1:
            time_ex = time.time() - start_time
            Water_Level = levelbassin - (time_ex * 0.27)
        else:
            Water_Level = levelbassin
            
        print('camera data')
        webcam = cv2.VideoCapture(0)
        # Check if the camera opened successfully
        if not webcam.isOpened():
            print("Error: Could not open video device")
            exit()

        # Set camera resolution (optional)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame")
            exit()

        print('disaply frame data')
        cv2.imshow('Camera', frame)
        b, g, r = 255, 255, 255
        while b != 0 and g != 0 and r != 0:
            _, imageFrame = webcam.read()
            hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            white_lower = np.array([0, 0, 255], np.uint8)
            white_upper = np.array([255, 255, 255], np.uint8)
            white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
            cv2.imwrite('frame.jpg', white_mask)
            im = cv2.imread('frame.jpg')
            b, g, r = im[300, 300]
            if b == 0 and g == 0 and r == 0:
                time_cam = time.time() - start_time - time_ex
                Disappearance_Height = time_ex * 0.27
                cv2.destroyAllWindows()
                break  # Exit the loop when the condition is met
            time.sleep(60)

        user_input = (f"temperature={avg_temp};Ph_value={avg_ph};water_level={Water_Level};conductivity={avg_ec};brightness={Disappearance_Height};")

        if self.connected:
            result = self.generate_json(user_input)
            self.publish_data(result)
            print('connected')

        else:
            self.control_actuators(avg_ph, avg_temp, avg_ec, Disappearance_Height)
            print(' not connected')

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

    def control_actuators(self, ph_value, temperature, conductivity, transparency):
        # Scale inputs
        scaled_ph = self.ph_scaler.transform(np.array([[ph_value]]).astype(np.float32))
        scaled_conductivity = self.ec_scaler.transform(np.array([[conductivity]]).astype(np.float32))
        scaled_temp = self.temperature_scaler.transform(np.array([[temperature]]).astype(np.float32))
        scaled_transparency = self.transparency_scaler.transform(np.array([[transparency]]).astype(np.float32))

        # Predict with TFLite models
        self.ph_model.set_tensor(self.ph_model.get_input_details()[0]['index'], np.array([[ph_value]], dtype=np.float32))
        self.ph_model.invoke()
        ph_prediction = self.ph_model.get_tensor(self.ph_model.get_output_details()[0]['index'])
        
        self.ec_model.set_tensor(self.ec_model.get_input_details()[0]['index'], scaled_conductivity)
        self.ec_model.invoke()
        ec_prediction = self.ec_model.get_tensor(self.ec_model.get_output_details()[0]['index'])
        
        self.temp_model.set_tensor(self.temp_model.get_input_details()[0]['index'], np.array([[temperature]], dtype=np.float32))
        self.temp_model.invoke()
        temp_prediction = self.temp_model.get_tensor(self.temp_model.get_output_details()[0]['index'])
        
        self.transparency_model.set_tensor(self.transparency_model.get_input_details()[0]['index'], scaled_transparency)
        self.transparency_model.invoke()
        transparency_prediction = self.transparency_model.get_tensor(self.transparency_model.get_output_details()[0]['index'])

        # Decode predictions
        ph_status = np.argmax(ph_prediction, axis=-1)[0]
        ec_status = np.argmax(ec_prediction, axis=-1)[0]
        temp_status = np.argmax(temp_prediction, axis=-1)[0]
        transparency_status = np.argmax(transparency_prediction, axis=-1)[0]

        # Control logic based on predictions
        

while True:
    start_time = time.time()
    turn_forward(2)
    time.sleep(1)
    handler = DataHandler()
    print("Measuring")
    handler.process_sensor_data()
    handler.client.disconnect()
    turn_backward(2)
    time.sleep(1)
    print("Sleeping for 2 hours...")
    time.sleep(60 * 60 * 2)
