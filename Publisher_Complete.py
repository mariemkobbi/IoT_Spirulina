import paho.mqtt.client as mqtt
import json
import time
import serial
import RPi.GPIO as GPIO
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Define motor control pins
IN1 = 8
IN2 = 7
ENB = 9


# Define Liquid Sensor pins
WLS = 2
GPIO.setmode(GPIO.BCM)  # Set mode 
GPIO.setup(WLS, GPIO.IN)

# Define serial port and baud rate to receive data from Arduino
ser = serial.Serial('/dev/ttyACM0', 9600)

# Function to turn motor forward
def turn_forward():
  GPIO.setmode(GPIO.BCM)  # Set mode 
  GPIO.setup(IN1, GPIO.OUT)
  GPIO.setup(IN2, GPIO.OUT)
  GPIO.setup(ENB, GPIO.OUT)
  GPIO.output(IN1, GPIO.HIGH)
  GPIO.output(IN2, GPIO.LOW)
  # Set PWM for motor speed control (optional)
  pwm = GPIO.PWM(ENB, 100)  # Adjust frequency as needed
  pwm.start(0)  # Initial duty cycle (0% speed)
  print("Turning forward...")
  # Replace with your desired forward movement duration
  time.sleep(10)  # Adjust forward movement time
  pwm.stop()
  GPIO.cleanup()  # Clean up GPIO pins
  
# Function to turn motor backward
def turn_backward():
  GPIO.setmode(GPIO.BCM)  # Set mode 
  GPIO.setup(IN1, GPIO.OUT)
  GPIO.setup(IN2, GPIO.OUT)
  GPIO.setup(ENB, GPIO.OUT)
  GPIO.output(IN1, GPIO.LOW)
  GPIO.output(IN2, GPIO.HIGH)
  # Set PWM for motor speed control (optional)
  pwm = GPIO.PWM(ENB, 100)  # Adjust frequency as needed
  pwm.start(0)  # Initial duty cycle (0% speed)
  print("Turning backward...")
  # Replace with desired forward movement duration
  time.sleep(10)  # Adjust backward movement time
  pwm.stop()
  GPIO.cleanup()  # Clean up GPIO pins

class DataHandler:
  def __init__(self):
    # MQTT configuration
    self.broker_address = "mqtt.eclipseprojects.io"
    self.topic = "Spirulina_Edge"
    self.api_key = os.environ['OPENAI_API_KEY']
    # Create MQTT client
    self.client = mqtt.Client()
    self.client.connect(self.broker_address)

  def process_sensor_data(self):
    # Read data from Arduino
    data = ser.readline().decode().strip()

    # Split the data into individual values
    ArdphValue, ArdTemperatureValue, ArdConductivityValue = map(float, data.split(","))

    # Calculate water level 
    levelbassin = 25
    if WLS == 1:
      time_ex = time.time() - start_time  
      Water_Level = levelbassin - (time_ex * 4)
    else:
      Water_Level = levelbassin
      
    #Camera 
    brightness = 5

    # Create user input string
    user_input = f"temperature={ArdTemperatureValue};Ph_value={ArdphValue};water_level={Water_Level};conductivity={ArdConductivityValue}; brightness={brightness};"
    # Generate JSON using LLM
    result = self.generate_json(user_input)
    # Publish JSON data to MQTT
    self.publish_data(result)
    
  def generate_json(self, user_input):
    application_prompt = """Make sure that all responses are in json format and the values are float
    DESCRIPTION:
    {user_input}
    """
    llm = ChatOpenAI(
        temperature=0.7,
        max_tokens=500,
        model = "gpt-3.5-turbo",
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

# Main loop
while True:
  start_time = time.time()
  # Turn motor forward
  turn_forward()
  time.sleep(1)
  # Create and run data handler
  handler = DataHandler()
  handler.process_sensor_data()
  # Disconnect from MQTT broker 
  handler.client.disconnect()
  # Turn motor backward
  turn_backward()
  time.sleep(1)
  # Sleep for 2 hours 
  print("Sleeping for 2 hours...")
  time.sleep(60 * 60 * 2)  # Sleep for 2 hours 
