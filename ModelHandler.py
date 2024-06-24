import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib


class ModelHandler:
    def __init__(self):
        self.load_models()
        self.load_scalers_and_encoders()


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


    def load_predictions(self, ph_value, temperature, conductivity, transparency):
        # Scale inputs
        scaled_ph = self.ph_scaler.transform(np.array([[ph_value]]).astype(np.float32))
        scaled_conductivity = self.ec_scaler.transform(np.array([[conductivity]]).astype(np.float32))
        scaled_temp = self.temperature_scaler.transform(np.array([[temperature]]).astype(np.float32))
        scaled_transparency = self.transparency_scaler.transform(np.array([[transparency]]).astype(np.float32))


        # Predict with TFLite models
        self.ph_model.set_tensor(self.ph_model.get_input_details()[0]['index'], scaled_ph)
        self.ph_model.invoke()
        ph_prediction = self.ph_model.get_tensor(self.ph_model.get_output_details()[0]['index'])
       
        self.ec_model.set_tensor(self.ec_model.get_input_details()[0]['index'], scaled_conductivity)
        self.ec_model.invoke()
        ec_prediction = self.ec_model.get_tensor(self.ec_model.get_output_details()[0]['index'])
       
        self.temp_model.set_tensor(self.temp_model.get_input_details()[0]['index'], scaled_temp)
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


        return ph_status, ec_status, temp_status, transparency_status




