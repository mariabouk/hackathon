import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import accuracy_score,confusion_matrix

#collection of data for validation of the model
import processing
import pandas as pd
import numpy as np
#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plot

class0 = pd.read_csv('Class0.csv')
class1 = pd.read_csv('Class1.csv')
class2 = pd.read_csv('Class2.csv')
class3 = pd.read_csv('Class3.csv')

class0 = class0.values.reshape(1500,64,128)
class1 = class1.values.reshape(1500,64,128)
class2 = class2.values.reshape(1500,64,128)
class3 = class3.values.reshape(1500,64,128)

model = tf.keras.models.load_model('model')


#test for the accuracy of the validation set
pred = model.predict(class0)
predictions=np.zeros(len(pred))
for i in range(len(pred)):
    maxx = pred[i].argmax()
    predictions[i]=maxx
predictions

y_val = np.ones(1500)*0
acc0 = accuracy_score(y_val,predictions)  
confusion_matrix(y_val,predictions)


pred = model.predict(class1)
predictions=np.zeros(len(pred))
for i in range(len(pred)):
    maxx = pred[i].argmax()
    predictions[i]=maxx
predictions

y_val = np.ones(1500)*1
acc1 =accuracy_score(y_val,predictions)  
confusion_matrix(y_val,predictions)

pred = model.predict(class2)
predictions=np.zeros(len(pred))
for i in range(len(pred)):
    maxx = pred[i].argmax()
    predictions[i]=maxx
predictions

y_val = np.ones(1500)*2
acc2=accuracy_score(y_val,predictions)  


pred = model.predict(class3)
predictions=np.zeros(len(pred))
for i in range(len(pred)):
    maxx = pred[i].argmax()
    predictions[i]=maxx
predictions

y_val = np.ones(1500)*3
acc3 = accuracy_score(y_val,predictions)  


#diagram of the accuracy of the validation set
sns.barplot(x=[0,1,2,3],y=[acc0,acc1,acc2,acc3])
plt.title('Accuracy for the validation tests')






#######-------------------Real Time Data---------------------------------------
import time
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()           #between zero and 1

for i in range(100):
    config_file = "radar_configs/RadarIfxBGT60.json"
    number_of_frames = 1

    import json
    with open(config_file) as json_file:
        c = json.load(json_file)["device_config"]["fmcw_single_shape"]
        chirp_duration = c["num_samples_per_chirp"]/c['sample_rate_Hz']
        frame_duration = (chirp_duration + c['chirp_repetition_time_s']) * c['num_chirps_per_frame']
        #print("With the current configuration, the radar will send out " + str(c['num_chirps_per_frame']) + \
        #      ' signals with varying frequency ("chirps") between ' + str(c['start_frequency_Hz']/1e9) + " GHz and " + \
        #      str(c['end_frequency_Hz']/1e9) + " GHz.")
        #print('Each chirp will consist of ' + str(c["num_samples_per_chirp"]) + ' ADC measurements of the IF signal ("samples").')
        #print('A chirp takes ' + str(chirp_duration*1e6) + ' microseconds and the delay between the chirps is ' + str(c['chirp_repetition_time_s']*1e6) +' microseconds.')
        #print('With a total frame duration of ' + str(frame_duration*1e3) + ' milliseconds and a delay of ' + str(c['frame_repetition_time_s']*1e3) + ' milliseconds between the frame we get a frame rate of ' + str(1/(frame_duration + c['frame_repetition_time_s'])) + ' radar frames per second.')
        


    raw_data    = []
    with RadarIfxAvian(config_file) as device:                             # Initialize the radar with configurations
        
        for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
            
            raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
            if(i_frame == number_of_frames-1):
                data = np.asarray(raw_data)
                range_doppler_map = processing.processing_rangeDopplerData(data)
                #del data
                break      



    X_val = raw_data.copy()
    X_val = np.array(X_val)  
    X_val = X_val.reshape(3,64*128)

    X_val = scaler.fit_transform(X_val)*255
    X_val = X_val.reshape(3,64,128)



    pred = testing_model.predict(X_val)
    predictions=np.zeros(len(pred))
    for i in range(len(pred)):
        maxx = pred[i].argmax()
        predictions[i]=maxx
    print(predictions)
    print('\n')
    time.sleep(1)
    



