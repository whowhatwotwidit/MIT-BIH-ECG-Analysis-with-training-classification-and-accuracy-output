from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/a/42121886

def random_process_data(imgs, labels, singles):
    indices = np.random.permutation(imgs.shape[0])
    imgs = imgs[indices]  
    labels = labels[indices]
    #print("indices:", indices)
    #print("labels:", labels)
    if singles:
        return imgs[0:1], labels[0:1]
    return imgs, labels

signal_data = np.load('signal_test.npy')
signal_data_number = len(signal_data)
symbol_data = np.load('symbol_test.npy')
symbol_data_number = len(symbol_data)

model = keras.models.load_model('mitbihecg_model.h5')

#signal_data, symbol_data = random_process_data(signal_data, symbol_data)

# Timer
total_inference_time = 0
num_samples = 1000 
N_count = 0
A_count = 0
V_count = 0
L_count = 0
R_count = 0
#len(signal_data)

new_signal_data, new_symbol_data = random_process_data(signal_data, symbol_data, False)
'''
eval = model.evaluate(new_signal_data[0:100], new_symbol_data[0:100],verbose=1)
print("0:100 Eval:", eval)
print("")

eval = model.evaluate(new_signal_data[0:200], new_symbol_data[0:200],verbose=1)
print("0:200 Eval:", eval)
print("")

eval = model.evaluate(new_signal_data[0:500], new_symbol_data[0:500],verbose=1)
print("0:500 Eval:", eval)
print("")

eval = model.evaluate(new_signal_data, new_symbol_data,verbose=1)
print("Whole Eval:", eval)
print("")
'''

for i in range(0, num_samples):
    new_signal_data, new_symbol_data = random_process_data(signal_data, symbol_data, True)

    #print("new_signal_data", new_signal_data.shape)
    #print("new_symbol_data", new_symbol_data.shape)

    try:
        start = time.time()
        prediction = model.predict(new_signal_data, new_symbol_data,verbose=0)
        elapsed_time = time.time() - start
        #eval = model.evaluate(new_signal_data, new_symbol_data,verbose=0)
        total_inference_time += elapsed_time
        ecgClassSet = ['N', 'A', 'V', 'L', 'R']
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = ecgClassSet[predicted_class_index]
        print(f"Sample: {i + 1}")
        
        if predicted_class_label == 'N':
            N_count +=1
        elif predicted_class_label == 'A':
            A_count +=1
        elif predicted_class_label == 'V':
            V_count +=1
        elif predicted_class_label == 'L':
            L_count +=1
        elif predicted_class_label == 'R':
            R_count +=1
        
        print(f"Predicted class: {predicted_class_label}")
        #print(f"Test Loss: {eval[0]:.4f}")
        #print(f"Test Accuracy: {1 - eval[0]:.4f}")
        print(f"Inference Time: {elapsed_time:.4f} seconds\n")
    except:
        print(f"Sample {i} skipped due to error")
        print("")
        continue
    
print ("class N :" + str(N_count))
print ("class A :" + str(A_count))
print ("class V :" + str(V_count))
print ("class L :" + str(L_count))
print ("class R :" + str(R_count))

# Calculate average inference time
average_inference_time = total_inference_time / num_samples

print('Average Inference Time:', average_inference_time, 'seconds')
#print(signal_data_number)
#print(symbol_data_number)