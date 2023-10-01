from pickle import TRUE
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import os


def random_process_data(imgs, labels, singles):
    indices = np.random.permutation(imgs.shape[0])
    imgs = imgs[indices]  
    labels = labels[indices]
    if singles:
        return imgs[0:1], labels[0:1]
    return imgs, labels

signal_data = np.load('signal_test.npy')
signal_data_number = len(signal_data)
symbol_data = np.load('symbol_test.npy')
symbol_data_number = len(symbol_data)

model = keras.models.load_model('mitbihecg_model.h5')

# Timer
total_inference_time = 0


batch_mode = True
####### MODE CHANGER #################
######################################
batch_mode_input = input("1 - by batch or 0 - by sample:   ")
#print(batch_mode_input)
if (batch_mode_input == '1'):
    batch_mode = True
else:
    batch_mode = False 
#print(batch_mode)
######################################
######################################



if batch_mode:
    
    all_time = 0 
    all_loss = 0
    all_accuracy = 0 
    print("batch mode")
    input_num_samples = input("enter the number samples to be evaluated (max is 18000): ")
    num_samples = int(input_num_samples)
    input_num_test = input("enter the number of times the batch of samples is to be evaluated: ")
    num_test = int(input_num_test)


    for i in range(num_test):
        new_signal_data, new_symbol_data = random_process_data(signal_data, symbol_data, False )
        start = time.time()
        eval = model.evaluate(new_signal_data[0:num_samples], new_symbol_data[0:num_samples],verbose=1)
        end = time.time()
        elapsed_time = end - start

        j = i+1
        print("Test number: " + str(j))
        print("Whole Eval:", eval)
        print('Test Loss:', eval[0])
        print('Test Accuracy:', eval[1])
        print("")
        print("")


        average_inference_time = elapsed_time / num_samples
        all_time += average_inference_time
        all_loss += eval[0]
        all_accuracy += eval[1]


    #AVERAGING OUT ALL OUT OF THE ADDED TESTS
    average_all_sample_time = (all_time/num_test)*1000 
    average_all_loss = all_loss/num_test 
    average_all_accuracy = all_accuracy/num_test 

    print("")
    print("")
    print("Average evaluation time for " + str(num_samples) + " samples test for " + str(num_test) + " times is: " + str(average_all_sample_time) + " ms")
    print("Average loss for " + str(num_samples) + " samples test for " + str(num_test) + " times is: " + str(average_all_loss))
    print("Average accuracy for " + str(num_samples) + " samples test for " + str(num_test) + " times is: " + str(average_all_accuracy))
else:
    total_inference_time = 0 
    N_count = 0
    A_count = 0
    V_count = 0
    L_count = 0
    R_count = 0
    print("sample mode")
    input_num_samples = input("enter the number samples to be evaluated (max is 18000): ")
    num_samples = int(input_num_samples)


    new_signal_data, new_symbol_data = random_process_data(signal_data, symbol_data, False)
    for i in range(0, num_samples):
        new_signal_data, new_symbol_data = random_process_data(signal_data, symbol_data, True)

        #print("new_signal_data", new_signal_data.shape)
        #print("new_symbol_data", new_symbol_data.shape)

        try:
            start = time.time()
            prediction = model.predict(new_signal_data, new_symbol_data,verbose=0)
            elapsed_time = time.time() - start
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
    

















