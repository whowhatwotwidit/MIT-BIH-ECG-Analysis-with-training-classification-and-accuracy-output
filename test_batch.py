from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/a/42121886

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


# Timer
total_inference_time = 0
num_test = 50
num_samples = 18000 #len(signal_data) 
print("the number of samples is: " + str(num_samples))
all_time = 0 
all_loss = 0
all_accuracy = 0 

'''
eval = model.evaluate(new_signal_data, new_symbol_data,verbose=1)

print("Eval:", eval)
print("")
'''
x = 50
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


########## IBANG METHOD
'''
for i in range(0, num_samples):
    new_signal_data, new_symbol_data = random_process_data(signal_data, symbol_data, True)

    #print("new_signal_data", new_signal_data.shape)
    #print("new_symbol_data", new_symbol_data.shape)
    start = time.time()
    try:

        prediction = model.predict(new_signal_data, new_symbol_data,verbose=0)
        elapsed_time = time.time() - start
        #eval = model.evaluate(new_signal_data, new_symbol_data,verbose=0)
        total_inference_time += elapsed_time
        ecgClassSet = ['N', 'A', 'V', 'L', 'R']
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = ecgClassSet[predicted_class_index]
        print(f"Sample: {i + 1}")
        print(f"Predicted class: {predicted_class_label}")
        #print(f"Test Loss: {eval[0]:.4f}")
        #print(f"Test Accuracy: {1 - eval[0]:.4f}")
        print(f"Inference Time: {elapsed_time:.4f} seconds\n")
    except:
        print(f"Sample {i} skipped due to error")
        print("")
        continue
end = time.time()
'''

average_all_sample_time = (all_time/num_test)*1000 
average_all_loss = all_loss/num_test 
average_all_accuracy = all_accuracy/num_test 

print("")
print("")
print("Average evaluation time for " + str(num_samples) + " samples test for " + str(num_test) + " times is: " + str(average_all_sample_time) + " ms")
print("Average loss for " + str(num_samples) + " samples test for " + str(num_test) + " times is: " + str(average_all_loss))
print("Average accuracy for " + str(num_samples) + " samples test for " + str(num_test) + " times is: " + str(average_all_accuracy))
'''
# Calculate average inference time
 SOLO CODE
average_inference_time = elapsed_time / num_samples
print("Elapsed Time for " + str(num_samples) + " samples is " + str(elapsed_time) +" seconds ")
print('Average Inference Time for each sample:', average_inference_time, 'seconds')
'''
