import numpy as np
import pandas as pd
import math
import random

dataset = pd.read_csv('subset_16P.csv', encoding='mac-roman') # reading dataset
non_shuffled_dataset_array = dataset.to_numpy() # this function creates an numpy array with dataset
dataset_array = []
'''
##### shuffling dataset #####
ct = 9999
while ct > -1:
    x = random.randint(0,ct)
    dataset_array.append(non_shuffled_dataset_array[x])
    non_shuffled_dataset_array = np.delete(non_shuffled_dataset_array,x,axis=0)
    ct -=1
dataset_array = np.array(dataset_array)
'''


normalization_max = [] # this list for saving maximum numbers  of every features
normalization_min = [] # this list for saving minimum numbers of every features
for i in range(1,61): ### 0th index is keeping IDs but we don't want to normalize this it doesn't effect personality. 
    min = 9999999 
    max = -9999999
    ### here is the classical code of finding minimum and maximum numbers.
    for data in dataset_array:
        number = data[i]
        if number < min:
            min = number
        if number > max:
            max = number
    normalization_max.append(max) # we are saving maximum number of every features in the normalization_max
    normalization_min.append(min) # we are saving minimum number of every features in the normalization_min
for i in range(1,61): ### in this loop we are doing normalization process for every number.
    for data in dataset_array:
        data[i] = (data[i] - normalization_min[i-1]) / (normalization_max[i-1]-normalization_min[i-1])



training_array = dataset_array[2000:10000] # validation 1
### if we want to choose test_data from middle of dataset we can skip chosen rows with skiprows method
# training_array = pd.read_csv('subset_16P.csv' , encoding = 'mac-roman' , skiprows = range(2001,4001)).to_numpy() # validation 2
# training_array = pd.read_csv('subset_16P.csv', encoding = 'mac-roman', skiprows = range(4001,6001)).to_numpy() # validation 3
# training_array = pd.read_csv('subset_16P.csv', encoding = 'mac-roman', skiprows = range(6001,8001)).to_numpy() # validation 4
# training_array = dataset_array[0:8000] # validation 5

test_array = dataset_array[0:2000] # validation 1
# test_array = dataset_array[2000:4000] # validation 2
# test_array = dataset_array[4000:6000] # validation 3
# test_array = dataset_array[6000:8000] # validation 4
# test_array = dataset_array[8000:10000] # validation 5
predictions = [] # this list for saving our predictions

##### 1 KNN #####

for test in test_array:
    distance = 999999999999
    personality = ''
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += abs(test[i]-training[i]) ** 2 ### euclidean distance formula
        total = math.sqrt(total)
        if total < distance:
            distance = total # classical finding minimum distance loop
            personality = training[61]
    predictions.append(personality) # after end of loop we are saving our predict in predictions list for every test row


'''
##### 3 KNN #####
for test in test_array:
    distances = [99999,999999,9999999] ### for 3 KNN I save the nearest 3 distances
    predicted_personalities = ['','',''] ### and of course 3 nearest neighbours personalities
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += abs(test[i]-training[i]) ** 2 ### euclidean distance formula
        total = math.sqrt(total)
        for i in range(3):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    isPredicted = False
    for i in range(2): 
    ### if there is a at least two same personalities in the 3 nearest neighbors algorithm assumes that personality as predicted.
        if distances.count(distances[i]) > 1: 
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    ### otherwise, algorithm assumes nearest neighbors as predicted personality/
    if not isPredicted:
        predictions.append(predicted_personalities[0])

'''

'''
##### 5 KNN #####
for test in test_array:
    distances = [99999,999999,9999999,99999999,999999999] ### for 5 KNN I save the nearest 5 distances
    predicted_personalities = ['','','','',''] ### and  of course 5 nearest neighbours personalities
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += abs(test[i]-training[i]) ** 2 ### euclidean distance formula
        total = math.sqrt(total)
        for i in range(5):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    isPredicted = False
    ### if there is a at least three same personalities in the 5 nearest neighbors algorithm assumes that personality as predicted.
    for i in range(3):
        if distances.count(distances[i]) > 2:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    ### otherwise, we are looking for two same personalities in the 5 nearest neighbors.
    if not isPredicted:
        for i in range(4):
            if distances.count(distances[i]) > 1:
                predictions.append(predicted_personalities[i])
                isPredicted = True
                break
    ### otherwise algorith assumes the nearest one as predicted personality.
    if not isPredicted:
        predictions.append(predicted_personalities[0])
'''

'''
##### 7 KNN #####
for test in test_array:
    distances = [999,9999,99999,999999,9999999,99999999,999999999] ### for 7 KNN I save the nearest 7 distances
    predicted_personalities = ['','','','','','',''] ### and of course 7 nearest neighbours personalities
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += abs(test[i]-training[i]) ** 2 ### euclidean distance formula
        total = math.sqrt(total)
        for i in range(7):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    ### if there is a at least four same personalities in the 7 nearest neighbors algorithm assumes that personality as predicted.
    isPredicted = False
    for i in range(4):
        if distances.count(distances[i]) > 3:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    ### otherwise, we are looking for three same personalities in the 7 nearest neighbors.
    if not isPredicted:
        for i in range(5):
            if distances.count(distances[i]) > 2:
                predictions.append(predicted_personalities[i])
                isPredicted = True
                break
        if not isPredicted:
        ### otherwise, we are looking for two same personalities in the 7 nearest neighbors.
            for i in range(6):
                if distances.count(distances[i]) > 1:
                    predictions.append(predicted_personalities[i])
                    isPredicted=True
                    break
    ### otherwise algorith assumes the nearest one as predicted personality.
    if not isPredicted:
        predictions.append(predicted_personalities[0])

'''


'''
##### 9 KNN #####
for test in test_array:
    ### for 9 KNN I save the nearest 9 distances
    distances = [999,9999,99999,999999,9999999,99999999,999999999,9999999999,99999999999]
    predicted_personalities = ['','','','','','','','',''] # 9 nearest neighbours personalities
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += abs(test[i]-training[i]) ** 2 ### euclidean distance formula
        total = math.sqrt(total)
        for i in range(9):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    ### if there is a at least five same personalities in the 9 nearest neighbors algorithm assumes that personality as predicted.
    isPredicted = False
    for i in range(5):
        if distances.count(distances[i]) > 4:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    ### otherwise, we are looking for four same personalities in the 9 nearest neighbors.
    if not isPredicted:
        for i in range(6):
            if distances.count(distances[i]) > 3:
                predictions.append(predicted_personalities[i])
                isPredicted = True
                break
        ### otherwise, we are looking for three same personalities in the 9 nearest neighbors.
        if not isPredicted:
            for i in range(7):
                if distances.count(distances[i]) > 2:
                    predictions.append(predicted_personalities[i])
                    isPredicted=True
                    break
            ### otherwise, we are looking for two same personalities in the 9 nearest neighbors.
            if not isPredicted:
                for i in range(8):
                    if distances.count(distances[i]) > 1:
                        predictions.append(predicted_personalities[i])
                        isPredicted=True
                        break
    ### otherwise algorith assumes the nearest one as predicted personality.
    if not isPredicted:
        predictions.append(predicted_personalities[0])
'''


''' for weighted knn algorithms I made an algorithm which calculates the difference sum with this formula
    (abs(test[i]-training[i]) * (1+i/60)) ** 2. This formula makes more important last features because when
    the index of feature coming bigger the affect on the sum is will be bigger.'''
'''
##### 1 weighted KNN #####
for test in test_array:
    distance = 999999999999
    personality = ''
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += (abs(test[i]-training[i]) * (1+i/60)) ** 2
        total = math.sqrt(total)
        if total < distance:
            distance = total
            personality = training[61]
    predictions.append(personality)

'''

'''
##### 3 weighted KNN #####
for test in test_array:
    distances = [99999,999999,9999999]
    predicted_personalities = ['','','']
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += (abs(test[i]-training[i]) * (1+i/60)) ** 2
        total = math.sqrt(total)
        for i in range(3):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    isPredicted = False
    for i in range(2):
        if distances.count(distances[i]) > 1:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    if not isPredicted:
        predictions.append(predicted_personalities[0])
'''


'''
##### 5 weighted KNN #####

for test in test_array:
    distances = [99999,999999,9999999,99999999,999999999]
    predicted_personalities = ['','','','','']
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += (abs(test[i]-training[i]) * (1+i/60)) ** 2
        total = math.sqrt(total)
        for i in range(5):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    isPredicted = False
    for i in range(3):
        if distances.count(distances[i]) > 2:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    if not isPredicted:
        for i in range(4):
            if distances.count(distances[i]) > 1:
                predictions.append(predicted_personalities[i])
                isPredicted = True
                break
    if not isPredicted:
        predictions.append(predicted_personalities[0])
'''


'''
##### 7 Weighted KNN #####
for test in test_array:
    distances = [999,9999,99999,999999,9999999,99999999,999999999]
    predicted_personalities = ['','','','','','','']
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += (abs(test[i]-training[i]) * (1+i/60)) ** 2
        total = math.sqrt(total)
        for i in range(7):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    isPredicted = False
    for i in range(4):
        if distances.count(distances[i]) > 3:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    if not isPredicted:
        for i in range(5):
            if distances.count(distances[i]) > 2:
                predictions.append(predicted_personalities[i])
                isPredicted = True
                break
        if not isPredicted:
            for i in range(6):
                if distances.count(distances[i]) > 1:
                    predictions.append(predicted_personalities[i])
                    isPredicted=True
                    break
    if not isPredicted:
        predictions.append(predicted_personalities[0])
'''

'''
##### 9 Weighted KNN #####            
for test in test_array:
    distances = [999,9999,99999,999999,9999999,99999999,999999999,9999999999,99999999999]
    predicted_personalities = ['','','','','','','','','']
    for training in training_array:
        total = 0
        for i in range(1,61):
            total += (abs(test[i]-training[i]) * (1+i/60)) ** 2
        total = math.sqrt(total)
        for i in range(9):
            if total < distances[i]:
                distances[i] = total
                predicted_personalities[i] = training[61]
                break
    isPredicted = False
    for i in range(5):
        if distances.count(distances[i]) > 4:
            predictions.append(predicted_personalities[i])
            isPredicted = True
            break
    if not isPredicted:
        for i in range(6):
            if distances.count(distances[i]) > 3:
                predictions.append(predicted_personalities[i])
                isPredicted = True
                break
        if not isPredicted:
            for i in range(7):
                if distances.count(distances[i]) > 2:
                    predictions.append(predicted_personalities[i])
                    isPredicted=True
                    break
            if not isPredicted:
                for i in range(8):
                    if distances.count(distances[i]) > 1:
                        predictions.append(predicted_personalities[i])
                        isPredicted=True
                        break
    if not isPredicted:
        predictions.append(predicted_personalities[0])
            
'''

personalities = ['ESTJ','ENTJ','ESFJ','ENFJ','ISTJ','ISFJ','INTJ','INFJ','ESTP','ESFP','ENTP','ENFP','ISTP','ISFP','INTP','INFP']
TP = 0
FP = 0
TN = 0
FN = 0
### Here is the calculating loop for TP, FP, TN and FN count for sum of all personality class.
for personality in personalities:
    for i in range(len(test_array)):
        if test_array[i][61] == personality and predictions[i] == personality:
            TP +=1
        elif test_array[i][61] != personality and predictions[i] == personality:
            print(test_array[i][0])
            FP +=1
        elif test_array[i][61] == personality and predictions[i] != personality:
            FN +=1
        else:
            TN +=1

accuracy = (TP+TN) / (TP+FP+FN+TN)

# for 1 KNN results [0.997125,0.99775,0.9965625,0.996625,0.996875]  avg 0.9969875
# for 3 KNN results [0.9968125,0.9975625,0.996375,0.99625,0.99675] avg 0.99675
# for 5 KNN results [0.99625,0.9959375,0.9956875,0.9955625,0.9958125] avg 0.99585
# for 7 KNN results [0.995,0.9941875,0.994125,0.994125,0.995] avg 0.9944875
# for 9 KNN results [0.99325,0.9925,0.99225,0.99225,0.9929375] avg 0.9926375
# for 1 Weighted KNN results [0.996875,0.99625,0.996125,0.99625,0.9966875] avg 0.9964375
# for 3 Weighted KNN results [0.996875,0.9968125,0.996,0.9959375,0.9964375] avg 0.9964125
# for 5 Weighted KNN results [0.996875,0.9968125,0.996,0.996,0.994125] avg 0.9959625
# for 7 Weighted KNN results [0.996125,0.996875,0.9965,0.998,0.994125] avg 0.996325
# for 9 Weighted KNN results [0.996125,0.996875,0.9965,0.994125,0.994125] avg 0.99555
# for normalized 1 KNN results [0.9951875,0.99625,0.9950625,0.99475,0.9951875] avg 0.9952875
# for normalized 3 KNN results [0.9950625,0.9958125,0.995125,0.9946875,0.995125] avg 0.9951625
# for normalized 5 KNN results [0.99375,0.9946875,0.9946875,0.993875,0.9940625] avg 0.9942125
# for normalized 7 KNN results [0.9919375,0.99325,0.99275,0.9924375,0.9923125] avg 0.9925375
# for normalized 9 KNN results [0.9919375,0.99325,0.99275,0.9924375,0.9923125] avg 0.9895125
# for normalized 1 Weighted KNN results [0.9946875,0.9945625,0.998875,0.992125,0.99475] avg 0.995
# for normalized 3 Weighted KNN results [0.991625,0.991625,0.992625,0.992125,0.992375] avg 0.992075
# for normalized 5 Weighted KNN results [0.991625,0.99175,0.99175,0.992125,0.992125] avg 0.991875
# for normalized 7 Weighted KNN results [0.99225,0.99125,0.9925,0.992,0.991125] avg 0.991825
# for normalized 9 Weighted KNN results [0.991,0.990875,0.993125,0.9925,0.9925] avg 0.992
precision = TP / (TP+FP)
# for 1 KNN results [0.977,0.982,0.9725,0.973,0.975] avg 0.9759
# for 3 KNN results [0.9745,0.9805,0.971,0.97,0.974] avg 0.974
# for 5 KNN results [0.97,0.9675,0.9655,0.9645,0.9665] avg 0.9668
# for 7 KNN results [0.96,0.9535,0.953,0.953,0.96] avg 0.9559
# for 9 KNN results [0.946,0.94,0.938,0.938,0.9435] avg 0.9411
# for 1 Weighted KNN results [0.975,0.97,0.969,0.97,0.9735] avg 0.9715
# for 3 Weighted KNN results [0.975,0.9745,0.968,0.9675,0.9715] avg 0.9713
# for 5 Weighted KNN results [0.975,0.9745,0.968,0.968,0.953] avg 0.9677
# for 7 Weighted KNN results [0.969,0.975,0.972,0.984,0.953] avg 0.9706
# for 9 Weighted KNN results [0.969,0.975,0.972,0.953,0.953] avg 0.9644
# for Normalized 1 KNN results [0.9615,0.97,0.9605,0.958,0.9615] avg 0.9623
# for Normalized 3 KNN results [0.9605,0.9665,0.961,0.9575,0.961] avg 0.9613
# for Normalized 5 KNN results [0.95,0.9575,0.9575,0.951,0.9525] avg 0.9537
# for Normalized 7 KNN results [0.9355,0.946,0.942,0.9395,0.9385] avg 0.9403
# for Normalized 9 KNN results [0.9095,0.9235,0.921,0.9165,0.91] avg 0.9161
# for Normalized 1 Weighted KNN results [0.9575,0.9565,0.991,0.937,0.958] avg 0.96
# for Normalized 3 Weighted KNN results [0.933,0.933,0.941,0.937,0.939] avg 0.9366
# for Normalized 5 Weighted KNN results [0.933,0.934,0.934,0.937,0.937] avg 0.935
# for Normalized 7 Weighted KNN results [0.938,0.93,0.94,0.936,0.929] avg 0.9346
# for Normalized 9 Weighted KNN results [0.928,0.927,0.945,0.94,0.94] avg 0.936
recall = TP/(TP+FN)
# for 1 KNN results [0.977,0.982,0.9725,0.973,0.975] avg 0.9759
# for 3 KNN results [0.9745,0.9805,0.971,0.97,0.974] avg 0.974
# for 5 KNN results [0.97,0.9675,0.9655,0.9645,0.9665] avg 0.9668
# for 7 KNN results [0.96,0.9535,0.953,0.953,0.96] avg 0.9559
# for 9 KNN results [0.946,0.94,0.938,0.938,0.9435] avg 0.9411
# for 1 Weighted KNN results [0.975,0.97,0.969,0.97,0.9735] avg 0.9715
# for 3 Weighted KNN results [0.975,0.9745,0.968,0.9675,0.9715] avg 0.9713
# for 5 Weighted KNN results [0.975,0.9745,0.968,0.968,0.953] avg 0.9677
# for 7 Weighted KNN results [0.969,0.975,0.972,0.984,0.953] avg 0.9706
# for 9 Weighted KNN results [0.969,0.975,0.972,0.953,0.953] avg 0.9644
# for Normalized 1 KNN results [0.9615,0.97,0.9605,0.958,0.9615] avg 0.9623
# for Normalized 3 KNN results [0.9605,0.9665,0.961,0.9575,0.961] avg 0.9613
# for Normalized 5 KNN results [0.95,0.9575,0.9575,0.951,0.9525] avg 0.9537
# for Normalized 7 KNN results [0.9355,0.946,0.942,0.9395,0.9385] avg 0.9403
# for Normalized 9 KNN results [0.9095,0.9235,0.921,0.9165,0.91] avg 0.9161
# for Normalized 1 Weighted KNN results [0.9575,0.9565,0.991,0.937,0.958] avg 0.96
# for Normalized 3 Weighted KNN results [0.933,0.933,0.941,0.937,0.939] avg 0.9366
# for Normalized 5 Weighted KNN results [0.933,0.934,0.934,0.937,0.937] avg 0.935
# for Normalized 7 Weighted KNN results [0.938,0.93,0.94,0.936,0.929] avg 0.9346
# for Normalized 9 Weighted KNN results [0.928,0.927,0.945,0.94,0.94] avg 0.936
print(TP, TN, FP, FN)
print(accuracy, precision, recall)


##### PART 2 ######
''' 

dataset2 = pd.read_csv('energy_efficiency_data.csv', encoding='mac-roman') # reading dataset
non_shuffled_dataset_array = dataset2.to_numpy() # this function creates an numpy array with dataset


##### shuffling dataset #####
dataset_array2 = []
ct = 767

while ct > -1:
    x = random.randint(0,ct)
    dataset_array2.append(non_shuffled_dataset_array[x])
    non_shuffled_dataset_array = np.delete(non_shuffled_dataset_array,x,axis=0)
    ct -=1
dataset_array = np.array(dataset_array2)

'''

''' 
normalization_max = [] # this list for saving maximum numbers  of every features
normalization_min = [] # this list for saving minimum numbers  of every features

### here is the classical code of finding minimum and maximum numbers.

for i in range(0,10):
    min = 9999999
    max = -9999999
    for data in dataset_array2:
        number = data[i]
        if number < min:
            min = number
        if number > max:
            max = number
    normalization_max.append(max) # we are saving maximum number of every features in the normalization_max
    normalization_min.append(min) # we are saving minimum number of every features in the normalization_min
for i in range(0,10): ### in this loop we are doing normalization process for every number.
    for data in dataset_array2:
        data[i] = (data[i] - normalization_min[i]) / (normalization_max[i]-normalization_min[i])
'''


'''
predictionsOfHeatingLoads = []
predictionsOfCoolingLoads = []
# training_array = dataset_array2[154:768] # validation 1
# training_array = pd.read_csv('subset_16P.csv' , encoding = 'mac-roman' , skiprows = range(155,309)).to_numpy() # validation 2
# training_array = pd.read_csv('subset_16P.csv', encoding = 'mac-roman', skiprows = range(309,463)).to_numpy() # validation 3
# training_array = pd.read_csv('subset_16P.csv', encoding = 'mac-roman', skiprows = range(463,617)).to_numpy() # validation 4
# training_array = dataset_array2[0:616] # validation 5



# test_array = dataset_array2[0:154] # validation 1
# test_array = dataset_array2[154:308] # validation 2
# test_array = dataset_array2[308:462] # validation 3
# test_array = dataset_array2[462:616] # validation 4
# test_array = dataset_array2[616:768] # validation 5
'''

'''
##### 1 KNN #####
for test in test_array:
    distance = 999999999999
    Heating_Load = 0
    Cooling_Load = 0
    for training in training_array:
        total = 0
        for i in range(0,8):
            total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        if total < distance:
            distance = total  #classical finding minimum distance loop
            Heating_Load = training[8]  
            Cooling_Load = training[9]
    predictionsOfHeatingLoads.append(Heating_Load) # assumes nearest neighbors Heating Load as predicted Heating Load
    predictionsOfCoolingLoads.append(Cooling_Load) # assumes nearest neighbors Cooling Load as predicted Cooling Load
'''


"""
##### 3 KNN #####
for test in test_array:
    distances = [99999,999999,9999999] ### for 3 KNN I save the nearest 3 distances
    Heating_Loads = [0,0,0] ### for 3 KNN I save the nearest 3 neighbors Heating Load data
    Cooling_Loads = [0,0,0] ### for 3 KNN I save the nearest 3 neighbors Cooling Load data
    for training in training_array:
        total = 0
        for i in range(0,8):
            total += abs(test[i]-training[i]) ** 2 ### euclidian formula
        total = math.sqrt(total)
        ### loop for finding the nearest 3 neighbor and their Heating Load and Cooling Load
        for i in range(3):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
                
    ### Predicting Heating Load and Colling Load as the nearest 3 neighbors Heating Load and Colling Load mean.
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
"""

'''
##### 5 KNN #####
for test in test_array:
    distances = [99999,999999,9999999,9999999,9999999] ### for 5 KNN I save the nearest 5 distances
    Heating_Loads = [0,0,0,0,0] ### for 5 KNN I save the nearest 5 neighbors Heating Load data
    Cooling_Loads = [0,0,0,0,0] ### for 5 KNN I save the nearest 5 neighbors Coolong Load data
    for training in training_array:
        total = 0
        for i in range(0,8):
            total += abs(test[i]-training[i]) ** 2 ### euclidian formula
        total = math.sqrt(total)
        ### loop for finding the nearest 5 neighbor and their Heating Load and Cooling Load
        for i in range(5):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
    ### Predicting Heating Load and Colling Load as the nearest 5 neighbors Heating Load and Colling Load mean.
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''

'''
##### 7 KNN #####
for test in test_array:
    distances = [99999,999999,9999999,9999999,9999999,9999999,9999999] ### for 7 KNN I save the nearest 7 distances
    Heating_Loads = [0,0,0,0,0,0,0] # for 7 KNN I save the nearest 7 neighbors Heating Load data
    Cooling_Loads = [0,0,0,0,0,0,0] # for 7 KNN I save the nearest 7 neighbors Cooling Load data
    for training in training_array:
        total = 0
        for i in range(0,8):
            total += abs(test[i]-training[i]) ** 2 ### euclidian formula
        total = math.sqrt(total)
        ### loop for finding the nearest 5 neighbor and their Heating Load and Cooling Load
        for i in range(7):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
    ### Predicting Heating Load and Colling Load as the nearest 7 neighbors Heating Load and Colling Load mean.
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''

'''
##### 9 KNN #####
for test in test_array:
    ### for 9 KNN I save the nearest 9 distances
    distances = [99999,999999,9999999,9999999,9999999,9999999,9999999,9999999,9999999]
    Heating_Loads = [0,0,0,0,0,0,0,0,0] # for 9 KNN I save the nearest 9 neighbors Heating Load data
    Cooling_Loads = [0,0,0,0,0,0,0,0,0] # for 9 KNN I save the nearest 9 neighbors Cooling Load data
    for training in training_array:
        total = 0
        for i in range(0,8):
            total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        ### loop for finding the nearest 5 neighbor and their Heating Load and Cooling Load
        for i in range(9):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
     ### Predicting Heating Load and Colling Load as the nearest 9 neighbors Heating Load and Colling Load mean.
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''


'''
 in the weighted knn algorithm, algorithm makes some multiplication such as 
 orientation * 0.1 , glazing area * 10 , glazing area distribution * 10 
 because when I observe some datas from dataframe I think glazing area
 and glazing are distribution is the most affecting features and orientation
 is not so much important. I believe this algorithm will work better than
 Unweighted knn. '''


'''
##### Weighted 1 KNN #####
for test in test_array:
    distance = 999999999999
    Heating_Load = 0
    Cooling_Load = 0
    for training in training_array:
        total = 0
        for i in range(0,8):    
            if i == 5:
                total += abs((test[i]-training[i])*0.1) ** 2
            elif i == 6 or i == 7:
                total += abs((test[i]-training[i])*10) ** 2
            else:
                total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        if total < distance:
            distance = total
            Heating_Load = training[8]
            Cooling_Load = training[9]
    predictionsOfHeatingLoads.append(Heating_Load)
    predictionsOfCoolingLoads.append(Cooling_Load)

'''


'''
##### Weighted 3 KNN #####
for test in test_array:
    distances = [99999,999999,9999999]
    Heating_Loads = [0,0,0]
    Cooling_Loads = [0,0,0]
    for training in training_array:
        total = 0
        for i in range(0,8):
            if i == 5:
                total += abs((test[i]-training[i])*0.1) ** 2
            elif i == 6 or i == 7:
                total += abs((test[i]-training[i])*10) ** 2
            else:
                total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        for i in range(3):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''


'''
##### Weighted 5 KNN #####
for test in test_array:
    distances = [99999,999999,9999999,9999999,9999999]
    Heating_Loads = [0,0,0,0,0]
    Cooling_Loads = [0,0,0,0,0]
    for training in training_array:
        total = 0
        for i in range(0,8):
            if i == 5:
                total += abs((test[i]-training[i])*0.1) ** 2
            elif i == 6 or i == 7:
                total += abs((test[i]-training[i])*10) ** 2
            else:
                total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        for i in range(5):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''

'''
##### Weighted 7 KNN #####
for test in test_array:
    distances = [99999,999999,9999999,9999999,9999999,9999999,9999999]
    Heating_Loads = [0,0,0,0,0,0,0]
    Cooling_Loads = [0,0,0,0,0,0,0]
    for training in training_array:
        total = 0
        for i in range(0,8):
            if i == 5:
                total += abs((test[i]-training[i])*0.1) ** 2
            elif i == 6 or i == 7:
                total += abs((test[i]-training[i])*10) ** 2
            else:
                total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        for i in range(7):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''

'''
##### Weighted 9 KNN #####
for test in test_array:
    distances = [99999,999999,9999999,9999999,9999999,9999999,9999999,9999999,9999999]
    Heating_Loads = [0,0,0,0,0,0,0,0,0]
    Cooling_Loads = [0,0,0,0,0,0,0,0,0]
    for training in training_array:
        total = 0
        for i in range(0,8):
            if i == 5:
                total += abs((test[i]-training[i])*0.1) ** 2
            elif i == 6 or i == 7:
                total += abs((test[i]-training[i])*10) ** 2
            else:
                total += abs(test[i]-training[i]) ** 2
        total = math.sqrt(total)
        for i in range(9):
            if total < distances[i]:
                distances[i] = total
                Heating_Loads[i] = training[8]
                Cooling_Loads[i] = training[9]
                break
    predictionsOfHeatingLoads.append(sum(Heating_Loads)/len(Heating_Loads))
    predictionsOfCoolingLoads.append(sum(Cooling_Loads)/len(Cooling_Loads))
'''


'''
MAE = 0 # mean absolute error for heating loads predictions
MAE2 = 0 # mean absolute error for cooling loads predictions
for i in range(len(predictionsOfHeatingLoads)):   
    MAE += (abs(test_array[i][8]-predictionsOfHeatingLoads[i])) / len(predictionsOfHeatingLoads) # MAE formula
for i in range(len(predictionsOfHeatingLoads)):   
    MAE2 += (abs(test_array[i][9]-predictionsOfCoolingLoads[i])) / len(predictionsOfCoolingLoads) # MAE formula

print(math.sqrt(MAE))
print(MAE2)
'''


##### MAE FOR 1 KNN [2.724740259740259,2.6664935064935054,2.4735064935064925,2.5226623376623376,2.510064935064935]
##### MAE2 FOR 1 KNN [2.20525974025974,2.0863636363636378,2.0222727272727266,2.056818181818183,2.177987012987015]
##### MAE FOR 3 KNN [1.5933549783549779,1.6779653679653677,1.582272727272727,1.7483982683982677,1.471103896103896]
##### MAE2 FOR 3 KNN [1.4476190476190467,1.47965367965368,1.4574025974025968,1.5398268398268407,1.333809523809523]
##### MAE FOR 5 KNN [1.9561168831168831,1.9855714285714294,1.7907012987012982,1.9008181818181817,1.939454545454545]
##### MAE2 FOR 5 KNN [1.610948051948052,1.7827922077922074,1.5937922077922082,1.5042727272727268,1.7818311688311692]
##### MAE FOR 7 KNN [2.1542949907235625,1.8531261595547315,2.0631168831168827,2.153645640074211,2.014035250463823]
##### MAE2 FOR 7 KNN [1.9992949907235624,1.7400556586270872,1.8090074211502778,1.90791280148423,1.7092393320964756]
##### MAE FOR 9 KNN [2.106067821067821,2.4753463203463215,2.148585858585857,1.73998556998557,1.8688744588744586]
##### MAE2 FOR 9 KNN [1.8665079365079364,2.0407720057720047,1.7991919191919195,1.6711616161616165,1.7410245310245307]
##### MAE FOR WEIGHTED 1 KNN [0.38493506493506513,0.3500649350649353,0.40266233766233767,0.3974025974025976,0.43363636363636343]
##### MAE2 FOR WEIGHTED 1 KNN [1.2775974025974026,1.1511038961038964,1.3544155844155847,1.5403246753246758,1.0861038961038965]
##### MAE FOR WEIGHTED 3 KNN [1.071969696969697,1.116645021645022,1.0548917748917754,1.0233982683982688,1.2862554112554114]
##### MAE2 FOR WEIGHTED 3 KNN [1.3745670995670989,1.6310389610389608,1.2807792207792206,1.3307359307359306,1.5198701298701294]
##### MAE FOR WEIGHTED 5 KNN [1.3934675324675327,1.6748701298701292,1.5712337662337665,1.6029610389610387,1.4236103896103898]
##### MAE2 FOR WEIGHTED 5 KNN [1.534051948051948,1.9147792207792205,1.7062207792207804,1.7090129870129875,1.6506623376623386]
##### MAE FOR WEIGHTED 7 KNN [1.707560296846011,1.6486363636363632,1.7580705009276452,1.8614192949907231,1.400695732838591]
##### MAE2 FOR WEIGHTED 7 KNN [1.7881910946196666,1.6868831168831169,1.7351020408163274,1.8122727272727281,1.6455473098330242]
##### MAE FOR WEIGHTED 9 KNN [1.6071139971139976,1.968189033189034,2.0801948051948065,1.968059163059163,1.7861111111111114]
##### MAE2 FOR WEIGHTED 9 KNN [1.72479797979798,1.908607503607504,1.6866378066378067,1.9294877344877344,1.9664502164502158]

### NORMALIZED
##### MAE FOR 1 KNN [0.3197387963090244,0.2814220490792094,0.29093200285832577,0.28696943773510675,0.304095799578659]
##### MAE2 FOR 1 KNN [0.10828223755775598,0.08691295238561605,0.08912350778766076,0.0879202940878136,0.10260369848304134]
##### MAE FOR 3 KNN [0.24052108187067142,0.24400073056586252,0.23434692138204086,0.23420742618911336,0.23400925001596268]
##### MAE2 FOR 3 KNN [0.06423610503869055,0.0656002135937498,0.060511622321479525,0.06254962382083307,0.06593482825640115]
##### MAE FOR 5 KNN [0.24719949468400626,0.23372652742905622,0.23741425223916546,0.2308686336174364,0.24117844412237688]
##### MAE2 FOR 5 KNN [0.06801655118380136,0.05838699409935606,0.06558284161300591,0.05782526119181115,0.06442929545541991]
##### MAE FOR 7 KNN [0.21997111265976907,0.24249528300977058,0.22713630343296762,0.25036295321540986,0.24104027386638954]
##### MAE2 FOR 7 KNN [0.05696427384704096,0.06351849066634954,0.057123419845131224,0.06911008655843401,0.06316422218080485]
##### MAE FOR 9 KNN [0.23201985542224887,0.2500174581905727,0.23559783576934828,0.23570267304928197,0.22233842173713195]
##### MAE2 FOR 9 KNN [0.05663654357432974,0.06414827354768102,0.06468070338256172,0.060573998225492975,0.05557867933305583]
##### MAE FOR WEIGHTED 1 KNN [0.1137452951892052,0.10842619851356251,0.09875925432175144,0.101279831916566,0.10600127261749565]
##### MAE2 FOR WEIGHTED 1 KNN [0.034534681585583836,0.035221982434479084,0.030183525066369128,0.036376228134913834,0.037324108694967835]
##### MAE FOR WEIGHTED 3 KNN [0.20724055676411127,0.21836359118444754,0.20433809711095238,0.18953187690576384,0.20126337271902373]
##### MAE2 FOR WEIGHTED 3 KNN [0.062036625731750984,0.06116161421844159,0.0468530481996682,0.04853195103666421,0.043021302245649116]
##### MAE FOR WEIGHTED 5 KNN [0.23445497307063412,0.2413562270288965,0.2513704893850615,0.25034876737765555,0.2321119780395086]
##### MAE2 FOR WEIGHTED 5 KNN [0.06805397672620943,0.06397739077512843,0.06905327368564644,0.06776016872973507,0.06671190377088575]
##### MAE FOR WEIGHTED 7 KNN [0.3010909054734314,0.3099355214906255,0.320281587568455,0.29489093713135467,0.31345447627900114]
##### MAE2 FOR WEIGHTED 7 KNN [0.08582841113332436,0.10082785899414735,0.10311261590550576,0.08612746570116425,0.1006459778534728]
##### MAE FOR WEIGHTED 9 KNN [0.38138697389909165,0.3612222699359098,0.3843009872387372,0.3846895401452676,0.36613042510841665]
##### MAE2 FOR WEIGHTED 9 KNN [0.14158533509462684,0.12901649327719894,0.14435960544228793,0.15173220411572147,0.12755503167568882]

