import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Change to your file paths
concepts_2011_path = './data/concepts_2011.txt'
trainset_gt_annotations_path = './data/trainset_gt_annotations.txt'
imagecleff2011_path = './data/imagecleffeats/imageclef2011_feats/'

# Obtain indices of spring, summer, winter, autumn from concepts_2011.txt
concepts = np.asarray(pd.read_fwf(concepts_2011_path))

concepts_dict = {}
for i in concepts:
    i = i[0].split('\t')
    concepts_dict[i[1]] = int(i[0])
    
spring_index = concepts_dict['Spring']
summer_index = concepts_dict['Summer']
winter_index = concepts_dict['Winter']
autumn_index = concepts_dict['Autumn']

# Load data and split by spring, summer, winter, autumn
season_data = np.asarray(pd.read_csv(trainset_gt_annotations_path, delim_whitespace=True, header=None))

seasons = [spring, summer, winter, autumn]

spring = []
summer = []
winter = []
autumn = []

for i in season_data:   
    if i[spring_index+1] == 1:
        spring.append(i)
    if i[summer_index+1] == 1:
        summer.append(i)
    if i[winter_index+1] == 1:
        winter.append(i)
    if i[autumn_index+1] == 1:
        autumn.append(i)

spring = np.asarray(spring)[:,0]
summer = np.asarray(summer)[:,0]
winter = np.asarray(winter)[:,0]
autumn = np.asarray(autumn)[:,0]

total_count = len(spring) + len(summer) + len(winter) + len(autumn)

spring_imageclef2011 = []
summer_imageclef2011 = []
winter_imageclef2011 = []
autumn_imageclef2011 = []

imageclef2011_directory = imagecleff2011_path
for i in spring:
    i = np.load(imageclef2011_directory + i + '_ft.npy')
    spring_imageclef2011.append(i)
for i in summer:
    i = np.load(imageclef2011_directory + i + '_ft.npy')
    summer_imageclef2011.append(i)
for i in winter:
    i = np.load(imageclef2011_directory + i + '_ft.npy')
    winter_imageclef2011.append(i)
for i in autumn:
    i = np.load(imageclef2011_directory + i + '_ft.npy')
    autumn_imageclef2011.append(i)

spring_imageclef2011 = np.asarray(spring_imageclef2011)
summer_imageclef2011 = np.asarray(summer_imageclef2011)
winter_imageclef2011 = np.asarray(winter_imageclef2011)
autumn_imageclef2011 = np.asarray(autumn_imageclef2011)

# Train, validation, test split
spring_train = spring_imageclef2011[0:int(len(spring_imageclef2011)*0.6)]
spring_validation = spring_imageclef2011[int(len(spring_imageclef2011)*0.6):int(len(spring_imageclef2011)*0.75)]
spring_test = spring_imageclef2011[int(len(spring_imageclef2011)*0.75):]
np.save('spring_train.npy', spring_train)
np.save('spring_validation.npy', spring_validation)
np.save('spring_test.npy', spring_test)

summer_train = summer_imageclef2011[0:int(len(summer_imageclef2011)*0.6)]
summer_validation = summer_imageclef2011[int(len(summer_imageclef2011)*0.6):int(len(summer_imageclef2011)*0.75)]
summer_test = summer_imageclef2011[int(len(summer_imageclef2011)*0.75):]
np.save('summer_train.npy', summer_train)
np.save('summer_validation.npy', summer_validation)
np.save('summer_test.npy', summer_test)

winter_train = winter_imageclef2011[0:int(len(winter_imageclef2011)*0.6)]
winter_validation = winter_imageclef2011[int(len(winter_imageclef2011)*0.6):int(len(winter_imageclef2011)*0.75)]
winter_test = winter_imageclef2011[int(len(winter_imageclef2011)*0.75):]
np.save('winter_train.npy', winter_train)
np.save('winter_validation.npy', winter_validation)
np.save('winter_test.npy', winter_test)

autumn_train = autumn_imageclef2011[0:int(len(autumn_imageclef2011)*0.6)]
autumn_validation = autumn_imageclef2011[int(len(autumn_imageclef2011)*0.6):int(len(autumn_imageclef2011)*0.75)]
autumn_test = autumn_imageclef2011[int(len(autumn_imageclef2011)*0.75):]
np.save('autumn_train.npy', autumn_train)
np.save('autumn_validation.npy', autumn_validation)
np.save('autumn_test.npy', autumn_test)

# Preparing training/val for the 4 SVMs
training_size = len(spring_train) + len(summer_train) + len(winter_train) + len(autumn_train)
validation_size = len(spring_validation) + len(summer_validation) + len(winter_validation) + len(autumn_validation)
val_x = np.concatenate((spring_validation, summer_validation, winter_validation, autumn_validation),axis=0)

# Spring
spring_svm_train_x = np.concatenate((spring_train, summer_train, winter_train, autumn_train),axis=0)
spring_svm_train_y = np.zeros(training_size)
spring_svm_train_y[:len(spring_train)] = 1.0
spring_svm_val_x = val_x
spring_svm_val_y = np.zeros(validation_size)
spring_svm_val_y[:len(spring_validation)] = 1.0

# Summer
summer_svm_train_x = np.concatenate((summer_train, spring_train, winter_train, autumn_train),axis=0)
summer_svm_train_y = np.zeros(training_size)
summer_svm_train_y[:len(summer_train)] = 1.0
summer_svm_val_x = val_x
summer_svm_val_y = np.zeros(validation_size)
summer_svm_val_y[len(spring_validation):len(spring_validation)+len(summer_validation)] = 1.0

# Winter
winter_svm_train_x = np.concatenate((winter_train, spring_train, summer_train, autumn_train),axis=0)
winter_svm_train_y = np.zeros(training_size)
winter_svm_train_y[:len(winter_train)] = 1.0
winter_svm_val_x = val_x
winter_svm_val_y = np.zeros(validation_size)
winter_svm_val_y[len(spring_validation)+len(summer_validation):len(summer_validation)+len(spring_validation)+len(winter_validation)] = 1.0


# Autumn
autumn_svm_train_x = np.concatenate((autumn_train, spring_train, summer_train, winter_train),axis=0)
autumn_svm_train_y = np.zeros(training_size)
autumn_svm_train_y[:len(autumn_train)] = 1.0
autumn_svm_val_x = val_x
autumn_svm_val_y = np.zeros(validation_size)
autumn_svm_val_y[len(spring_validation)+len(summer_validation)+len(winter_validation):len(summer_validation)+len(spring_validation)+len(winter_validation)+len(autumn_validation)] = 1.0

# val_y
val_y = np.zeros(validation_size)
val_y[0:len(spring_validation)] = 0.0
val_y[len(spring_validation):len(spring_validation)+len(summer_validation)] = 1.0
val_y[len(spring_validation)+len(summer_validation):len(spring_validation)+len(summer_validation)+len(winter_validation)] = 2.0
val_y[len(spring_validation)+len(summer_validation)+len(winter_validation):len(summer_validation)+len(spring_validation)+len(winter_validation)+len(autumn_validation)] = 3.0

def run_validation(c):
    svm_spring = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_spring.fit(spring_svm_train_x, spring_svm_train_y)

    svm_summer = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_summer.fit(summer_svm_train_x, summer_svm_train_y)

    svm_winter = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_winter.fit(winter_svm_train_x, winter_svm_train_y)

    svm_autumn = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_autumn.fit(autumn_svm_train_x, autumn_svm_train_y)

    svm_spring_proba = svm_spring.predict_proba(val_x)[:,1]
    svm_summer_proba = svm_summer.predict_proba(val_x)[:,1]
    svm_winter_proba = svm_winter.predict_proba(val_x)[:,1]
    svm_autumn_proba = svm_autumn.predict_proba(val_x)[:,1]

    val_pred = []
    for i in range(len(svm_spring_proba)):
        index = np.argmax([svm_spring_proba[i], svm_summer_proba[i], svm_winter_proba[i], svm_autumn_proba[i]])    
        val_pred.append(index)
    val_pred = np.asarray(val_pred)
    val_vanilla_acc = sum(val_y==val_pred)/len(val_y)

    spring_count = sum(val_y == 0)
    spring_match = sum((val_y == val_pred) & (val_pred == 0)) 
    spring_acc = spring_match/spring_count
    
    summer_count = sum(val_y == 1)
    summer_match = sum((val_y == val_pred) & (val_pred == 1))
    summer_acc = summer_match/summer_count    
    
    winter_count = sum(val_y == 2)
    winter_match = sum((val_y == val_pred) & (val_pred == 2))
    winter_acc = winter_match/winter_count
    
    autumn_count = sum(val_y == 3)
    autumn_match = sum((val_y == val_pred) & (val_pred == 3))
    autumn_acc = autumn_match/autumn_count
    
    val_class_wise_avg_acc = (spring_acc + summer_acc + winter_acc + autumn_acc)/4.0
    
    return val_vanilla_acc, val_class_wise_avg_acc


def select_best_c():
    set_c = [0.01, 0.1, pow(0.1,0.5), 1.0, pow(10,0.5), 10, pow(1000,0.5)]
    val_vanilla_acc_list = []
    val_class_wise_avg_acc_list = []
    for c in set_c:
        val_vanilla_acc, val_class_wise_avg_acc = run_validation(c)
        val_vanilla_acc_list.append(val_vanilla_acc)
        val_class_wise_avg_acc_list.append(val_class_wise_avg_acc)
    print(val_vanilla_acc_list)
    print(val_class_wise_avg_acc_list)  
    
    return set_c[np.argmax(val_class_wise_avg_acc_list)], max(val_vanilla_acc_list), max(val_class_wise_avg_acc_list)

best_c, best_vanilla_acc_val, best_class_wise_avg_acc_val = select_best_c() 
print("The best c value is: " + str(best_c))
print("The best vanilla accuracy for validation set is: " + str(best_vanilla_acc_val))
print("The best class wise average accuracy for validation set is: " + str(best_class_wise_avg_acc_val))

# Preparing the training/test for the 4 SVMs
spring_size = len(spring_train) + len(spring_validation)
summer_size = len(summer_train) + len(summer_validation)
winter_size = len(winter_train) + len(winter_validation)
autumn_size = len(autumn_train) + len(autumn_validation)
training_size = spring_size + summer_size + winter_size + autumn_size
test_size = len(spring_test) + len(summer_test) + len(winter_test) + len(autumn_test)
test_x = np.concatenate((spring_test, summer_test, winter_test, autumn_test),axis=0)

# Spring
spring_svm_train_x = np.concatenate((spring_train, spring_validation, summer_train, summer_validation, winter_train, winter_validation, autumn_train, autumn_validation),axis=0)
spring_svm_train_y = np.zeros(training_size)
spring_svm_train_y[:spring_size] = 1.0
spring_svm_test_x = test_x
spring_svm_test_y = np.zeros(test_size)
spring_svm_test_y[:len(spring_test)] = 1.0

# Summer
summer_svm_train_x = np.concatenate((summer_train, summer_validation, spring_train, spring_validation, winter_train, winter_validation, autumn_train, autumn_validation),axis=0)
summer_svm_train_y = np.zeros(training_size)
summer_svm_train_y[:summer_size] = 1.0
summer_svm_test_x = test_x
summer_svm_test_y = np.zeros(test_size)
summer_svm_test_y[len(spring_test):len(spring_test)+len(summer_test)] = 1.0

# Winter
winter_svm_train_x = np.concatenate((winter_train, winter_validation, spring_train, spring_validation, summer_train, summer_validation, autumn_train, autumn_validation),axis=0)
winter_svm_train_y = np.zeros(training_size)
winter_svm_train_y[:winter_size] = 1.0
winter_svm_test_x = test_x
winter_svm_test_y = np.zeros(test_size)
winter_svm_test_y[len(spring_test)+len(summer_test):len(summer_test)+len(spring_test)+len(winter_test)] = 1.0

# Autumn
autumn_svm_train_x = np.concatenate((autumn_train, autumn_validation, spring_train, spring_validation, summer_train, summer_validation, winter_train, winter_validation),axis=0)
autumn_svm_train_y = np.zeros(training_size)
autumn_svm_train_y[:autumn_size] = 1.0
autumn_svm_test_x = test_x
autumn_svm_test_y = np.zeros(test_size)
autumn_svm_test_y[len(spring_test)+len(summer_test)+len(winter_test):len(summer_test)+len(spring_test)+len(winter_test)+len(autumn_test)] = 1.0

# test_y
test_y = np.zeros(test_size)
test_y[0:len(spring_test)] = 0.0
test_y[len(spring_test):len(spring_test)+len(summer_test)] = 1.0
test_y[len(spring_test)+len(summer_test):len(spring_test)+len(summer_test)+len(winter_test)] = 2.0
test_y[len(spring_test)+len(summer_test)+len(winter_test):len(summer_test)+len(spring_test)+len(winter_test)+len(autumn_test)] = 3.0

def run_test(c):
    svm_spring = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_spring.fit(spring_svm_train_x, spring_svm_train_y)

    svm_summer = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_summer.fit(summer_svm_train_x, summer_svm_train_y)

    svm_winter = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_winter.fit(winter_svm_train_x, winter_svm_train_y)

    svm_autumn = SVC(C=c, kernel='linear', class_weight='balanced', probability=True)
    svm_autumn.fit(autumn_svm_train_x, autumn_svm_train_y)

    svm_spring_proba = svm_spring.predict_proba(test_x)[:,1]
    svm_summer_proba = svm_summer.predict_proba(test_x)[:,1]
    svm_winter_proba = svm_winter.predict_proba(test_x)[:,1]
    svm_autumn_proba = svm_autumn.predict_proba(test_x)[:,1]

    test_pred = []
    for i in range(len(svm_spring_proba)):
        index = np.argmax([svm_spring_proba[i], svm_summer_proba[i], svm_winter_proba[i], svm_autumn_proba[i]])    
        test_pred.append(index)
    test_pred = np.asarray(test_pred)
    test_vanilla_acc = sum(test_y==test_pred)/len(test_y)
       
    spring_count = sum(test_y == 0)
    spring_match = sum((test_y == test_pred) & (test_pred == 0)) 
    spring_acc = spring_match/spring_count
    
    summer_count = sum(test_y == 1)
    summer_match = sum((test_y == test_pred) & (test_pred == 1))
    summer_acc = summer_match/summer_count    
    
    winter_count = sum(test_y == 2)
    winter_match = sum((test_y == test_pred) & (test_pred == 2))
    winter_acc = winter_match/winter_count
    
    autumn_count = sum(test_y == 3)
    autumn_match = sum((test_y == test_pred) & (test_pred == 3))
    autumn_acc = autumn_match/autumn_count
    
    test_class_wise_avg_acc = (spring_acc + summer_acc + winter_acc + autumn_acc)/4.0
    
    return test_vanilla_acc, test_class_wise_avg_acc

test_vanilla_acc, test_class_wise_avg_acc = run_test(best_c)

print("The vanilla accuracy for test set is: " + str(test_vanilla_acc))
print("The class wise average accuracy for test set is: " + str(test_class_wise_avg_acc))