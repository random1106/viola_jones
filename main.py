import numpy as np
import pickle
from viola_jones import ViolaJones
import time

start_time = time.time()

def viola_train(t):
    model = ViolaJones(t)
    with open("train.pkl", "rb") as file:
        training_datas = pickle.load(file)
    train_pos_num = 0
    train_neg_num = 0
    for data in training_datas:
        if data[1] == 1:
            train_pos_num += 1
        elif data[1] == 0:
            train_neg_num += 1
        else:
            print("train_data has label outside of (0, 1)")
    start_time = time.time()
    model.train(training_datas, train_pos_num, train_neg_num)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"time used to train the model is {time_elapsed / 60.0} minutes")
    evaluate(model, training_datas)
    with open('./my_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    

def viola_test():
    with open("test.pkl", "rb") as file:
        testing_datas = pickle.load(file)
    with open('./my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    evaluate(model, testing_datas)


def evaluate(model, data):
    data_num = 0
    correct_num = 0
    for x, y in data:
        data_num += 1
        if model.classify(x) == y:
            correct_num += 1
    print(f"Accuracy: {correct_num} / {data_num} = {correct_num * 1.0 / data_num}")

        
with open('./my_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("train.pkl", "rb") as file:
        training_datas = pickle.load(file)  

with open("test.pkl", "rb") as file:
        testing_datas = pickle.load(file)
        
# viola_train(10)
# viola_test()


