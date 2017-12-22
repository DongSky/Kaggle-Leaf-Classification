from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import pandas
import argparse
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

TRAIN_DATASET = "./train.csv"
TEST_DATASET = "./test.csv"
OUTPUT_PATH = "./result.csv"
#RANDOM_SEED = random.randint(1,10000)
RANDOM_SEED = 12345
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store",dest="train",default="./train.csv",type=str)
    parser.add_argument("--test",action="store",dest="test",default="./test.csv",type=str)
    parser.add_argument("--out",action="store",dest="output",default="./result.csv",type=str)
    parse = parser.parse_args()
    TRAIN_DATASET = parse.train
    TEST_DATASET = parse.test
    OUTPUT_PATH = parse.output

    train_set = pandas.read_csv(TRAIN_DATASET)
    test_set = pandas.read_csv(TEST_DATASET)
    encoder = LabelEncoder().fit(train_set["species"])
    train = train_set.drop(["species","id"],axis=1).values
    label = encoder.transform(train_set["species"])
    test = test_set.drop(["id"],axis=1).values
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    scaler = StandardScaler().fit(test)
    test = scaler.transform(test)

    p1 = np.random.permutation(train.shape[0])
    train_p1 = train[p1]
    label_p1 = label[p1].reshape((label.shape[0],1))
    p2 = np.random.permutation(train.shape[0])
    train_p2 = train[p2]
    label_p2 = label[p2].reshape((label.shape[0],1))
    p3 = np.random.permutation(train.shape[0])
    train_p3 = train[p3]
    label_p3 = label[p3].reshape((label.shape[0],1))
    p4 = np.random.permutation(train.shape[0])
    train_p4 = train[p4]
    label_p4 = label[p4].reshape((label.shape[0],1))
    p5 = np.random.permutation(train.shape[0])
    train_p5 = train[p5]
    label_p5 = label[p5].reshape((label.shape[0],1))
    # train_tot = np.vstack((train_p1,train_p2,train_p3,train_p4,train_p5))
    # label_tot = np.vstack((label_p1,label_p2,label_p3,label_p4,label_p5))
    train_tot = np.vstack((train_p1,train_p2))
    label_tot = np.vstack((label_p1,label_p2))
    label_tot = to_categorical(label_tot)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state=RANDOM_SEED)
    train_index, val_index = next(iter(sss.split(train_tot, label_tot)))
    train_tot, train_val = train_tot[train_index], train_tot[val_index]
    label_tot, label_val = label_tot[train_index], label_tot[val_index]

    print(train_tot.shape)
    print(label_tot.shape)

    model = Sequential()
    model.add(Dense(768,input_dim=192,init="glorot_normal",activation="tanh"))
    model.add(Dropout(0.4))
    model.add(Dense(768,input_dim=768,activation="tanh"))
    model.add(Dropout(0.4))
    model.add(Dense(99,input_dim=768,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics = ["accuracy"])
    early_stopping = EarlyStopping(monitor="val_loss",patience=300)
    xjb_fit = model.fit(train_tot,label_tot,batch_size=64,nb_epoch=2500,callbacks=[early_stopping],validation_data=(train_val,label_val))
    output_prob = model.predict_proba(test)
    test_id = test_set.pop("id")
    result = pandas.DataFrame(output_prob,index=test_id,columns=encoder.classes_)
    result.to_csv(OUTPUT_PATH)
    print('val_acc: ',max(xjb_fit.history['val_acc']))
    print('val_loss: ',min(xjb_fit.history['val_loss']))
    print('train_acc: ',max(xjb_fit.history['acc']))
    print('train_loss: ',min(xjb_fit.history['loss']))
    print("train/val loss ratio: ", min(xjb_fit.history['loss'])/min(xjb_fit.history['val_loss']))
