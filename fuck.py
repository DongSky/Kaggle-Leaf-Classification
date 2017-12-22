from tensorflow.contrib.learn import DNNClassifier
from tensorflow.contrib.layers import real_valued_column
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas
import argparse
import numpy as np

TRAIN_DATASET = "./train.csv"
TEST_DATASET = "./test.csv"
OUTPUT_PATH = "./result.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store",dest="train",default="./train.csv",type=str)
    parser.add_argument("--test",action="store",dest="test",default="./test.csv",type=str)
    parser.add_argument("--out",action="store",dest="output",default="./result.csv",type=str)
    parse = parser.parse_args()
    TRAIN_DATASET = parse.train
    TEST_DATASET = parse.test
    OUTPUT_PATH = parse.output
    np.random.seed(19260817)

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

    feature_columns = [real_valued_column("",dimension=192)]
    classifier = DNNClassifier(feature_columns=feature_columns,n_classes=99,hidden_units=[1024,512,256],optimizer=tf.train.AdamOptimizer)
    classifier.fit(x=train,y=label,steps=1000)
    output = classifier.predict(test)
    output_prob = classifier.predict_proba(test)
    test_id = test_set.pop("id")
    result = pandas.DataFrame(output_prob,index=test_id,columns=encoder.classes_)
    result.to_csv(OUTPUT_PATH)