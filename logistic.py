from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

    p1 = np.random.permutation(train.shape[0])
    train_p1 = train[p1]
    label_p1 = label[p1].reshape((label.shape[0],1))
    p2 = np.random.permutation(train.shape[0])
    train_p2 = train[p2]
    label_p2 = label[p2].reshape((label.shape[0],1))
    p3 = np.random.permutation(train.shape[0])
    train_p3 = train[p3]
    label_p3 = label[p3].reshape((label.shape[0],1))
    train_tot = np.vstack((train_p1,train_p2,train_p3))
    label_tot = np.vstack((label_p1,label_p2,label_p3)).reshape((2970,))
    print(train_tot.shape)
    print(label_tot.shape)

    param = {"C":[10,100,200,1000,1200,1500,2000],"tol":[1e-4,5e-4,2e-4,1e-5]}
    logis = LogisticRegression(multi_class="multinomial",solver="lbfgs")
    grid = GridSearchCV(logis,param,scoring="neg_log_loss",refit="True",n_jobs=8,cv=5)
    grid.fit(train_tot,label_tot)
    print("-----------")
    print(str(grid.best_params_))
    output_prob = grid.predict_proba(test)
    test_id = test_set.pop("id")
    result = pandas.DataFrame(output_prob,index=test_id,columns=encoder.classes_)
    result.to_csv(OUTPUT_PATH)