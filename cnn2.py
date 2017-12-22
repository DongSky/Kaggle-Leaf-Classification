from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge, MaxPooling2D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, img_to_array, array_to_img, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import pandas
import argparse
import os
import random
import numpy as np

TRAIN_DATASET = "./train.csv"
TEST_DATASET = "./test.csv"
OUTPUT_PATH = "./result.csv"
RANDOM_SEED = random.randint(1,10000)

def load_train_csv():
    train = pandas.read_csv(TRAIN_DATASET)
    ID = train.pop("id")
    Y = train.pop("species")
    Y = LabelEncoder().fit(Y).transform(Y)
    X = StandardScaler().fit(train).transform(train)
    return ID, X, Y

def load_test_csv():
    test = pandas.read_csv(TEST_DATASET)
    ID = test.pop("id")
    X = StandardScaler().fit(test).transform(test)
    return ID, X

def resize(img,dim=96):
    max_axis = 1 if img.size[1] > img.size[0] else 0
    scale = dim / float(img.size[max_axis])
    return img.resize((int(scale * img.size[0]), int(scale * img.size[1])))

def load_img_data(ids,dim=96):
    X = np.empty((len(ids),dim,dim,1))
    for i, ide in enumerate(ids):
        x = resize(load_img(os.path.join("images", str(ide) + '.jpg'),grayscale=True),dim=dim)
        x = img_to_array(x)
        height = x.shape[0]
        width = x.shape[1]
        h1 = int((dim - height) / 2)
        h2 = h1 + height
        w1 = int((dim - width) / 2)
        w2 = w1 + width
        X[i, h1:h2, w1:w2, 0:1] = x
    return np.around(X / 255.0)

def load_train_data():
    ID, X_num_train, Y = load_train_csv()
    X_img_train = load_img_data(ID)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1,random_state=RANDOM_SEED)
    train_index, test_index = next(sss.split(X_num_train, Y))
    X_num_tr, X_img_tr, Y_tr = X_num_train[train_index], X_img_train[train_index], Y[train_index]
    X_num_val, X_img_val, Y_val = X_num_train[test_index], X_img_train[test_index], Y[test_index]
    return (X_num_tr, X_img_tr, Y_tr), (X_num_val, X_img_val, Y_val)

def load_test_data():
    ID, X_num_test = load_test_csv()
    X_img_test = load_img_data(ID)
    return ID, X_num_test, X_img_test

class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        with self.lock:
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

def generator(imgen, X):
    while True:
        for i in range(X.shape[0]):
            batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]
            yield [batch_img, x], batch_y

if __name__ == "__main__":

    (X_num_tr, X_img_tr, Y_tr), (X_num_val, X_img_val, Y_val) = load_train_data()
    Y_tr_cat = to_categorical(Y_tr)
    Y_val_cat = to_categorical(Y_val)

    imgen = ImageDataGenerator2(rotation_range=20,zoom_range=0.2,horizontal_flip=True,vertical_flip=True,fill_mode="nearest")
    imgen_train = imgen.flow(X_img_tr, Y_tr_cat, seed=RANDOM_SEED)

    image = Input(shape=(96,96,1),name="image")
    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))(x)
    x = (Convolution2D(16, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = (Convolution2D(64, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = (Convolution2D(128, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = Flatten()(x)
    numerical = Input(shape=(192,),name="numerical")
    concat = merge([x, numerical],mode="concat", concat_axis=1)
    x = Dense(100,activation="relu")(concat)
    x = Dropout(0.5)(x)
    out = Dense(99,activation="softmax")(x)
    model = Model(input=[image,numerical],output=out)
    model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
    model_file = "cnn.mod"
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit_generator(generator(imgen_train,X_num_tr),samples_per_epoch=X_num_tr.shape[0],nb_epoch=120,validation_data=([X_img_val,X_num_val],Y_val_cat),nb_val_samples=X_num_val.shape[0],verbose=0,callbacks=[checkpoint])
    model = load_model(model_file)
    labels = sorted(pandas.read_csv(TRAIN_DATASET).species.unique())
    index, X_num_test, X_img_test = load_test_data()
    result = model.predict([X_img_test,X_num_test])
    result = pandas.DataFrame(result,index=index,columns=labels)
    result.to_csv(OUTPUT_PATH)
    pass