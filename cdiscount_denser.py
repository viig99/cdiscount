import os
import sys
import math
import io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct
import matplotlib.pyplot as plt
import keras
import threading
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from custom_callbacks import ModelBatchCheckpoint
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.models import Model
from keras.layers.core import Dense
import keras.backend as K
import tensorflow as tf
from collections import defaultdict
from tqdm import *
from bsoniterator import BSONIterator
# from resnext import ResNext
# from densenet import DenseNet, DenseNetImageNet169, DenseNetImageNet121, DenseNetImageNet161
# from densenet import preprocess_input
# from se_resnet import preprocess_input
from se_densenet import preprocess_input
from keras.optimizers import TFOptimizer, Adam, SGD, Nadam, NadamAccum, SGDAccum
# from adam_accumulate import Adam_accumulate
from se_resnet import SEResNet101
from se_densenet import SEDenseNetImageNet161

def preprocess_input_partial(x):
    # preprocess_input(x, data_format=None, mode='tf')
    return preprocess_input(x)

def schedule(index):
    lr = 0.01
    if index >=2 and index < 4:
        lr = 0.001
    elif index >= 4:
        lr = 0.0001
    return lr

def formOrLoadCategoryTable(data_dir):
    categories_path = os.path.join(data_dir, "category_names.csv")
    categories_formed_path = os.path.join(data_dir, "categories.csv")
    if os.path.exists(categories_formed_path):
        categories_df = pd.read_csv(
            categories_formed_path, index_col='category_id')
    else:
        categories_df = pd.read_csv(categories_path, index_col="category_id")
        categories_df["category_idx"] = pd.Series(
            range(len(categories_df)), index=categories_df.index)
        categories_df.to_csv(categories_formed_path)
    return categories_df


def make_category_tables(data_dir):
    cat2idx = {}
    idx2cat = {}
    categories_df = formOrLoadCategoryTable(data_dir)
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


def read_bson(data_dir, num_records, with_categories, type_file='train'):
    rows = {}
    bson_path = os.path.join(data_dir, type_file + ".bson")
    csv_path = os.path.join(
        '/home/vigi99/kaggle/Cdiscount/data/', type_file + "_offsets.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col='product_id')
    else:
        with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
            offset = 0
            while True:
                item_length_bytes = f.read(4)
                if len(item_length_bytes) == 0:
                    break

                length = struct.unpack("<i", item_length_bytes)[0]

                f.seek(offset)
                item_data = f.read(length)
                assert len(item_data) == length

                item = bson.BSON.decode(item_data)
                product_id = item["_id"]
                num_imgs = len(item["imgs"])

                row = [num_imgs, offset, length]
                if with_categories:
                    row += [item["category_id"]]
                rows[product_id] = row

                offset += length
                f.seek(offset)
                pbar.update()

        columns = ["num_imgs", "offset", "length"]
        if with_categories:
            columns += ["category_id"]

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "product_id"
        df.columns = columns
        df.sort_index(inplace=True)
        df.to_csv(csv_path)
    return df


def formOrGetValSet(data_dir, num_records, split_percentage=0.2, drop_percentage=0.85):
    image_path = '/home/vigi99/kaggle/Cdiscount/data/'
    train_image_path = os.path.join(image_path, 'train_images_10.csv')
    val_image_path = os.path.join(image_path, 'val_images_10.csv')
    if os.path.exists(train_image_path) and os.path.exists(val_image_path):
        train_images_df = pd.read_csv(train_image_path)
        val_images_df = pd.read_csv(val_image_path)
    else:
        df = read_bson(data_dir, num_records, True, 'train')
        train_images_df, val_images_df = make_val_set(
            data_dir, df, split_percentage=split_percentage, drop_percentage=drop_percentage)
        train_images_df.to_csv(train_image_path, index=False)
        val_images_df.to_csv(val_image_path, index=False)
    return train_images_df, val_images_df


def make_val_set(data_dir, df, split_percentage=0.2, drop_percentage=0):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    cat2idx, idx2cat = make_category_tables(data_dir)
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(
                    product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation
            # set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(
                    product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()

    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df


def getDenseNetModel(num_classes, target_size):
    input_size = target_size + (3,)
    model = SEDenseNetImageNet161(input_shape=input_size, include_top=False)
    for layer in model.layers[:-15]:
        layer.trainable = False
    x = model.layers[-1].output
    x = Dense(num_classes, activation='softmax')(x)
    optimizers = Adam_accumulate(lr=0.01, epsilon=0.1, accum_iters=4)
    # optimizers = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model = Model(model.input, x, name='densenet')
    model.load_weights("densenet161_model_0.01.5000.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=[categorical_accuracy, top_k_categorical_accuracy])
    model.summary()
    return model

def getSEDenseNetModelScratch(num_classes, target_size):
    input_size = target_size + (3,)
    model = SEDenseNetImageNet161(input_shape=input_size, include_top=True, classes=num_classes)
    optimizers = SGDAccum(lr=1e-4, decay=1e-4, momentum=0.9, nesterov=True, accum_iters=32)
    # optimizers = Adam(lr=1e-4)
    # optimizers = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=False)
    model.load_weights("sedensenet161_model.09-2.57.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=[categorical_accuracy, top_k_categorical_accuracy])
    # model.summary()
    return model

def getSEResnetModel(num_classes, target_size):
    input_size = target_size + (3,)
    model = SEResNet101(input_shape=input_size, include_top=True, classes=num_classes)
    optimizers = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=[categorical_accuracy, top_k_categorical_accuracy])
    model.summary()
    return model

if __name__ == '__main__':
    data_dir = "/run/media/vigi99/HDD/Cdiscount/data/"
    bson_dir = "/run/media/vigi99/SDD/Kaggle/Cdiscount/data/"
    num_train_products = 7069896
    num_test_products = 1768182
    train_bson_path = os.path.join(bson_dir, "train.bson")
    test_bson_path = os.path.join(data_dir, "test.bson")

    train_offsets_df = read_bson(
        bson_dir, num_train_products, True, type_file='train')
    train_images_df, val_images_df = formOrGetValSet(
        bson_dir, num_train_products)
    train_bson_file = open(train_bson_path, "rb")

    num_classes = 5270
    batch_size = 32
    num_train_images = len(train_images_df)
    num_val_images = len(val_images_df)
    target_size = (160, 160)
    lock = threading.Lock()

    # Tip: use ImageDataGenerator for data augmentation and preprocessing.
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_partial
                                       # rotation_range=15,
                                       # width_shift_range=5. / 32,
                                       # height_shift_range=5. / 32,
                                       # shear_range=5./32,
                                       # zoom_range=5./32,
                                       # channel_shift_range=5./32
                                       )
    train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                             num_classes, train_datagen, lock, batch_size=batch_size, shuffle=True, target_size=target_size)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_partial)
    val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                           num_classes, val_datagen, lock, batch_size=batch_size, shuffle=True, target_size=target_size)

    # print('Val data process started')
    # val_data = [next(val_gen) for x in range(500)]
    # val_data_x = np.array([x[0] for x in val_data])
    # val_data_y = np.array([x[1] for x in val_data])
    # val_data_x = val_data_x.reshape(val_data_x.shape[0] * val_data_x.shape[1], *val_data_x.shape[2:])
    # val_data_y = val_data_y.reshape(val_data_y.shape[0] * val_data_y.shape[1], *val_data_y.shape[2:])
    # val_data =  (val_data_x, val_data_y)
    # print('Val data process ended')

    # model = getDenseNetModel(num_classes, target_size)
    # model_name = 'densenet161_model'
    model = getSEDenseNetModelScratch(num_classes, target_size)
    model_name = 'sedensenet161_model'

    # logfile = "./logs161"
    logfile = "./logsse161"

    checkpointer = ModelBatchCheckpoint(
        filepath = model_name + '.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True, save_weights_only=True, period=1)
    tensorboard = TensorBoard(log_dir=logfile, histogram_freq=0, write_images=False, write_graph=False)
    reducelr = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, min_lr=1e-4, epsilon=0.01, factor=0.1)
    schedulelr = LearningRateScheduler(schedule)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1, mode='auto')
    model.fit_generator(train_gen,
                        initial_epoch=8,
                        steps_per_epoch=num_train_images // batch_size,
                        epochs=10,
                        validation_data=val_gen,
                        validation_steps=1000,
                        workers=10,
                        verbose=1,
                        max_q_size=10,
                        callbacks=[earlystopping, reducelr, checkpointer, tensorboard])
    model.save_weights(model_name + ".hdf5")
