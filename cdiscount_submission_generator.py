from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from tqdm import *
import bson
from densenet import DenseNet, DenseNetImageNet169, DenseNetImageNet121, preprocess_input
import os
import io
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import TFOptimizer, Adam

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


def getDenseNetModel(num_classes, target_size):
    input_size = target_size + (3,)
    # model = DenseNet(input_shape=input_size, depth=121, nb_dense_block=4,
    # growth_rate=24, nb_layers_per_block=[6, 12, 24, 16], bottleneck=True,
    # reduction=0.5, classes=num_classes)
    model = DenseNetImageNet169(input_shape=input_size, include_top=False)
    for layer in model.layers:
        layer.trainable = False
    x = model.layers[-1].output
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model.input, x, name='densenet')
    model.load_weights("denset_model_only_top.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        lr=0.0001, epsilon=0.1), metrics=["accuracy"])
    # model.summary()
    return model

if __name__ == '__main__':
    data_dir = "/run/media/vigi99/HDD/Cdiscount/data/"
    bson_dir = "/run/media/vigi99/SDD/Kaggle/Cdiscount/data/"
    num_test_products = 1768182
    target_size = (180, 180)
    num_classes = 5270
    test_bson_path = os.path.join(bson_dir, "test.bson")
    submission_path = "data/submission_densenet161_only_top.csv.gz"
    submission_df = pd.read_csv(submission_path)
    total_done = submission_df.shape[0]
    ids_not_done = submission_df[submission_df.done == 0]._id.values
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    data = bson.decode_file_iter(open(test_bson_path, "rb"))
    cat2idx, idx2cat = make_category_tables(data_dir)
    model = getDenseNetModel(num_classes, target_size)
    started = False

    with tqdm(total=total_done) as pbar:
        for c, d in enumerate(data):
            product_id = d["_id"]
            if product_id in ids_not_done:
                started = True
                num_imgs = len(d["imgs"])

                batch_x = np.zeros((num_imgs,) + target_size + (3,), dtype = K.floatx())

                for i in range(num_imgs):
                    bson_img=d["imgs"][i]["picture"]

                    # Load and preprocess the image.
                    img=load_img(io.BytesIO(bson_img), target_size = target_size)
                    x=img_to_array(img)
                    x=test_datagen.random_transform(x)
                    x=test_datagen.standardize(x)

                    # Add the image to the batch.
                    batch_x[i]=x

                prediction=model.predict(batch_x, batch_size = num_imgs)
                avg_pred=prediction.mean(axis = 0)
                cat_idx=np.argmax(avg_pred)

                submission_df.iloc[c]["category_id"]=idx2cat[cat_idx]
                submission_df.iloc[c]["done"]=1
                pbar.update()
            else:
                pbar.update()
            if c % 50000 == 0 and c != 0 and started:
                submission_df.to_csv(submission_path, compression = "gzip", index = False)
                print('50000 examples predicted and file saved.')

    submission_df.drop('done', inplace=True, axis=1)
    submission_df.to_csv(submission_path, compression = "gzip", index = False)