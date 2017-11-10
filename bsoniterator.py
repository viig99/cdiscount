from keras.preprocessing.image import Iterator, load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import bson
import io

class BSONIterator(Iterator):

    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180), with_labels=True,
                 batch_size=32, shuffle=False, seed=None):
        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        print("Found %d images belonging to %d classes." %
              (self.samples, self.num_class))
        super(BSONIterator, self).__init__(
            self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) +
                           self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros(
                (len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"].astype(np.int))
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            try:
                item = bson.BSON.decode(item_data)
                img_idx = image_row["img_idx"]
                bson_img = item["imgs"][img_idx]["picture"]

                # Preprocess the image.
                img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
                x = img_to_array(img)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)

                # Add the image and the label to the batch (one-hot encoded).
                batch_x[i] = x
                if self.with_labels:
                    batch_y[i, image_row["category_idx"]] = 1
            except (bson.errors.InvalidBSON):
                pass
        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
