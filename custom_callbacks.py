from keras.callbacks import ModelCheckpoint

class ModelBatchCheckpoint(ModelCheckpoint):
    def on_batch_end(self, batch, logs=None):
        if batch % 5000 == 0 and batch != 0:
            # model_name = 'densenet161_model'
            model_name = 'sedensenet161_model'
            filepath = model_name + '_0.001.{batch:02d}.hdf5'.format(**logs)
            self.model.save_weights(filepath, overwrite=True)
            print('\nBatch %02d: saving model to %s' % (batch, filepath))
