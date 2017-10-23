import time
from keras.callbacks import Callback

class KerasOutputCallBack(Callback): 
    train_start = 0
    epoch_start = 0
    
    def __init__(self, skip_epochs=0):
        self.emod = skip_epochs+1
    
    def on_train_begin(self, logs):
        self.train_start = time.time()
    
    def on_train_end(self, logs):
        n = time.time() - self.epoch_start
        seconds = int(time.time() - self.train_start + .5)
        print('Training round took {}:{:02}'.format(seconds // 60, seconds % 60))
        
    def on_epoch_begin(self, epoch, logs):
        if epoch % self.emod is not 0:
            return
        self.epoch_start = time.time()
        print("Epoch " + str(epoch+1), end='', flush=True)
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.emod is not 0:
            return
        n = time.time() - self.epoch_start
        print("({}s) - train_loss: {:.5f}; train_acc: {:.3f}%; val_loss: {:.5f}; val_acc: {:.3f}%".format(
            int(time.time() - self.epoch_start + .5),
            logs['loss'], 100*logs['acc'], logs['val_loss'], 100*logs['val_acc']))