import os
import Utils
from keras import models
import joblib

h5filename = 'models/CnnModel.h5'
model = Utils.load_model(h5filename)
model.summary()
score = Utils.evaluate_model(model)
print('the accuracy on test datadry without data augment hit %2.2f%%' % (score[1] * 100))
