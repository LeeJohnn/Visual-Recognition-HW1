# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
import pandas as pd
from utils import load_model
import glob 
import os
from tqdm import trange
if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model()
    model.load_weights('models/model.96-0.89.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    test_path = '../testing_data/testing_data/'

    test_images = glob.glob(os.path.join(test_path, '*.jpg'))
    test_images = sorted(test_images)
    # test_images = [f for f in os.listdir(test_path) if
    #                os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]

    # num_samples = 20
    # samples = random.sample(test_images, num_samples)
    # results = []

    _ids, _preds = [], []

    for i in trange(len(test_images)):
        filename = test_images[i]

        # print('Start processing image: {}'.format(filename))
        bgr_img = cv.imread(filename)
        bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)

        _id = filename.split('/')[-1].split('.')[0]
        _pred = class_names[class_id][0][0]

        _ids.append(str(_id))
        _preds.append(_pred)
        # text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
        # results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
        # cv.imwrite('images/{}_out.png'.format(i), bgr_img)

    d = {'id': _ids, 'label': _preds}
    # print(_ids)
    # print(_preds)

    # print(d)
    df = pd.DataFrame.from_dict(d)
    df.to_csv('./output.csv', index=False)
    # with open('results.json', 'w') as file:
    #     json.dump(results, file, indent=4)

    K.clear_session()
