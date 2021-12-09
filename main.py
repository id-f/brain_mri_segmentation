
import os
from flask import Flask, request, redirect, render_template, flash, session

from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import segmentation_models as sm

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import uuid

smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)


image_size = 256

UPLOAD_FOLDER = "/tmp/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])

app = Flask(__name__)
app.secret_key = "super secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            flash(file.filename)

            filename = secure_filename(file.filename)
            flash('test')
            flash(filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            img = cv2.resize(img ,(image_size, image_size))
            img = img / 255
            img = img[np.newaxis, :, :, :]
            pred=model.predict(img)
            
            raw_filename = str(uuid.uuid1()) + ".jpg"

            raw_img_url = os.path.join("static", raw_filename)
            plt.imshow(np.squeeze(img))
            plt.savefig(raw_img_url)                        
            
            pred_filename="pred_"+ raw_filename

            pred_img_url = os.path.join("static", pred_filename)
            plt.imshow(np.squeeze(pred) > .5)
            plt.savefig(pred_img_url)

            return render_template("result.html", uploaded=raw_filename, predicted=pred_filename)
    else:
        return render_template("index.html",raw_img_url="")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)