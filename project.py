import tensorflow as tf
print(tf.__version__)

import os

from google.colab import drive
drive.mount('/content/drive')

checkpoint_dir = '/content/drive/My Drive/checkpoints'

os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.weights.h5')

!kaggle datasets download -d vipoooool/new-plant-diseases-dataset

import matplotlib.pyplot as plt
import numpy as np
import datetime
import zipfile

def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
    filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall()
    zip_ref.close()

def model_checkpoint_cb(dirname):
  cb=tf.keras.callbacks.ModelCheckpoint(dirname,verbose=1,save_best_only=False,save_weights_only=True,)
  return cb

unzip_data("/content/new-plant-diseases-dataset.zip")

from sklearn.model_selection import train_test_split

import pandas as pd
import os

train="/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
valid="/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
test="/content/test"

tra=tf.keras.preprocessing.image_dataset_from_directory(train,image_size=(224,224),batch_size=32,shuffle=True,label_mode="categorical")
tes=tf.keras.preprocessing.image_dataset_from_directory(test,image_size=(224,224),batch_size=32,shuffle=False,label_mode="categorical")
val=tf.keras.preprocessing.image_dataset_from_directory(valid,image_size=(224,224),batch_size=32,shuffle=True,label_mode="categorical")

train_dataset = tra.prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = val.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = tes.prefetch(buffer_size=tf.data.AUTOTUNE)

len(valid_dataset)

input=tf.keras.Input(shape=(224,224,3))

model=tf.keras.applications.EfficientNetB7(include_top=False)
model.trainable=False
x=model(input,training=False)
#x=tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)(x)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
#x=tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.016), activity_regularizer=tf.keras.regularizers.l1(0.006),bias_regularizer=tf.keras.regularizers.l1(0.006))(x)
output=tf.keras.layers.Dense(38,activation="softmax")(x)
model2=tf.keras.Model(input,output)
model2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

model2.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=1e-6,
    verbose=1
)

istr1=model2.fit(train_dataset,epochs=50,validation_data=valid_dataset,verbose=1,callbacks=[early_stopping,reduce_lr,model_checkpoint_cb(checkpoint_path)])

from google.colab import drive
drive.mount('/content/drive')

model_save_path = '/content/drive/My Drive/Project2_NewPLANT_VILLAGE_DATASET/model2.h5'
model2.save(model_save_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model2)
tflite_model = converter.convert()

# Save the TFLite model
with open('path_to_your_model2.tflite', 'wb') as f:
    f.write(tflite_model)

testt="/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"
tess=tf.keras.preprocessing.image_dataset_from_directory(testt,image_size=(224,224),batch_size=32,shuffle=False,label_mode="categorical")
test_datasett = tess.prefetch(buffer_size=tf.data.AUTOTUNE)

pred_prob=model2.predict(test_datasett)

pred=pred_prob.argmax(axis=1)

pred[:10]

y_lab=[]
for imm,lab in test_datasett.unbatch():
  y_lab.append(lab.numpy().argmax())

len(y_lab)

y_lab[pred[0]]

actual=val.class_names

actual

from sklearn.metrics import accuracy_score

accur=accuracy_score(y_pred=pred,y_true=y_lab)

accur

from sklearn.metrics import classification_report

rep=classification_report(y_lab,pred,target_names=val.class_names)

rep=classification_report(y_lab,pred,output_dict=True)

rep

f1={}
for k,v in rep.items():
  if(k=="accuracy"):
    break
  else:
    f1[val.class_names[int(k)]]=v["f1-score"]

f1.keys()

f1

f1.values()

import pandas as pd

f1_df=pd.DataFrame({"class_name":list(f1.keys()), "f1_score": list(f1.values())}).sort_values("f1_score",ascending=False)

fig,ax=plt.subplots(figsize=(15,18))
scores=ax.barh(range(len(f1_df)),f1_df["f1_score"].values)
ax.set_yticks(range(len(f1_df)))
ax.set_yticklabels(f1_df["class_name"])
ax.set_xlabel("F1_SCORE")
ax.set_title("38 CLASSES")
ax.invert_yaxis()

def autolabel(rects):
  """
  Attach a text label above each bar displaying its height (it's value).
  """
  for rect in rects:
    width = rect.get_width()
    ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
            f"{width:.2f}",
            ha='center', va='bottom')

autolabel(scores)

filepath=[]
for filez in val.list_files("/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/*/*",shuffle=False):
  filepath.append(filez.numpy())

import random

def custom_load(dirname=testt):
  file = random.choice(os.listdir(dirname))
  i = random.choice(os.listdir(dirname + "/" + file))
  filename = dirname + "/" + file + "/" + i

  im = tf.io.read_file(filename)
  im = tf.io.decode_image(im, channels=3)
  im = tf.image.resize(im, (224, 224))
  #im = im / 255

  im = tf.expand_dims(im, axis=0)
  pred = model2.predict(im)
  predd = np.argmax(pred)

  classs = test_datasett.class_names
  pred_clas = classs[predd]
  pred_prob = pred.max()

  # Plotting
  plt.figure(figsize=(8, 6))
  plt.imshow(tf.squeeze(im/255, axis=0))
  plt.title(f"Actual: {file}, Predicted: {pred_clas}, Probability: {pred_prob:.2f}", color="green" if file == pred_clas else "red")
  plt.axis('off')
  plt.show()

classs = val.class_names

print(len(filepath))
print(len(y_lab))
print(len(pred))

dff=pd.DataFrame({
    "im_path":filepath,
    "actual_labels":y_lab,
    "pred_labels":pred,
    "confidence":pred_prob.max(axis=1),
    "actual_ classNames": [classs[i] for i in y_lab],
    "predicted_classNames":[classs[i] for i in pred]

})

dff

dff["pred_correct"]=dff["actual_labels"]==dff["pred_labels"]

dff

wron_preds=dff[dff["pred_correct"]==False].sort_values("confidence",ascending=False)

wron_preds

for i, filename in enumerate(wron_preds[0:0+9].itertuples()):
  x=filename[1]

to_view = 9
start_index = 0
plt.figure(figsize=(20, 10))
for i, filename in enumerate(wron_preds[start_index:start_index+to_view].itertuples()):
  if(i==9):
    break
  else:
    plt.subplot(3, 3, i+1)


    im = tf.io.read_file(filename[1])
    im = tf.io.decode_image(im, channels=3)
    im = tf.image.resize(im, (224, 224))
    #im = im / 255

    im = tf.expand_dims(im, axis=0)
    pred = model2.predict(im)
    predd = np.argmax(pred)

    classs = val.class_names


    pred_prob = pred.max()

    # Plotting
    plt.imshow(tf.squeeze(im/255, axis=0))
    plt.title(f"Actual: {filename[5]}, Predicted: {filename[6]},Probabi: {filename[4]} ,Probability: {pred_prob}")
    plt.axis('off')
    plt.show()

wron_preds.head(20)

len(wron_preds)

def plot_loss_curves(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

