import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout,Flatten
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from  sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

num_classes = 17
num_nodes = 1024
learningrate = 0.0001
batch_size = 128
image_size = (299  , 299)
epochs = 20
dataset_dir = 'dataset'
isFreeze = False 

base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(299, 299,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(num_nodes, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

if isFreeze :
    for layer in base_model.layers:
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=learningrate), loss='categorical_crossentropy', metrics=['accuracy'])

def grayscale_conversion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.0,
    horizontal_flip=True,

    # preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
    preprocessing_function=grayscale_conversion,
    rescale=1.0/255.0,
    fill_mode='nearest'
)



train_generator = datagen.flow_from_directory(
    dataset_dir + '/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = datagen.flow_from_directory(
    dataset_dir + '/validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = datagen.flow_from_directory(
    dataset_dir + "/test",
    class_mode="categorical",
    target_size=image_size, 
    shuffle=False,
    batch_size=1)


step_size_train = len(train_generator)
step_size_val = len(validation_generator)

history=model.fit_generator(generator=train_generator,
                            steps_per_epoch=step_size_train,
                            validation_data = validation_generator,
                            validation_steps = step_size_val,
                            epochs=epochs,
                            verbose = 1)

N = list(range(1, epochs + 1))
plt.plot(N, history.history["accuracy"], label="Train_acc")
plt.plot(N, history.history["val_accuracy"], label="Validate_acc")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'result/acc_graph/acc_inceptionV3_bs{batch_size}_e{epochs}_grey_lr{learningrate}_nn{num_nodes}_c{num_classes}_fr{isFreeze}.png')  # Save the accuracy plot as an image
plt.close()  # Close the current figure to clear the plot

plt.plot(N, history.history['loss'], label="Train_loss")
plt.plot(N, history.history['val_loss'], label="Validate_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'result/loss_graph/loss_inceptionV3_bs{batch_size}_e{epochs}_grey_lr{learningrate}_nn{num_nodes}_c{num_classes}_fr{isFreeze}.png')  

print("batch size : ",batch_size)
print("epochs : ",epochs)
print("num nodes : ",num_nodes)
y_true = test_generator.classes
preds = model.predict_generator(test_generator)
print(preds.shape)
print(preds)
y_pred = np.argmax(preds,axis=1)
print(test_generator.classes)
print(y_pred)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
model.save(f'result/model/inception_bs{batch_size}_e{epochs}_grey_lr{learningrate}_nn{num_nodes}_c{num_classes}_fr{isFreeze}.h5')