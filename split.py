import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset_dir = 'Celebrity Faces Dataset'
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
people = os.listdir(dataset_dir)
for person in people:
    person_dir = os.path.join(dataset_dir, person)
    train_person_dir = os.path.join(train_dir, person)
    os.mkdir(train_person_dir)
    validation_person_dir = os.path.join(validation_dir, person)
    os.mkdir(validation_person_dir)
    test_person_dir = os.path.join(test_dir, person)
    os.mkdir(test_person_dir)

    # Split the images into train, validation, and test sets (adjust the test_size and validation_size as needed)
    train_images, temp_images = train_test_split(os.listdir(person_dir), test_size=0.2, random_state=42)
    validation_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    for image in train_images:
        src = os.path.join(person_dir, image)
        dst = os.path.join(train_person_dir, image)
        shutil.copyfile(src, dst)

    for image in validation_images:
        src = os.path.join(person_dir, image)
        dst = os.path.join(validation_person_dir, image)
        shutil.copyfile(src, dst)

    for image in test_images:
        src = os.path.join(person_dir, image)
        dst = os.path.join(test_person_dir, image)
        shutil.copyfile(src, dst)