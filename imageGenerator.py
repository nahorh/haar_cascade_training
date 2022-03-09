# from keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array, ImageDataGenerator
# Total Generated number
total_number = 100

data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=1,
                              zoom_range=0.1, horizontal_flip=True, vertical_flip=False, rotation_range=5, brightness_range=[0.1, 1.0])

# Create image to tensor
img = load_img("./neg5.png", color_mode='grayscale')
arr = img_to_array(img)
tensor_image = arr.reshape((1, ) + arr.shape)
imgnum = 100
for i, _ in enumerate(data_gen.flow(x=tensor_image,
                                    batch_size=total_number,
                                    save_to_dir="neg5",
                                    save_prefix="",
                                    save_format=".png",)):
    imgnum += 1
    if imgnum > total_number:
        break
