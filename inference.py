import efficientnet.tfkeras
import tensorflow as tf 

import cv2
import numpy as np
import argparse
import utils   

def predict_image(image_path, model):
    img = cv2.imread(image_path)

    img_copy = img.copy()
    img_copy = cv2.resize(img_copy , (224, 224))
    img_copy = img_copy / 255.
    img_copy = np.expand_dims(img_copy, axis=0)
    
    prediction = model.predict(img_copy)
    
    return img, prediction

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-i','--image', default='data/seg_test/seg_test/forest/20056.jpg'
    )
    
    argparser.add_argument(
        '-w', '--weights', default='weights/effnet_b5_weights.h5'
    )
    argparser.add_argument(
        '-m', '--memory', type=int, default=3024
    )

    args = argparser.parse_args()

    # load model first
    utils.limit_gpu_memory(args.memory)
    model = tf.keras.models.load_model(
        args.weights, custom_objects={'SwishActivation': utils.SwishActivation(utils.swish_act)}, compile=False
    )
    
    img, prediction = predict_image(args.image, model)
    
    selected_label = utils.labels[np.argmax(prediction)]
    
    cv2.imshow(selected_label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()