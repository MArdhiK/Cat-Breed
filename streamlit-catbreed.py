import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from urllib.request import urlopen

def main():
    # importing tensorflow model
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    tf.keras.backend.clear_session()

    pre_trained_model = InceptionV3(input_shape = (225, 225, 3), 
                                include_top = False, 
                                weights = 'imagenet')


    for layer in pre_trained_model.layers:
        layer.trainable = False
  
    # pre_trained_model.summary()

    # cut off at the layer named 'mixed7'
    last_layer = pre_trained_model.get_layer('mixed7')

    # know its output shape
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    from tensorflow.keras.optimizers import RMSprop

    # Feed the last cut-off layer to our own layers
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)

    # Add a dropout rate of 0.4
    x = tf.keras.layers.Dropout(0.4)(x) 
    # Add a fully connected layer with 256 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)

    # Add a dropout rate of 0.3
    x = tf.keras.layers.Dropout(0.3)(x)
    # Add a fully connected layer with 256 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)

    # Add a final dropout layer
    x = tf.keras.layers.Dropout(0.2)(x)                  
    # Add a final softmax layer for classification
    x = tf.keras.layers.Dense  (6, activation='softmax')(x)           

    model_inceptionv3 = tf.keras.Model(pre_trained_model.input, x) 

    model_inceptionv3.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
    model_inception.load_weights('model_inceptionv3_weights.h5')

    def predict_image(image_upload, model = model_inceptionv3):
        im = Image.open(image_upload)
        resized_im = im.resize((225, 225))
        im_array = np.asarray(resized_im)
        im_array = im_array*(1/225)
        im_input = tf.reshape(im_array, shape = [1, 225, 225, 6])

        predict_array = model.predict(im_input)[0]
        americashorthair_prob = predict_array[0]
        bengal_prob = predict_array[1]
        mainecoon_prob = predict_array[2]
        ragdoll_prob = predict_array[3]
        scottishfold_prob = predict_array[4]
        sphinx_prob = predict_array[5]

        s = [americanshorthair_prob, bengal_prob, mainecoon_prob, ragdoll_prob, scottishfold_prob, sphynx_prob]

        import pandas as pd
        df = pd.DataFrame(predict_array)
        df = df.rename({0:'Probability'}, axis = 'columns')
        breed = ['American Shorthair', 'Bengal', 'Maine Coon', 'Ragdoll', 'Scottish Fold', 'Sphynx']
        df['Cat Breed'] = breed
        df = df[['Cat Breed', 'Probability']]

        predict_label = np.argmax(model.predict(im_input))

        if predict_label == 0:
            predict_catbreed = 'The uploaded image similar as American Shorthair Cat.'
        elif predict_label == 1:
            predict_catbreed = 'The uploaded image similar as Bengal Cat.'
        elif predict_label == 2:
            predict_catbreed = 'The uploaded image similar as Maine Coon Cat.'
        elif predict_label == 3:
            predict_catbreed = 'The uploaded image similar as Ragdoll Cat.'
        elif predict_label == 4:
            predict_catbreed = 'The uploaded image similar as Scotttish Fold Cat.'
        else:
            predict_catbreed = 'The uploaded image similar as Sphynx Cat.'


        return predict_catbreed, df, im, s

   
        st.title('Welcome to the Cat Breed Image Clasification Project')
        st.markdown('This is a web application of deep learning model which has been trained to classify \
            six cat breed (American Shorthair, Bengal, Maine Coon, Ragdoll, Scottish Fold, Sphinx) based on their images.')
        st.markdown("Paste a link to an image and the deployed deep learning model will classify it.")
        st.markdown("The linked image should be in 'jpg', 'jpeg', or 'png' format.")
        st.markdown("Tips: right click on an image and choose 'Copy Image Link'.")
        
        image_url = st.text_input("Paste the image file's link here")
        
        if st.button("Classify the image"):
            
            file = BytesIO(urlopen(image_url).read())
            img = file
            label, df_output, uploaded_image, s = predict_image(img)
            st.image(uploaded_image, width = None)
            st.write(label)
            st.write(df_output)

            
            fig, ax = plt.subplots()
            ax = sns.barplot(x = 'Cat Breed', y = 'Probability',data = df_output)

            for i,p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,
                    height + 0.003, str(round(s[i]*100,2))+'%',
                    ha="center") 

            st.pyplot(fig)
          
if __name__ == '__main__':
    main()
