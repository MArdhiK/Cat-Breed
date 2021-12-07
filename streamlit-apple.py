import os
import io
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import random
import PIL
from PIL import Image
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from urllib.request import urlopen
from io import BytesIO

def main():
    # importing tensorflow model
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    tf.keras.backend.clear_session()

    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = 'imagenet')


    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    # pre_trained_model.summary()

    # cut off at the layer named 'mixed7'
    last_layer = pre_trained_model.get_layer('mixed7')

    # know its output shape
    last_output = last_layer.output

    from tensorflow.keras.optimizers import RMSprop

    # Feed the last cut-off layer to our own layers
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)
    # Add a dropout rate of 0.4
    x = tf.keras.layers.Dropout(0.4)(x) 
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a dropout rate of 0.3
    x = tf.keras.layers.Dropout(0.3)(x)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a final dropout layer
    x = tf.keras.layers.Dropout(0.2)(x)                  
    # Add a final softmax layer for classification
    x = tf.keras.layers.Dense  (3, activation='softmax')(x)           

    model_inception = tf.keras.Model(pre_trained_model.input, x) 

    model_inception.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
    model_inception.load_weights('model_inception_weights.h5')

    def predict_image(image_upload, model = model_inception):
        im = Image.open(image_upload)
        resized_im = im.resize((150, 150))
        im_array = np.asarray(resized_im)
        im_array = im_array*(1/225)
        im_input = tf.reshape(im_array, shape = [1, 150, 150, 3])

        predict_array = model.predict(im_input)[0]
        mac_proba = predict_array[0]
        ipad_proba = predict_array[1]
        iphone_proba = predict_array[2]

        s = [mac_proba, ipad_proba, iphone_proba]

        import pandas as pd
        df = pd.DataFrame(predict_array)
        df = df.rename({0:'Probability'}, axis = 'columns')
        prod = ['Macbook', 'iPad', 'iPhone']
        df['Product'] = prod
        df = df[['Product', 'Probability']]

        predict_label = np.argmax(model.predict(im_input))

        if predict_label == 0:
            predict_product = 'The uploaded image resembles a Macbook.'
        elif predict_label == 1:
            predict_product = 'The uploaded image resembles an iPad.'
        else:
            predict_product = 'The uploaded image resembles an iPhone.'



        return predict_product, df, im, s

    st.sidebar.title('Navigation')
    pages = st.sidebar.radio("Pages", ("Home Page", "Image Classifier", "About the Project", "About the Author"))
    if pages == "Home Page":
        st.title('Welcome to the Apple Products Image Clasification Project')
        st.image('apple.jpg', width=650)
        st.markdown('This is a deployment page to house a deep learning model which has been trained to classify \
            three Apple products (Macbook, iPad, and iPhone) based on their images.')

    elif pages == "Image Classifier":
        st.title("Image Classifier Testing Page")
        st.markdown("Paste a link to an image and the deployed deep learning model will classify it as either a Macbook, iPad, or iPhone.")
        st.markdown("The linked image should be in 'jpg', 'jpeg', or 'png' format.")
        if st.button("See disclaimers"):
            st.markdown("Disclaimer: There are a lot of other Apple products such as AirPods, AppleWatch, or their full tower Mac, but in this project, \
            we'll focus on the three products: Macbook, iPad, and iPhone.")
            st.markdown("Second Disclaimer: Some image links might be forbidden to be read. If any error occurs, please try another image.")
            st.markdown("Initially, the author implements a file upload system, where the uploaded user file will be put into the deep learning model for classification. \
            However, it only works for locally deployed web app. When the author host the web app online via share.streamlit.io, everytime we upload file, we are given with \
            HTTP 500 error issues. This problem has been reported by the Streamlit community and is expected to be resolved in future updates to the package.")
         
        st.markdown('Example of a link to an iPhone image: https://cnet1.cbsistatic.com/img/DK2ELJz5acYtFQXGiXpDcMnj0B8=/532x299/2018/06/04/6cf86c5c-687f-4755-b893-c12ecb2e1124/apple-wwdc-2018-1098.jpg')
        st.markdown("Tips: right click on an image to 'open it on a new tab' or 'copy image address' to get the image's link")
        
        image_url = st.text_input("Paste the image file's link here")
        
        if st.button("Classify the image"):
            
            file = BytesIO(urlopen(image_url).read())
            img = file
            label, df_output, uploaded_image, s = predict_image(img)
            st.image(uploaded_image, width = None)
            st.write(label)
            st.write(df_output)

            
            fig, ax = plt.subplots()
            ax = sns.barplot(x = 'Product', y = 'Probability',data = df_output)

            for i,p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,
                    height + 0.003, str(round(s[i]*100,2))+'%',
                    ha="center") 

            st.pyplot(fig)

    elif pages == "About the Project":
        st.title('Project Insights')
        st.subheader('Dataset')
        st.markdown('I collected the images myself by downloading the top 400-500 images from Google Image Search of Macbook, iPad, and iPhone.')
        st.markdown('The dataset is available to be downloaded in my [Kaggle profile](https://www.kaggle.com/radvian/apple-products-image-dataset).')
        st.markdown("NB: I have a Kaggle profile to host private datasets and download public datasets. Regarding Data Science, I'm more active in github (uploading personal projects) and writing medium posts.")
        st.subheader('CNN Model')
        st.markdown('The deployed model is a deep learning model with multiple Convolutional Neural Network layers, constructed in TensorFlow.')
        st.markdown('I use transfer learning to take trained layers from InceptionV3 model which has been previously trained in the ImageNet dataset.\
            The model details can be read [here](https://cloud.google.com/tpu/docs/inception-v3-advanced).')
        st.markdown("The InceptionV3 model is cut off after its 'mixed7' layer, and then joined with a few untrained Dense and Dropout layers, before using \
            a 'softmax' activation function on the final layer to make prediction.")
        st.markdown("The model is trained using 'Reduce-LR-on-Plateau' and 'Early Stopping' callbacks. While initially plotted to train for 100 layers, the model \
            stopped training under 30 epochs, and have a 0.92 accuracy on training set, with 0.88 accuracy on validation set.")
        st.subheader('How to make it better')
        st.markdown('- We can always try to add more images into the dataset. Also, we might want to introduce more kinds of images such as \
            different product lines of iPad, iPhone, and/or Macbook. I chose not to do this for two reasons: first, memory constraint (as I am training on a free-tier Google Colab notebook), \
            and second, I wanted to see if I can get a good model accuracy with just 400-500 images of each products taken without any further filtering from Google Images top searches.')
        st.markdown('- Choose different optimizers and/or loss function.')
        st.markdown('- Choose different model architecture and/or pretrained weights. Other models include Xception and InceptionResNetV2.')

    elif pages == "About the Author":
        st.title('About the Author')
        st.subheader("Greetings! My name is Grady Matthias Oktavian. Nice to meet you!")
        st.write("I graduated at 2020 from Universitas Pelita Harapan with the title Bachelor of Mathematics, majoring in Actuarial Science. \
            Currently, I'm studying at Pradita University as a student in the Master of Information Technology degree majoring in Big Data and IoT. \
            I am also employed as an IFRS 17 Actuary at PT Asuransi Adira Dinamika.") 
        st.write("I like learning about statistics and mathematics. Since today is the age of Big Data, I find that most people who \
            aren't majoring in mathematics might find themselves overwhelmed with large amounts of data. My dream is to help people make better \
            decision through a data-driven approach.") 
        st.write("In order to do that, I am happy to wrangle and analyze raw data, creating models based on it, and \
            conveying insights I gained to others who are not well-versed in data, so they can understand it without having to \
            get their hands dirty with the data. I hope through my help, people can understand things better, busineess owners can \
            make better contingency plans and take better decisions.")
        st.markdown("Currently, I am certified by Google as a [TensorFlow Developer](https://www.credential.net/794f2bb6-d377-4b5b-ac9d-9d3bed582d2d), \
            and a [Cloud Professional Data Engineer](https://www.credential.net/df7c3d9d-011a-41fd-9d64-49ada5a0619c#gs.ktrahi).")
        st.write("If you wish to have a suggestion for this project, or contact me for further corresnpondences, \
            please reach out to me via email at gradyoktavian@gmail.com, or send me a message to \
            my [LinkedIn profile](https://www.linkedin.com/in/gradyoktavian/).")
        st.write("Thank you! Have a nice day.")



if __name__ == '__main__':
    main()
