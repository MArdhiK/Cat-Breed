# Cat Breed Image Classification
In this project I built CNN model and use transfer learning method to build image classifier. <br>
The dataset consist of TRAIN folder with 200 images for each class and TEST folder with 50 images for each class. <br>
All of the images scraped from google image, shutterstcok and another image provider with this google chrome [plug-in](https://chrome.google.com/webstore/detail/download-all-images/nnffbdeachhbpfapjklmpnmjcgamcdmm)  <br>
The full dataset can be downloaded here https://www.kaggle.com/solothok/cat-breed

My aim in this project is to classify these 6 cat breed :
1. American Shorthair
2. Bengal 
3. Maine Coon
4. Ragdoll
5. Scottish Fold
6. Sphinx

With this model I can achieve 88,67% accuracy on training set, and 91,67% accuracy on validation set

Here's the web application of this model https://share.streamlit.io/mardhik/cat-breed/catbreed.py

#### A Glimpse of the Web

<img src=images/1.interface.png width=500>     <img src=images/2.output.png width=500>
Just enter the image link, and it will show the probabilty of the images between 6 cats
