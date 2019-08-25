# DS_Book_Recommender
### Book Recommender Capstopne Project for Udacity Data Science nanodegree

This Flask Web app uses Machine Learning to predict the classification of a given typical disaster emergency response text message. For example, if the input is: "My house is burning down", the classification category should be "Fire". This application is useful to identify emergency response services required.

This Python script uses a dataset containing six million ratings for the ten thousand most popular books and classified with tags. It requests the user to rate as many as 60 books on a scale 1 - 5; and then requests the user to include a number of tags that may be of interest from a list of 39. 

The information provided by the user is used by the engine to compare the user against other readers in a database containing over 53,000 readers with ratings of 10,000 books.

The data was obtained from fastml.com which in turn is based on the well-known goodreads.com website. 

This script was developed using Python and SKLearn machine learning libraries for cosine similarity and nearest neighbors (Kmeans).

### Installation: 
Clone the GitHub repository and use Anaconda distribution of Python 3.6.7.

    $ git clone https://github.com/cristiandatum/DS_Response_Pipeline.git

In addition This will require pip installation of the following:

    $ pip install SQLAlchemy
    $ pip install nltk

The code can be viewed and modified with Jupyter Notebooks.

### Instructions:

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database:

    $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves model:

    $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command: model in the app's directory to run your web app: 

    $ python app/run.py
    
 Go to: `http://0.0.0.0:3001/`

### Built With:
- Visual Studio Code
- Udacity Project Workspace IDE

### Authors:
Cristian Alberch
https://github.com/cristiandatum

### License:
This project is licensed under the MIT License.

Feel free to use the code in the Jupyter Notebook as you like.

### Acknowledgments:
The .csv files containing data from real disaster response messages was provided by Udacity from Figure Eight.
The started code was provided by Udacity.
