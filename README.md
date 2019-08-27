# DS_Book_Recommender
### Book Recommender Capstopne Project for Udacity Data Science nanodegree

This Python script uses a dataset containing six million ratings and tags for the ten thousand most popular books. It requests the user to rate as many as 60 books on a scale 1 - 5; and then requests the user to enter a number of tags that may be of interest from a list of 39. 

The engine uses the user's input and compares it to other readers in a database containing over 53,000 readers with ratings of 10,000 books.

The data was obtained from fastml.com which in turn is based on the well-known goodreads.com website. 

This script was developed using Python and SKLearn machine learning libraries for cosine similarity and nearest neighbors (Kmeans).

### Installation: 
Clone the GitHub repository and use Anaconda distribution of Python 3.6.7.

    $ git clone https://github.com/cristiandatum/DS_Book_Recommender.git


The code can be viewed and modified with Jupyter Notebooks.

### Instructions:

Run the following commands in the project's root directory to set up your database and model.

To run the book recommendation engine:

    $ python run.py

### Built With:
- Jupyter Notebooks
- Visual Studio Code

### Authors:
Cristian Alberch
https://github.com/cristiandatum

You can contact the author on: cristian.alberch@outlook.com

### License:
This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. 

To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

Feel free to use the code in the Jupyter Notebook as you like.

### Acknowledgments:
The .csv files containing data from Goodreads was obtained from:

http://fastml.com/goodbooks-10k-a-new-dataset-for-book-recommendations/

https://github.com/zygmuntz/goodbooks-10k.git

#### Thanks to:

- Philipp Spachtholz, for his thorough analysis of the data and building the first recommender based on goodbooks-10k.
- Maciej Kula, for adding the dataset to Spotlight
