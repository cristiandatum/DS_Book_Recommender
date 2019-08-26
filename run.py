#Recommendation Engine
import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd
import random
import operator
from operator import itemgetter
from statistics import mean 

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

df_ratings = pd.read_csv('df_ratings.csv' )
df_books = pd.read_csv('df_books.csv' )
df_tags = pd.read_csv('df_tags.csv' )

query_dict=
{1: 'The Hunger Games (The Hunger Games, #1) - by: Suzanne Collins',
 2: "Harry Potter and the Sorcerer's Stone (Harry Potter, #1) - by: J.K. Rowling",
 3: 'Twilight (Twilight, #1) - by: Stephenie Meyer',
 4: 'To Kill a Mockingbird - by: Harper Lee',
 7: 'The Hobbit - by: J.R.R. Tolkien',
 9: 'Angels & Demons  (Robert Langdon, #1) - by: Dan Brown',
 10: 'Pride and Prejudice - by: Jane Austen',
 11: 'The Kite Runner - by: Khaled Hosseini',
 12: 'Divergent (Divergent, #1) - by: Veronica Roth',
 13: '1984 - by: George Orwell',
 16: 'The Girl with the Dragon Tattoo (Millennium, #1) - by: Stieg Larsson',
 18: 'Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) - by: J.K. Rowling',
 25: 'Harry Potter and the Deathly Hallows (Harry Potter, #7) - by: J.K. Rowling',
 30: 'Gone Girl - by: Gillian Flynn',
 31: 'The Help - by: Kathryn Stockett',
 36: 'The Giver (The Giver, #1) - by: Lois Lowry',
 37: 'The Lion, the Witch, and the Wardrobe (Chronicles of Narnia, #1) - by: C.S. Lewis',
 39: 'A Game of Thrones (A Song of Ice and Fire, #1) - by: George R.R. Martin',
 48: 'Fahrenheit 451 - by: Ray Bradbury',
 59: "Charlotte's Web - by: E.B. White",
 117: "A Wrinkle in Time (A Wrinkle in Time Quintet, #1) - by: Madeleine L'Engle",
 422: 'Harry Potter Boxset (Harry Potter, #1-7) - by: J.K. Rowling',
 562: 'The Way of Kings (The Stormlight Archive, #1) - by: Brandon Sanderson',
 780: 'Calvin and Hobbes - by: Bill Watterson',
 862: 'Words of Radiance (The Stormlight Archive, #2) - by: Brandon Sanderson',
 964: 'J.R.R. Tolkien 4-Book Boxed Set: The Hobbit and The Lord of the Rings - by: J.R.R. Tolkien',
 1010: 'The Essential Calvin and Hobbes: A Calvin and Hobbes Treasury - by: Bill Watterson',
 1308: 'A Court of Mist and Fury (A Court of Thorns and Roses, #2) - by: Sarah J. Maas',
 1380: 'The Complete Maus (Maus, #1-2) - by: Art Spiegelman',
 1618: 'A Voice in the Wind (Mark of the Lion, #1) - by: Francine Rivers',
 1788: 'The Calvin and Hobbes Tenth Anniversary Book - by: Bill Watterson',
 2093: 'The Stand: Soul Survivors - by: Roberto Aguirre-Sacasa',
 2167: 'Empire of Storms (Throne of Glass, #5) - by: Sarah J. Maas',
 2244: 'Saga, Vol. 2 (Saga, #2) - by: Brian K. Vaughan',
 3275: 'Harry Potter Boxed Set, Books 1-5 (Harry Potter, #1-5) - by: J.K. Rowling',
 3395: 'The Kindly Ones (The Sandman #9) - by: Neil Gaiman',
 3628: 'The Complete Calvin and Hobbes - by: Bill Watterson',
 7264: 'Master of the Senate (The Years of Lyndon Johnson, #3) - by: Robert A. Caro',
 9360: 'The Green Mile, Part 6: Coffey on the Mile - by: Stephen King',
 9566: 'Attack of the Deranged Mutant Killer Monster Snow Goons - by: Bill Watterson',
 9807: "Corrupt (Devil's Night, #1) - by: Penelope Douglas",
 9821: 'The Art of Amy Brown - by: Amy Brown',
 9823: 'The Turner House - by: Angela Flournoy',
 9846: 'The Tiger Who Came to Tea - by: Judith Kerr',
 9854: 'The Last Days of Dogtown - by: Anita Diamant',
 9859: 'Beacon 23: The Complete Novel (Beacon 23, #1-5) - by: Hugh Howey',
 9871: 'Black Cherry Blues (Dave Robicheaux, #3) - by: James Lee Burke',
 9874: 'Dancing Wu Li Masters: An Overview of the New Physics (Perennial Classics) - by: Gary Zukav',
 9877: 'Trumps of Doom (The Chronicles of Amber, #6) - by: Roger Zelazny',
 9879: 'Locked On (Jack Ryan Universe, #14) - by: Tom Clancy',
 9896: 'Turn of Mind - by: Alice LaPlante',
 9922: 'Sea of Silver Light (Otherland, #4) - by: Tad Williams',
 9923: 'The Green Mile, Part 5: Night Journey - by: Stephen King',
 9940: 'Demon Thief (The Demonata, #2) - by: Darren Shan',
 9960: 'The Prize Winner of Defiance, Ohio: How My Mother Raised 10 Kids on 25 Words or Less - by: Terry Ryan',
 9963: 'Krondor: The Betrayal (The Riftwar Legacy, #1) - by: Raymond E. Feist',
 9966: 'The Ground Beneath Her Feet - by: Salman Rushdie',
 9981: 'The Twelfth Card (Lincoln Rhyme, #6) - by: Jeffery Deaver',
 9991: 'The Famished Road - by: Ben Okri',
 9995: 'Billy Budd, Sailor - by: Herman Melville'}


genres={1:'action', 2:'adult', 3:'adventure', 4:'all-time-favorites', 5:'american', 6:'biography', \
        7:'bookclub', 8:'british', 9:'children', 10:'classics', 11:'comedy', 12:'coming-of-age', \
        13:'contemporary', 14:'crime', 15:'drama', 16:'english', 17:'family', 18:'fantasy', 19:'friendship' , \
        20:'historical', 21:'history', 22:'horror', 23:'kids', 24:'literature', 25:'love', 26:'magic',\
        27:'mystery', 28:'non-fiction', 29:'paranormal', 30:'philosophy', 31:'relationships', 32:'romance' , \
        33:'school', 34:'sci-fi', 35:'suspense', 36:'teen', 37:'war', 38:'women', 39:'SURPRISE-ME!' }

def query_book_ratings(query_dict):
    
    '''
    INPUT:
    dictionary with the most representative books.
    
    OUTPUT:
    returns a dictionary with key: tag_id; and value: user book rating
    '''
    
    user_ratings={}
    
    counter=1
    
    print("Please evaluate these 60 books. Instructions: \n \
    - Please rate from 1 to 5. \n \
    - Enter 0 if you haven't read the book. \n \
    - Enter 99 if you want to stop rating.")

    for book in list(query_dict.keys()):
    
        while True:
            
            try:
                print("* Book Rating",counter, "/ 60")
                print(query_dict.get(book))
                rating=int(input("Your rating: "))
            
            except ValueError:
                print("Sorry, I didn't understand that.")
                continue
            
            if rating==99:
                break
                
            if rating <1 or rating>5:
                print("Please rate from 1 to 5.")
                continue
                
            else:
                user_ratings.update({book:rating})
                break
        counter+=1
        
        if rating==99:
            break
        
    return user_ratings

def query_genre_likes(genres):
    
    user_genres=[]
    
    counter=1
    
    print ("Please enter your genres of interest: \n \
    1: action , 2: adult , 3: adventure , 4: all-time-favorites , 5: american , 6: biography , 7: bookclub ,\
    8: british , 9: children , 10: classics , 11: comedy , 12: coming-of-age , 13: contemporary , 14: crime ,\
    15: drama , 16: english , 17: family , 18: fantasy , 19: friendship , 20: historical , 21: history ,\
    22: horror , 23: kids , 24: literature , 25: love , 26: magic , 27: mystery , 28: non-fiction ,\
    29: paranormal , 30: philosophy , 31: relationships , 32: romance , 33: school , 34: sci-fi , 35: suspense ,\
    36: teen , 37: war , 38: women , 39: SURPRISE-ME")

    print ("Type 99 when finished.")
    
    while True:
        
        try:
            genre=int(input("Please type genre and ENTER: "))
            
        except ValueError:
            print ("Sorry, I didn't understand that.")
            continue
        
        if genre==99:
            break
        
        if counter>10:
            break
        
        if genre in user_genres:
            print ("This genre was already included.")
        
        if genre<1 or genre>39:
            print ("Please limit to the choices listed.")
            continue
        else:
            user_genres.append(genre)
        print("99 ENTER when finished.\n")
        
    counter+=1
    
    return sorted(list(set(user_genres)))

user_genres=query_genre_likes(genres)

def weighted_mean(df_ratings, user_ratings, sample_size = 8000):
    
    '''   
    INPUT:
    - df_ratings: dataframe containing book ratings. Ratings arranged by columns: user_id; book_id; rating.
    - user_ratings: dictionary containing user's book ratings. Arranged by book_id (key) and rating (value).
    - sample_size: integer. Size of random sample from all the readers' ratings.
    
    OUTPUT:
    - book_wmeans: dictionary containing weighted means for every book that the user has not already rated.
      Arranged by book_id (key), and rating (value) in descending order.
    '''
    
    #randomly select a sample of size "sample_size" among the 53,424 readers.
    random.seed(6)
    random_ids=random.sample(list(df_ratings['user_id'].unique()),sample_size) #randomly select a sample of 
    
    #unstack the readers' ratings to a dataframe containing book ratings.
    #Ratings arranged by readers (rows) and books (columns).
    df_reader_ratings=df_ratings[df_ratings['user_id'].isin(random_ids)].\
                                                    groupby(['user_id', 'book_id'])['rating'].max().unstack()
           
    #Add a new row "0" for the User.
    df_reader_ratings.loc[0] = None

    #Sort df_reader_ratings index so that the new row "0" is shown at the top:
    df_reader_ratings=df_reader_ratings.sort_index()

    #Populate row "0" according to the user's ratings included in user_ratings.
    for key in list(user_ratings.keys()):        
        df_reader_ratings.loc[0][key]=user_ratings.get(key)
        
    #create a copy of the unstacked dataframe filling all NaN values with zero's.
    df_reader_ratings_dummy=df_reader_ratings.copy().fillna(0)
    
    #create a matrix with all the cosine similarities between readers.
    cosine_ratings=cosine_similarity(df_reader_ratings_dummy,df_reader_ratings_dummy)
    
    #create a dataframe from the cosine_ratings matrix. Rows = users; Columns = users.
    cosine_ratings=pd.DataFrame(cosine_ratings,index=df_reader_ratings.index,columns=df_reader_ratings.index)
    
    #dictionary to contain weighted means for each book. 
    book_wmeans={}
    
    #get a series containing the cosine similarity between User (row 0) and every other user     
    cosine_reader=cosine_ratings[0]
            
    #For every book with rating:
    for book in list(df_reader_ratings.columns):
        
    #If book has not already been rated by the user:
        if book not in list(user_ratings.keys()):

            #ratings of the book in iteration by all readers.
            reader_ratings=df_reader_ratings[book] #all of the reader ratings for given book 'book'

            #index containing the rreaders with NaN ratings for the book in iteration.        
            #https://stackoverflow.com/questions/14016247/find-integer-index-of-rows-with-nan-in-pandas-dataframe
            idx_nans = reader_ratings[reader_ratings.isnull()].index #which indices contained NaNs
    
            #ratings of the book in iteration by all readers without any NaNs.
            reader_ratings=df_reader_ratings[book].dropna()

            #remove the readers identified with NaN values for the book in iteration.
            cosine_book=cosine_reader.drop(index=idx_nans)
        
            #carry out the dot product between both series. These do not contain NaN values.
            wmean_rating=np.dot(cosine_book,reader_ratings)/cosine_reader.sum()

            #include weighted mean in dictionary.
            book_wmeans.update({book:wmean_rating})
    
    #sort dictionary value items in descending order.
    book_wmeans=sorted(book_wmeans.items(), key=operator.itemgetter(1),reverse=True)
    
    return book_wmeans


def genre_filter(user_genres,user_ratings,book_recommends,number=5):
    '''
    INPUT:
    user_id1 - the first user_id of an individual as int
    user_id2 - the second user_id of an individual as int
    
    OUTPUT:
    pw_dist - float value with the distance between user_id1 and used_id2 (smaller values indicate greater similarity)
    '''
    
    #books corresponding to user tag preferences.
    df_book_filter=df_tags[df_tags['tag_id'].isin(user_genres)]
    
    #filter out books already rated by user.
    df_book_filter=df_book_filter[~df_book_filter['book_id'].isin(list(user_ratings.keys()))]
    
    #filter for books included in book_recommends.
    df_book_filter=df_book_filter[df_book_filter['book_id'].isin(book_recommends[:number])]
    
    if df_book_filter.shape[0]<number:
        
        print("Please try including more tags in your selection.")            
        
    
    final_recommend=get_book_titles(list(df_book_filter['book_id']))
    
    return final_recommend

    print(genre_filter(user_genres,user_ratings,book_recommends,number=5))