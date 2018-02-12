
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# column headers for the dataset
data_cols = ['userID', 'movieID', 'rating', 'timestamp']
item_cols = ['movieID', 'movie_title', 'release_date', 'video_release_date','IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy','Film-Noir','Horror', 'Musical','Mystery','Romance ','Sci-Fi','Thriller', 'War', 'Western']
user_cols = ['userID', 'age', 'gender', 'occupation', 'zip_code']

# importing the data files onto dataframes
users = pd.read_csv('./ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
item = pd.read_csv('./ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
data = pd.read_csv('./ml-100k/u.data', sep='\t', names=data_cols, encoding='latin-1')

# merging 3 data sets
dataset = pd.merge(pd.merge(item, data), users)

n_users = data.userID.unique().shape[0]
n_movies = data.movieID.unique().shape[0]

print(n_users, n_movies)

# splitting data to testing and training data
train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)

print(train_data.shape)
print(test_data.shape)

train_data_matrix = np.zeros((n_users, n_movies))
for line in data.itertuples():
    # print(line)
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    # -1 bcoz there is no user or movie with id 0

# print(train_data_matrix)
# using pairwise distance from from sklearn
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
movie_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
print(user_similarity)
user_similarity_crr = pairwise_distances(train_data_matrix, metric='correlation')
movie_similarity_crr = pairwise_distances(train_data_matrix.T, metric='correlation')

def predict ( ratings, similarity, type='user' ):

    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # here axis = 1 means along the row. Default is axis = 0 which means along the column
        #You use np.newaxis so that mean_user_rating has same format as ratings         
        #print(ratings.shape)  (943, 1682)
        #print(mean_user_rating[:, np.newaxis].shape) (943,1)


        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

movie_prediction = predict (train_data_matrix, movie_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

movie_prediction_crr = predict(train_data_matrix, movie_similarity_crr, type='item')
user_prediction_crr = predict(train_data_matrix, user_similarity_crr, type='user')

print(movie_prediction[ :6])
#print(user_prediction)
print(movie_prediction_crr[ :6])


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(movie_prediction, train_data_matrix)))

print ('User-based CF RMSE: ' + str(rmse(user_prediction_crr, train_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(movie_prediction_crr, train_data_matrix)))






''' def pearson_correlation(user1, user2):
    df1= dataset.loc[(dataset.userID == user1 ),'movieID'].tolist()
    df2= dataset.loc[(dataset.userID == user2 ),'movieID'].tolist()

    print(len(df1))
    print(len(df2))

    both_rated = { }
    for i in df1:
        if i in df2:
            both_rated[i] = 1       
    
    number_of_rating = len(both_rated)
    print(df1)
    print(df2)
    print(number_of_rating)

    if number_of_rating == 0:
        return 0 
    # Add up all the preferences of each user
    #print(user1_preferences_sum = sum(df1[item] for item in both_rated))
    #print(user2_preferences_sum = sum(df2[item] for item in both_rated))

    # Sum up the squares of preferences of each user
    user1_square_preferences_sum = sum([pow(df1[item],2) for item in both_rated])
    user2_square_preferences_sum = sum([pow(df2[item],2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum( [dataset[person1][item] * dataset[person2][item] for item in both_rated] )
 
    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - ( user1_preferences_sum*user2_preferences_sum/number_of_ratings )
    denominator_value = sqrt( (user1_square_preferences_sum - pow(user1_preferences_sum,2)/number_of_ratings) * (user2_square_preferences_sum - pow(user2_preferences_sum,2)/number_of_ratings) )
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value/denominator_value
        return r

print( pearson_correlation(1, 2))
 

pearson_correlation(4, 2)


# check_both_rated(1, 2) '''