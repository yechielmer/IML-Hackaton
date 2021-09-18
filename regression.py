
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

import pickle
import bz2

from preprocessing import Preprocessing


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # Load from file
    with bz2.open('pickle_model.pkl', 'rb') as file:
        pickle_model = pickle.load(file)
        df = Preprocessing(csv_file).df.drop(['revenue', 'vote_average'], axis=1, errors='ignore')
        y_hat = pickle_model.predict(df)
        return y_hat

# #
# if __name__ == '__main__':
#     print(predict('test_0.3.csv'))