import pandas as pd
import numpy as np
import json

import generes_data
import productions_data
import keywords_data
import cast_data
import crew_data
import collections_data

from itertools import chain
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer

import ast


# _------------------------------------------------------
# ----- HARDCODE any encoding so we can handle well new ones (ignore them)
# _------------------------------------------------------

def add_missing_encodings(encodings: pd.DataFrame, cols_names: set):
    cols = set(encodings.columns)
    missing = cols_names.difference(cols)
    for col in missing:
        encodings[col] = 0
    return encodings


def parse_column(data: str):
    if not data:
        return data

    try:
        # d = data.replace('\'', '"')  # might need a better solution
        # obj = json.loads(d)
        obj = ast.literal_eval(data)
        return pd.DataFrame([obj]) if isinstance(obj, dict) else pd.DataFrame(obj)
    except Exception as e:
        return data


def get_val(obj, v, default=None, first=False):
    try:
        return obj[v] if not first else obj[v][0]
    except Exception as e:
        return default


def list_col_name(dataframe, col):
    try:
        return dataframe[col].tolist()
    except Exception as e:
        return np.NAN


JSON_COLS = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries',
             'spoken_languages', 'keywords', 'cast', 'crew']
CONVERTERS = {key: parse_column for key in JSON_COLS}

SEED = 13


def to_names_list(sub_df, allowed=None):
    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
        if 'name' in sub_df.columns:
            vals = set(sub_df['name'].tolist())
            return vals.intersection(allowed) if allowed else vals
    return []


class Preprocessing:
    ALLOWED_GENRES = generes_data.data
    BEST_PRODUCTIONS = productions_data.data
    KEYWORDS = keywords_data.data
    KNOWN_CAST = cast_data.data
    KNOWN_CREW = crew_data.data
    COLLECTION_COUNT = collections_data.data

    @staticmethod
    def generate_frequent_values(df: pd.DataFrame, col: str, thresh):
        vals = df[col].apply(to_names_list)
        v = pd.Series(Counter(chain.from_iterable(vals)))
        return {ind for ind in v.index if (v > thresh)[ind]}

    def __init__(self, src, encoding=None):
        self.df: pd.DataFrame = pd.read_csv(src, converters=CONVERTERS, header=0, encoding=encoding)
        self.__process()
        self.df = self.df.fillna(0)

    def split(self, frac):
        pass

    def save(self, path):
        self.df.to_csv(path)

    # ======================================
    # ----------- Processing ---------------
    # ======================================

    def __process(self):
        self.__prepare_belongs_to_collection()  # NEEDS TO BE COMPUTED WITHOUT ALL THE DATA!!
        self.__prepare_genres()
        self.__prepare_homepage()
        self.__prepare_original_language()
        self.__prepare_production_companies()
        self.__prepare_date()
        self.__prepare_keywords()
        self.__prepare_tagline()
        self.__prepare_status()
        self.__prepare_runtime()
        self.__prepare_crew()
        self.__prepare_cast()

        self.__drop_useless()

    def __drop_useless(self):
        self.df.drop(['original_title', 'overview', 'tagline', 'status', 'title', 'original_language',
                      'cast', 'crew', 'keywords', 'spoken_languages', 'id',
                      'production_companies', 'production_countries'], axis=1, inplace=True)

    def __prepare_status(self):
        self.df['is_released'] = (self.df['status'] == 'Released').astype(int)

    def __prepare_tagline(self):
        length = self.df['tagline'].str.len()
        self.df['is_short_tagline'] = ~(length.isna()) & (length < 120)

    def __prepare_keywords(self):
        mlb = MultiLabelBinarizer()
        keywords_list = lambda x: to_names_list(x, Preprocessing.KEYWORDS)
        keywords_encoding = pd.DataFrame(mlb.fit_transform(self.df['keywords'].apply(keywords_list)),
                                         columns=mlb.classes_, index=self.df.index)
        keywords_encoding = add_missing_encodings(keywords_encoding, Preprocessing.KEYWORDS)
        self.df = self.df.join(keywords_encoding)

    def __prepare_genres(self):
        mlb = MultiLabelBinarizer()
        genres_list = lambda x: to_names_list(x, Preprocessing.ALLOWED_GENRES)
        generes_encoding = pd.DataFrame(mlb.fit_transform(self.df['genres'].apply(genres_list)), columns=mlb.classes_,
                                        index=self.df.index)
        generes_encoding = add_missing_encodings(generes_encoding, Preprocessing.ALLOWED_GENRES)
        self.df = self.df.join(generes_encoding)
        self.df.drop(['genres'], axis=1, inplace=True)

    def __prepare_belongs_to_collection(self):
        self.df['belongs_to_collection'] = self.df['belongs_to_collection'].apply(
            lambda x: get_val(x, 'id', first=True))
        self.df['collection_count'] = self.df['belongs_to_collection'].apply(
            lambda x: get_val(Preprocessing.COLLECTION_COUNT, x, default=0))
        self.df.drop(['belongs_to_collection'], axis=1, inplace=True)

    def __prepare_original_language(self):
        # maybe drop completely (A LOT in english others not so much)
        self.df['originally_english'] = (self.df['original_language'] == 'en').astype(int)

    def __prepare_production_companies(self):
        mlb = MultiLabelBinarizer()
        productions_list = lambda x: to_names_list(x, Preprocessing.BEST_PRODUCTIONS)
        productions_encoding = pd.DataFrame(mlb.fit_transform(self.df['production_companies'].apply(productions_list)),
                                            columns=mlb.classes_, index=self.df.index)
        productions_encoding = add_missing_encodings(productions_encoding, Preprocessing.BEST_PRODUCTIONS)
        self.df = self.df.join(productions_encoding)

    def __prepare_date(self):
        release_date = pd.to_datetime(self.df['release_date'], dayfirst=True, errors='coerce')
        self.df['release_month'] = release_date.dt.month

        today = pd.to_datetime('today')
        weeks_delta = (today - release_date) / np.timedelta64(1, 'W')
        years_delta = (today - release_date) / np.timedelta64(1, 'Y')
        self.df['release_very_new'] = (weeks_delta < 2)
        self.df['release_very_old'] = (years_delta > 50)
        self.df.drop(['release_date'], axis=1, inplace=True)

    def __prepare_homepage(self):
        self.df['homepage'] = (~(self.df['homepage'].isna()) & (self.df['homepage'].str.len() > 0)).astype(int)

    def __prepare_runtime(self):
        counts = self.df.groupby('runtime')['runtime'].transform('size')
        self.df['runtime_big'] = (counts > 170).astype(int)
        self.df['runtime_medium'] = ((60 < counts) & (counts < 170)).astype(int)
        self.df['runtime_small'] = ((0 < counts) & (counts < 60)).astype(int)
        self.df.drop(['runtime'], axis=1, inplace=True)

    def __prepare_crew(self):
        mlb = MultiLabelBinarizer()
        crew_list = lambda x: to_names_list(x, Preprocessing.KNOWN_CREW)
        crew_encoding = pd.DataFrame(mlb.fit_transform(self.df['crew'].apply(crew_list)),
                                     columns=mlb.classes_, index=self.df.index)
        crew_encoding = add_missing_encodings(crew_encoding, Preprocessing.KNOWN_CREW)
        self.df = self.df.join(crew_encoding)

    def __prepare_cast(self):
        mlb = MultiLabelBinarizer()
        cast_list = lambda x: to_names_list(x, Preprocessing.KNOWN_CAST)
        cast_encoding = pd.DataFrame(mlb.fit_transform(self.df['cast'].apply(cast_list)),
                                     columns=mlb.classes_, index=self.df.index)
        cast_encoding = add_missing_encodings(cast_encoding, Preprocessing.KNOWN_CAST)
        self.df = self.df.join(cast_encoding)


if __name__ == '__main__':
    pre = Preprocessing('train_0.7.csv')
    # pre.save('./init.csv')
