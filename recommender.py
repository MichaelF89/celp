from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator

# Functie om data van alle steden samen te voegen in 1 DataFrame
def citymerge(var):
    return pd.concat([pd.DataFrame(var[city]) for city in var]).reset_index()

	
# Creeer Utility Matrix en Mean Utility Matrix uit een variabele van data.py (REVIEWS, USERS, BUSINESSES, etc.)
def create_utility_matrix(var):
    df = citymerge(var)

    utility_matrix  = pd.pivot_table(df, index='business_id', columns='user_id', values='stars')

    mean_ultility_matrix = utility_matrix - utility_matrix.mean()
    
    return utility_matrix, mean_ultility_matrix

# Sla Utility Matrix en Mean Utility Matrix op voor later gebruik.    
utility_matrix, mean_utility_matrix = create_utility_matrix(REVIEWS)

# Creeer Similarity Matrix uit Mean Utility Matrix
def similarity(mum):
    return pd.DataFrame(cosine_similarity(mum.fillna(0)), index=mum.index, columns=mum.index).replace(0, np.nan)

similarity_matrix = similarity(mean_utility_matrix)


def select_neighborhood(similarity_matrix, utility_matrix, target_user, target_business):
    """selects all items with similarity > 0"""
    # Controleer of target_user en target_business wel in de matrix zijn te vinden.
    if (target_business in similarity_matrix.index) and (target_user in utility_matrix.columns):

        # Maak een boolean mask van bedrijven die de gebruiker beoordeeld heeft met een similarity hoger dan 0.
        SelectedBusinesses = (similarity_matrix[target_business].index.isin(utility_matrix[target_user].dropna().index)) & (similarity_matrix[target_business] > 0)
    
        # return de bedrijven met de similarity door gebruik te maken van de eerder gecreeerde boolean mask.
        return similarity_matrix[target_business][SelectedBusinesses].sort_values(ascending = False)
    
    # Bij waarden die niet gevonden kunnen worden geef None terug.
    else:
        return pd.Series()


def weighted_mean(neighborhood, utility_matrix, user_id):
    # Controleer of neighborhood wel een Series is en utility_matrix wel een DataFrame, anders return 0.
    if isinstance(neighborhood, pd.Series) and isinstance(utility_matrix, pd.DataFrame):
        # Als neighborhood of de utility_matrix leeg zijn return dan 0.
        if (neighborhood.empty) or (utility_matrix.empty):
            return 0
        
        # Controleer of user_id als kolom te vinden is, anders return 0.
        elif user_id in utility_matrix.columns:
    
            # Gebruik de bovenstaande formule om het gewogen gemiddelde voor de neighborhood te berekenen.
            return ((utility_matrix[user_id] * neighborhood).dropna().sum()) / (neighborhood.sum())

        else:
            return 0
    else:
        return 0
		

def predictions(utility_matrix, similarity_matrix, user_id):
    predictdict = defaultdict()
    for business_id in similarity_matrix.index:
        predictdict[business_id] = weighted_mean(select_neighborhood(similarity_matrix, utility_matrix, user_id, business_id), utility_matrix, user_id)
    
    return predictdict


def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
	
    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

	