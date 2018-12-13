"""
logisticregression.py
author: Hiteshi Shah, Rushik Vartak
description: Training and testing the accuracy of a logistic model over the given data
"""

import json
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    recipes = [] # list of all the recipes in the dataset
    cuisines = [] # list of all the cuisines in the dataset
    ingredients = set() # list of individual ingredients used in the recipes
    with open("finaldata.json") as file:
        data = json.load(file)
        for d in data:
            for key, value in d.items():
                if key == "ingredients":
                    recipes.append(value)
                    for item in value:
                        ingredients.add(item)
                elif key == "cuisine":
                    cuisines.append(value)

    # splitting the initial dataset into training and testing datasets
    training_data, testing_data, training_cuisine, testing_cuisine = train_test_split(recipes, cuisines, test_size=0.2)

    # initializing a sparse matrix for the training data
    training_data_matrix = scipy.sparse.dok_matrix((len(training_data), len(ingredients)))

    # changing the value to 1 in the training matrix for every ingredient occurs in a recipe
    for i, recipe in enumerate(training_data):
        for j, ingredient in enumerate(ingredients):
            if ingredient in recipe:
                training_data_matrix[i, j] = 1

    lr = LogisticRegression()
    lr.fit(training_data_matrix, training_cuisine) # fitting the logistic model according to the given training data


    # initializing a sparse matrix for the testing data
    testing_data_matrix = scipy.sparse.dok_matrix((len(testing_data), len(ingredients)))

    # changing the value to 1 in the testing matrix for every ingredient occurs in a recipe
    for i, recipe in enumerate(testing_data):
        for j, ingredient in enumerate(ingredients):
            if ingredient in recipe:
                testing_data_matrix[i, j] = 1

    result = lr.predict(testing_data_matrix) # returns the predicted outcome per sample in the testing data

    print(classification_report(testing_cuisine, result))

main()