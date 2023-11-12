from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import imageio.v2 as imageio
import os
import shutil

# The commented out code is for clearing the directory in case I need it to be cleared
# def clear_directory(directory):
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')
#
#
# clear_directory('2.5')

########################################################################################################################
# Step 2 ###############################################################################################################
########################################################################################################################

# Setting an option for printing all the columns when printing in the console
pd.set_option('display.max_columns', None)

# Reading the data from penguins.csv
og_data_penguins = pd.read_csv('penguins.csv')

# Converting categorical variables into 1-hot vectors
penguins_hotv_data = pd.get_dummies(og_data_penguins, columns=['island', 'sex'])

# Converting categorical variables (hardcoded)
penguins_manual_data = og_data_penguins

# Island values conversion
island_values = penguins_manual_data['island'].unique()
island_mapping = {name: i for i, name in enumerate(island_values)}
penguins_manual_data['island'] = penguins_manual_data['island'].map(island_mapping)

# Sex values conversion
sex_values = penguins_manual_data['sex'].unique()
sex_mapping = {name: i for i, name in enumerate(sex_values)}
penguins_manual_data['sex'] = penguins_manual_data['sex'].map(sex_mapping)

# Writing datasets to csv files
penguins_hotv_data.to_csv('penguins/penguins_hotv.csv', index=False)
penguins_manual_data.to_csv('penguins/penguins_manual.csv', index=False)

########################################################################################################################
########################################################################################################################

# Reading the data from abalone.csv
og_data_abalone = pd.read_csv('abalone.csv')

# Writing datasets to csv files
og_data_abalone.to_csv('abalone/abalone_no_conversion.csv', index=False)


########################################################################################################################
# Step 2.2 #############################################################################################################
########################################################################################################################

def plot_class_distribution(data, output_col, save_as):
    # Ensure the directory exists
    gif_directory = "GIF"
    if not os.path.exists(gif_directory):
        os.makedirs(gif_directory)

    # Prepend directory to save_as
    full_save_path = os.path.join(gif_directory, save_as)

    class_counts = data[output_col].value_counts(normalize=True) * 100
    class_counts.plot(kind='bar', color='skyblue')
    plt.ylabel('Percentage')
    plt.title(f'Percentage of Instances in Each {output_col} Class')
    plt.grid(axis='y')

    # Save the figure temporarily as a PNG
    temp_filename = "temp_image.png"
    plt.savefig(temp_filename, format='png')

    # Read the PNG image and save as a GIF
    image = imageio.imread(temp_filename)
    imageio.mimsave(full_save_path, [image], duration=0.5)

    # Remove the temporary PNG image
    os.remove(temp_filename)


# Plot for the penguin hot vector dataset
plot_class_distribution(penguins_hotv_data, 'species', 'penguin-hotv-classes.gif')

# Plot for the penguin manual dataset
plot_class_distribution(penguins_manual_data, 'species', 'penguin-manual-classes.gif')

# Plot for the abalone dataset
plot_class_distribution(og_data_abalone, 'Type', 'abalone-classes.gif')

########################################################################################################################
# Step 2.3 #############################################################################################################
########################################################################################################################

# Penguins dataset with 1-hot vector encoding
# Separation of label and features
penguins_hotv_features = penguins_hotv_data.drop('species', axis=1)
penguins_hotv_label = penguins_hotv_data['species']

# Splitting up data into training set and testing set for features and labels (1-hot vector encoding)
penguins_hotv_features_train, penguins_hotv_features_test, penguins_hotv_label_train, penguins_hotv_label_test = train_test_split(
    penguins_hotv_features, penguins_hotv_label
)

# Penguins dataset with manual categorical encoding
# Separation of label and features
penguins_manual_features = penguins_manual_data.drop('species', axis=1)
penguins_manual_label = penguins_manual_data['species']

# Splitting up data into training set and testing set for features and labels (manual categorical encoding)
penguins_manual_features_train, penguins_manual_features_test, penguins_manual_label_train, penguins_manual_label_test = train_test_split(
    penguins_manual_features, penguins_manual_label
)

# Abalone dataset
# Separation of label and features
abalone_features = og_data_abalone.drop('Type', axis=1)
abalone_label = og_data_abalone['Type']

# Splitting up data into training set and testing set for features and labels
abalone_features_train, abalone_features_test, abalone_label_train, abalone_label_test = train_test_split(
    abalone_features, abalone_label
)


########################################################################################################################
# Step 2.4 #############################################################################################################
########################################################################################################################

# I added the Step 5 function here because it caused errors when I initialize it at the end of the file
# like if it was visible because it is not initialized yet
def evaluate_and_save_decision_tree_model(model, features_test, labels_test, model_description, filename):
    # Create the directory if it doesn't exist
    directory = '2.5'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    with open(file_path, 'a') as file:
        # Separator and Model Description (A)
        file.write('-' * 50 + '\n')
        file.write(f"(A) Model Evaluation: {model_description}\n")

        # Predict the labels for the test set
        labels_pred = model.predict(features_test)

        # Confusion Matrix (B)
        file.write("\n(B) Confusion Matrix:\n")
        file.write(str(confusion_matrix(labels_test, labels_pred)) + '\n')

        # Precision, Recall, and F1-Measure for each class (C)
        file.write("\n(C) Classification Report:\n")
        file.write(classification_report(labels_test, labels_pred) + '\n')

        # Accuracy, Macro-average F1 and Weighted-average F1 of the model (D)
        accuracy = accuracy_score(labels_test, labels_pred)
        report = classification_report(labels_test, labels_pred, output_dict=True)
        macro_avg_f1 = report['macro avg']['f1-score']
        weighted_avg_f1 = report['weighted avg']['f1-score']

        file.write("\n(D) Model Performance Metrics:\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Macro-average F1: {macro_avg_f1:.2f}\n")
        file.write(f"Weighted-average F1: {weighted_avg_f1:.2f}\n")
        file.write('-' * 50 + '\n')


########################################################################################################################
# 4.a ##################################################################################################################
########################################################################################################################
def train_and_save_base_decision_tree(features, labels, max_depth, directory, filename, description, file_name):
    # Create and train the decision tree model
    decision_tree_model = DecisionTreeClassifier(max_depth=max_depth)
    decision_tree_model.fit(features, labels)

    # Save the decision tree model
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Plot and save the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree_model, filled=True, feature_names=features.columns,
              class_names=np.unique(labels).astype(str), rounded=True)
    plt.savefig(os.path.join(directory, filename), format='png', bbox_inches='tight')
    plt.close()

    # For Step 2.5
    evaluate_and_save_decision_tree_model(decision_tree_model, features, labels, description,
                                          file_name)


train_and_save_base_decision_tree(abalone_features_train, abalone_label_train, 3, '4.a',
                                  'Base-DT-abalone.png', "Base-DT-Abalone Max Depth 3", "abalone-performance.txt")
train_and_save_base_decision_tree(penguins_hotv_features_train, penguins_hotv_label_train, None, '4.a',
                                  'Base-DT-hotv-penguins.png', 'Base-DT-penguins-hotv', 'penguins-hotv-performance.txt')
train_and_save_base_decision_tree(penguins_manual_features_train, penguins_manual_label_train, None, '4.a',
                                  'Base-DT-manual-penguins.png', 'Base-DT-penguins-manual',
                                  'penguins-manual-performance.txt')


########################################################################################################################
# 4.b ##################################################################################################################
########################################################################################################################

def perform_grid_search_and_visualize(features, labels, param_grid, base_dir, top_filename, scoring, cv, description,
                                      file_name):
    # Initialize and fit the GridSearchCV
    dt_grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring=scoring, cv=cv)
    dt_grid_search.fit(features, labels)

    # Get the best decision tree model
    best_dt_model = dt_grid_search.best_estimator_

    # Visualize and save the best decision tree model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    plt.figure(figsize=(20, 10))
    plot_tree(best_dt_model, filled=True, feature_names=features.columns, class_names=np.unique(labels).astype(str),
              rounded=True)
    plt.savefig(os.path.join(base_dir, top_filename), format='png', bbox_inches='tight')
    plt.close()

    description = description, " scoring=", scoring, " cv=", cv

    evaluate_and_save_decision_tree_model(best_dt_model, features, labels, description, file_name)


# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 15, None],
    'min_samples_split': [5, 50, 200]
}

perform_grid_search_and_visualize(abalone_features_train, abalone_label_train, param_grid, '4.b',
                                  'Top-DT-abalone.png', 'accuracy', 5, 'Top-DT-abalone', 'abalone-performance.txt')
perform_grid_search_and_visualize(penguins_hotv_features_train, penguins_hotv_label_train, param_grid, '4.b',
                                  'Top-DT-hotv-penguins.png', 'accuracy', 5, 'Top-DT-hotv-penguins',
                                  'penguins-hotv-performance.txt')
perform_grid_search_and_visualize(penguins_manual_features_train, penguins_manual_label_train, param_grid, '4.b',
                                  'Top-DT-manual-penguins.png', 'accuracy', 5, 'Top-DT-manual-penguins',
                                  'penguins-manual-performance.txt')


########################################################################################################################
# 4.c ##################################################################################################################
########################################################################################################################

def train_base_mlp(features, labels, description, file_name):
    # Create the MLP model
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')

    # Train the model
    mlp_model.fit(features, labels)

    description = description, ' activation=', mlp_model.activation, ' solver=', mlp_model.solver

    evaluate_and_save_decision_tree_model(mlp_model, features, labels, description, file_name)

    return mlp_model


# Train the model for each dataset
mlp_abalone = train_base_mlp(abalone_features_train, abalone_label_train, 'mlp-abalone', 'abalone-performance.txt')
mlp_penguins_hotv = train_base_mlp(penguins_hotv_features_train, penguins_hotv_label_train, 'mlp-hotv-penguins',
                                   'penguins-hotv-performance.txt')
mlp_penguins_manual = train_base_mlp(penguins_manual_features_train, penguins_manual_label_train, 'mlp-manual-penguins',
                                     'penguins-manual-performance.txt')

########################################################################################################################
# 4.d ##################################################################################################################
########################################################################################################################

# Define the parameter grid
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],  # logistic for sigmoid
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd']
}


def perform_mlp_grid_search(features, labels, description, file_name):
    mlp = MLPClassifier()
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(features, labels)

    description = description, ' scoring=', grid_search.scoring, ' cv=', grid_search.cv, ' n_jobs=', grid_search.n_jobs

    evaluate_and_save_decision_tree_model(grid_search, features, labels, description, file_name)

    return grid_search.best_estimator_


# Apply the grid search to each dataset
top_mlp_abalone = perform_mlp_grid_search(abalone_features_train, abalone_label_train, 'Top-mlp-abalone',
                                          'abalone-performance.txt')
top_mlp_penguins_hotv = perform_mlp_grid_search(penguins_hotv_features_train, penguins_hotv_label_train,
                                                'Top-mlp-hotv-penguins', 'penguins-hotv-performance.txt')
top_mlp_penguins_manual = perform_mlp_grid_search(penguins_manual_features_train, penguins_manual_label_train,
                                                  'Top-mlp-manual-penguins', 'penguins-manual-performance.txt')

########################################################################################################################
# Step 2.5 #############################################################################################################
########################################################################################################################

# Step 2.5 has been moved to Step 2.4
