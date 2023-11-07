from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv
import imageio.v2 as imageio
import os

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
    imageio.mimsave(full_save_path, [image], duration=0.5)  # You can adjust the duration if needed

    # Remove the temporary PNG image
    os.remove(temp_filename)


# Plot for the penguin hot vector dataset
plot_class_distribution(penguins_hotv_data, 'species', 'penguin-hotv-classes.gif')

# Plot for the penguin manual dataset
plot_class_distribution(penguins_manual_data, 'species', 'penguin-manual-classes.gif')

# Plot for the abalone dataset
plot_class_distribution(og_data_abalone, 'Type', 'abalone-classes.gif')

