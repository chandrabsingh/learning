#!/usr/bin/env python
# coding: utf-8

# # Store Sales Forecasting with TensorFlow

# Goal -  The project involves making sales predictions for various stores of a large retail corporation, based on past sales of these stores, using TensorFlow
# 
# The dataset comes in three CSV files: weekly_sales.csv, features.csv, and stores.csv
# 

# ## Preliminary Data Analysis
# ### The Dataset

# In[1]:


import pandas as pd
train_df = pd.read_csv('./storeSalesForecasting/train.csv')
print(train_df)


# In[2]:


def read_dataframes():
    train_df = pd.read_csv('./storeSalesForecasting/train.csv')
    features_df = pd.read_csv('./storeSalesForecasting/features.csv')
    stores_df = pd.read_csv('./storeSalesForecasting/stores.csv')
    return (train_df, features_df, stores_df)

(train_df, features_df, stores_df) = read_dataframes()


# In[3]:


general_features = features_df.columns

print(general_features)
print('General Features: {}\n'.format(general_features.tolist()))

store_features = stores_df.columns
print('Store Features: {}'.format(store_features.tolist()))


# In[4]:


merged_features = features_df.merge(stores_df, on='Store')

print(merged_features)


# ### Missing Features

# In[5]:


na_values = pd.isna(merged_features) # Boolean DataFrame
na_features = na_values.any() # Boolean Series
print(na_features)


# ### Dropping Features

# In[6]:


print(f"Total = {len(na_values)}")
print(f"MarkDown1 = {sum(na_values['MarkDown1'])}")
print(f"MarkDown2 = {sum(na_values['MarkDown2'])}")
print(f"MarkDown3 = {sum(na_values['MarkDown3'])}")
print(f"MarkDown4 = {sum(na_values['MarkDown4'])}")
print(f"MarkDown5 = {sum(na_values['MarkDown5'])}")
print(f"CPI = {sum(na_values['CPI'])}")
print(f"Unemployment = {sum(na_values['Unemployment'])}")


# In[7]:


# Drop MarkDown columns, while drop missing CPI and Unemployment missing values
markdowns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', ]
merged_features = merged_features.drop(columns=markdowns)
merged_features.columns.tolist()


# ### Filling in Data
# - find missing CPI and Unemployment values
# - fill it with previous row values

# In[8]:


merged_features.info()


# In[9]:


# merged_features[['CPI', 'Unemployment']]
import numpy as np

na_cpi_int = na_values['CPI'].astype(int)
na_indexes_cpi = na_cpi_int.to_numpy().nonzero()[0]
na_une_int = na_values['Unemployment'].astype(int)
na_indexes_une = na_une_int.to_numpy().nonzero()[0]

print(np.array_equal(na_indexes_cpi, na_indexes_une)) # Same index for both fields


# In[10]:


na_indexes = na_indexes_cpi
na_rows = merged_features.iloc[na_indexes]
print(na_rows['Date'].unique()) # missing value weeks
print(merged_features['Date'].unique()[-13:]) # last 13 weeks
print(na_rows.groupby('Store').count()['Date'].unique())


# In[11]:


# na_rows
# merged_features
# na_rows.groupby('Store').count()['Date'].unique() # 13
# merged_features.groupby('Store').count()['Date'].unique() # 182
# merged_features['Date'].unique()

# Fill in the values
# merged_features
print(na_indexes[0])  # first missing value row index
# print(na_indexes)
print()

first_missing_row = merged_features.iloc[169]
print(first_missing_row[['Date','CPI','Unemployment']])
print()

final_val_row = merged_features.iloc[168]
print(final_val_row[['Date','CPI','Unemployment']])
print()

cpi_final_val = merged_features.at[168, 'CPI']
une_final_val = merged_features.at[168, 'Unemployment']
merged_features.at[169, 'CPI'] = cpi_final_val
merged_features.at[169, 'Unemployment'] = une_final_val

new_row = merged_features.iloc[169]
print(new_row[['Date','CPI','Unemployment']])
print()


# In[12]:


def impute_data(merged_features, na_indexes_cpi, na_indexes_une):
    for i in na_indexes_cpi:
        merged_features.at[i, 'CPI'] = merged_features.at[i-1, 'CPI']
    for i in na_indexes_une:
        merged_features.at[i, 'Unemployment'] = merged_features.at[i-1, 'Unemployment']
        
impute_data(merged_features, na_indexes_cpi, na_indexes_une)
na_values2 = pd.isna(merged_features) # Boolean DataFrame
na_features2 = na_values2.any() # Boolean Series
print(na_features2)


# ### Merging Data

# In[13]:


print(train_df.columns.tolist())
print(merged_features.columns.tolist())


# In[14]:


# Common columns - ['Store', 'Date', 'IsHoliday']
features = ['Store', 'Date', 'IsHoliday']
final_dataset = train_df.merge(merged_features, on=features)
final_dataset
# print(train_df['Store'].unique())
# print(train_df['Date'].unique())# Less dates
# print(train_df['IsHoliday'].unique())

# print(merged_features['Store'].unique())
# print(merged_features['Date'].unique()) # More dates
# print(merged_features['IsHoliday'].unique())


# In[15]:


# 'Date' is not used in ML model, so drop it
final_dataset = final_dataset.drop(columns=['Date'])


# ### Categorical Data

# In[16]:


final_dataset.describe(include='all')


# In[17]:


print(final_dataset['Type'].unique())
print(final_dataset['Dept'].unique())


# In[18]:


# Cast IsHoliday to type int - features must be integer, float, or string
final_dataset = final_dataset.astype({"IsHoliday": int})
final_dataset.info()


# ### 1. Plot data

# In[19]:


import matplotlib.pyplot as plt

plot_df = final_dataset[['Weekly_Sales', 'Temperature']]
rounded_temp = plot_df['Temperature'].round()
plot_df = plot_df.groupby(rounded_temp).mean()
plot_df.plot.scatter(x='Temperature', y='Weekly_Sales')
plt.title('Temperature vs. Weekly Sales')
plt.xlabel('Temperature (Fahrenheit)')
plt.ylabel('Avg Weekly Sales (Dollars)')
plt.show()


# In[20]:


plot_df = final_dataset[['Weekly_Sales', 'Fuel_Price']]
rounded_temp = plot_df['Fuel_Price'].round(2)
plot_df = plot_df.groupby(rounded_temp).mean()
plot_df.plot.scatter(x='Fuel_Price', y='Weekly_Sales')
plt.title('Fuel Price vs. Weekly Sales')
plt.xlabel('Fuel Price ($/gallon, Nearest Hundredth)')
plt.ylabel('Avg Weekly Sales (Dollars)')
plt.show()
# plot_df['Fuel_Price'].round(1).unique()


# In[21]:


plot_df = final_dataset[['Weekly_Sales', 'Type']]
plot_df = plot_df.groupby('Type').mean()
print(plot_df)
plot_df.plot.bar()
plt.title('Store Type vs. Weekly Sales')
plt.xlabel('Type')
plt.ylabel('Avg Weekly Sales (Dollars)')
plt.show()


# ## 1. Data Processing
# - create efficient input pipeline
# 
# 
# ### 1. Splitting Datasets
# - training and evaluation
# - set proportions
# - removing systematic trends
# 

# In[22]:


# from random import randrange
# print(randrange(10))


# X1 = np.random.randint(low=0, high=10, size=(15,))
# print(X1)

import random
# random.sample(range(100), 10)


# In[23]:


def split_train_eval(final_dataset):
    len_dataset = len(final_dataset)
    len_train_dataset = round(0.9*len_dataset)
    len_evaluate_dataset = len_dataset - len_train_dataset
    rnd_train_samples = random.sample(range(len_dataset), len_train_dataset)
    rnd_evaluate_samples = list(set(range(len_dataset)) - set(rnd_train_samples))
    train_dataset = final_dataset.iloc[rnd_train_samples]
    evaluate_dataset = final_dataset.iloc[rnd_evaluate_samples]
    return (train_dataset, evaluate_dataset)

train_set, eval_set = split_train_eval(final_dataset)


# In[24]:


train_set


# In[25]:


eval_set


# ### Integer Features

# In[26]:


# Add the integer features of a DataFrame’s row to a feature dictionary

import tensorflow as tf

def add_int_features(dataset_row, feature_dict):
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    for feature_name in int_vals:
        list_val = tf.train.Int64List(value=[dataset_row[feature_name]])
        feature_dict[feature_name] = tf.train.Feature(int64_list = list_val)


# ### Float Features

# In[27]:


# Add the float Feature objects to the feature dictionary
def add_float_features(dataset_row, feature_dict, has_labels):
    # We only use the 'Weekly_Sales' feature if has_labels is True. 
    # This is because the 'Weekly_Sales' feature represents the label used 
    # in training/evaluating the machine learning model, which is not present when making predictions.

    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', ]
    if has_labels:
        float_vals.append('Weekly_Sales')
    for feature_name in float_vals:
        list_val = tf.train.FloatList(value = [dataset_row[feature_name]])
        feature_dict[feature_name] = tf.train.Feature(float_list = list_val)


# ### String Features

# In[28]:


s = 'hello world'
byte_s = s.encode()  # byte string
bytes_list = tf.train.BytesList(value=[byte_s])
feature = tf.train.Feature(bytes_list=bytes_list)
print(feature)


# In[29]:


# Create an Example object from a pandas DataFrame row
def create_example(dataset_row, has_labels):
    feature_dict = {}
    add_int_features(dataset_row, feature_dict)
    add_float_features(dataset_row, feature_dict, has_labels)
    # CODE HERE
    byte_type = dataset_row['Type']
    list_val = tf.train.BytesList(value = [byte_type.encode()])
    feature_dict['Type'] = tf.train.Feature(bytes_list = list_val)
    features_obj = tf.train.Features(feature = feature_dict)
    return tf.train.Example(features = features_obj)


# ### Writing TFRecords

# In[30]:


# Write serialized Example objects(the training and evaluation set data) to a TFRecords file

def write_tfrecords(dataset, tfrecords_file, has_labels = True):
    writer = tf.io.TFRecordWriter(tfrecords_file)
    for i in range(len(dataset)):
        example = create_example(dataset.iloc[i], has_labels)
        writer.write(example.SerializeToString())
    writer.close()


# In[31]:


# train_set is the training DataFrame
write_tfrecords(train_set, 'train.tfrecords')

# eval_set is the evaluation DataFrame
write_tfrecords(eval_set, 'eval.tfrecords')


# ### Example Spec
# - the data is stored as serialized Example objects in TFRecords file
# - create an Example spec which parses the serialized examples in the input pipeline 
# - the example spec gives specifications on each of the dataset’s features, specifically the shape and type of the feature’s values.

# In[32]:


example_spec = {}
example_spec['Store'] = tf.io.FixedLenFeature((), tf.int64)
example_spec['CPI'] = tf.io.FixedLenFeature((), tf.float32)
example_spec['Type'] = tf.io.FixedLenFeature((), tf.string)


# In[33]:


# Create the spec used when parsing the Example object
def create_example_spec(has_labels):
    example_spec = {}
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if has_labels:
        float_vals.append('Weekly_Sales')
    
    for feature_name in int_vals:
        example_spec[feature_name] = tf.io.FixedLenFeature((), tf.int64)
    for feature_name in float_vals:
        example_spec[feature_name] = tf.io.FixedLenFeature((), tf.float32)
    example_spec['Type'] = tf.io.FixedLenFeature((), tf.string)
    return example_spec


# ### Parsing Example
# - parse feature data from serialized Example objects

# In[34]:


example_spec = create_example_spec(True)

# Parsing feature data from a serialized Example (ser_ex) using its corresponding Example spec (example_spec).
# parsed_example = tf.io.parse_single_example(ser_ex, example_spec)
# print(parsed_example)


# In[35]:


# Helper function to convert serialized Example objects into features
def parse_features(ser_ex, example_spec, has_labels):
    # CODE HERE
    parsed_features = tf.io.parse_single_example(ser_ex, example_spec)
    # The 'Weekly_Sales' feature is not actually used as an input for the machine learning model. 
    # Instead, it is used as a label during training and evaluation.
    features = {k: parsed_features[k] for k in parsed_features if k!='Weekly_Sales'}
    if not has_labels:
        return features
    label = parsed_features['Weekly_Sales']
    return (features, label)
    


# ### TFRecords Dataset

# In[36]:


# Create a TFRecords dataset for the input pipeline
# Parsing feature data from a serialized Example (ser_ex) using its corresponding Example spec (example_spec)

train_file = 'train.tfrecords'
eval_file = 'eval.tfrecords'
train_dataset = tf.data.TFRecordDataset(train_file)
eval_dataset = tf.data.TFRecordDataset(eval_file)


# In[37]:


# Using the functions from above to modify the TFRecords datasets.

example_spec = create_example_spec(True)
parse_fn = lambda ser_ex: parse_features(ser_ex, example_spec, True)
train_dataset = train_dataset.map(parse_fn)
eval_dataset = eval_dataset.map(parse_fn)


# In[38]:


print(eval_dataset)


# In[39]:


example_spec


# In[40]:


# Configure the dataset - Shuffling datasets is always a good idea for training and evaluation, 
# since it randomizes the order in which the data is passed into the machine learning model. 

train_dataset = train_dataset.shuffle(421570)
eval_dataset = eval_dataset.shuffle(421570)


# In[41]:


type(train_dataset)


# In[42]:


# Repeating the datasets indefinitely. The training will run until we manually kill the process.

# We also want to run training indefinitely, until we decide to kill the model 
# running process manually (i.e. with CTRL+C or CMD+C). Evaluation is done with a 
# single run-through of the dataset.

train_dataset = train_dataset.repeat()
print(train_dataset)


# In[43]:


# set the dataset batch sizes, so that each training/evaluation step contains multiple data observations

train_dataset = train_dataset.batch(100)
eval_dataset = eval_dataset.batch(20)


# ### Numeric columns

# In[44]:


def add_numeric_columns(feature_columns):
    numeric_features = ['Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for feature_name in numeric_features:
        feature_col = tf.feature_column.numeric_column(feature_name, shape=())
        feature_columns.append(feature_col)

# Add the numeric feature columns to the list of dataset feature columns
dataset_feature_columns = []
add_numeric_columns(dataset_feature_columns)
print(dataset_feature_columns)


# In[45]:


from pprint import pprint
pprint(dataset_feature_columns)
# help(tf.feature_column.numeric_column)


# ### Indicator Columns
# - Process the indicator feature columns used for the machine learning model’s input layer
#   - One-hot indicators
#     - 'IsHoliday' (0 and 1) 
#     - 'Type' (A, B, C)
#   - Categorical column base
# 

# In[46]:


# Categorical columns for the 'IsHoliday' and 'Type' features.

type_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'Type', ['A', 'B', 'C'], dtype=tf.string)
holiday_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'IsHoliday', [0, 1], dtype=tf.int64)


# In[47]:


# Converting categorical columns to indicator feature columns.

type_feature_col = tf.feature_column.indicator_column(type_col)
holiday_feature_col = tf.feature_column.indicator_column(holiday_col)


# In[48]:


# Add the indicator feature columns to the list of feature columns
def add_indicator_columns(final_dataset, feature_columns):
    indicator_features = ['IsHoliday', 'Type']
    for feature_name in indicator_features:
        # CODE HERE
        dtype = tf.int64 if feature_name == 'IsHoliday' else tf.string
        vocab_list = list(final_dataset[feature_name].unique())
        vocab_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab_list, dtype = dtype)
        feature_col = tf.feature_column.indicator_column(vocab_col)
        feature_columns.append(feature_col)


# In[49]:


# help(tf.feature_column.indicator_column)


# ### Embedding Columns

# In[50]:


stores = list(range(1, 46)) # There are 45 stores in the dataset, labeled from 1-45
stores_col = tf.feature_column.categorical_column_with_vocabulary_list('StoreID', stores, dtype=tf.int64)
embedding_dim = int(45**0.25)  # 4th root - 
# set the vector dimension to anything, but a good rule of thumb is to set it equal to the 
# 4th root of the size of the vocabulary list.

feature_col = tf.feature_column.embedding_column(stores_col, embedding_dim)


# In[51]:


# Add the embedding feature columns to the list of feature columns
def add_embedding_columns(final_dataset, feature_columns):
    embedding_features = ['Store', 'Dept']
    for feature_name in embedding_features:
        vocab_list = list(final_dataset[feature_name].unique())
        vocab_col = tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocab_list, dtype=tf.int64)
        embedding_dim = int(len(vocab_list) ** 0.25)
        
        feature_col = tf.feature_column.embedding_column(vocab_col, embedding_dim)
        feature_columns.append(feature_col)


# ### Model Input Layer
# - Aggregate the feature columns for the machine learning model’s input layer

# In[52]:


def create_feature_columns(final_dataset):
    feature_columns = []
    add_numeric_columns(feature_columns)
    add_indicator_columns(final_dataset, feature_columns)
    add_embedding_columns(final_dataset, feature_columns)
    return feature_columns

feature_columns = create_feature_columns(final_dataset)
pprint(feature_columns)


# ## Model Predictions
# - predicting weekly sales for various retail stores
# 
# ### Model Layers

# In[53]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions


# ### Regression Function

# In[54]:


# inputs = input_layer(features, cols)


# In[55]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        
# help(tf.compat.v1.feature_column.input_layer)


# ### Training mode

# In[56]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# ### Evaluation Mode

# In[57]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # regression function’s evaluation code
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)


# ### Prediction Mode

# In[58]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # regression function’s evaluation code
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        # regression function’s prediction code
        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_info = {'predictions': batch_predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=prediction_info)


# ### Regression Model
# - create an Estimator object for the regression model

# In[59]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # regression function’s evaluation code
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        # regression function’s prediction code
        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_info = {'predictions': batch_predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=prediction_info)
        
    def create_regression_model(self, ckpt_dir):
        # configuration for training logs the loss and global step values every 5000 training steps
        config = tf.estimator.RunConfig(log_step_count_steps=5000)
        regression_model = tf.estimator.Estimator(self.regression_fn, config=config, model_dir=ckpt_dir)
        return regression_model


# ### Model Training
# - Train the regression model using train.tfrecords file

# In[60]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # regression function’s evaluation code
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        # regression function’s prediction code
        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_info = {'predictions': batch_predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=prediction_info)
        
    def create_regression_model(self, ckpt_dir):
        # configuration for training logs the loss and global step values every 5000 training steps
        config = tf.estimator.RunConfig(log_step_count_steps=5000)
        regression_model = tf.estimator.Estimator(self.regression_fn, config=config, model_dir=ckpt_dir)
        return regression_model
    
    def run_regression_training(self, ckpt_dir, batch_size, num_training_steps=None):
        # Setting steps to None will run training until is manually terminated
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda:create_tensorflow_dataset('train.tfrecords', batch_size)
        regression_model.train(input_fn, steps=num_training_steps)


# ### Model Evaluation

# In[61]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputts)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # regression function’s evaluation code
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        # regression function’s prediction code
        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_info = {'predictions': batch_predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=prediction_info)
        
    def create_regression_model(self, ckpt_dir):
        # configuration for training logs the loss and global step values every 5000 training steps
        config = tf.estimator.RunConfig(log_step_count_steps=5000)
        regression_model = tf.estimator.Estimator(self.regression_fn, config=config, model_dir=ckpt_dir)
        return regression_model
    
    def run_regression_training(self, ckpt_dir, batch_size, num_training_steps=None):
        # Setting steps to None will run training until is manually terminated
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda:create_tensorflow_dataset('train.tfrecords', batch_size)
        regression_model.train(input_fn, steps=num_training_steps)
        
    def run_regression_eval(self, ckpt_dir):
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda:create_tensorflow_dataset('eval.tfrecords', 50, training=False)
        return regression_model.evaluate(input_fn)


# ### Making Predictions
# - using regression model to make predictions on unlabeled test dataset

# In[62]:


class SalesModel(object):
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def model_layers(self, inputs):
        layer = inputs
        for num_nodes in self.hidden_layers:
            layer = tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu)(layer)
        batch_predictions = tf.keras.layers.Dense(1)(layer)
        return batch_predictions

    def regression_fn(self, features, labels, mode, params):
        feature_columns = create_feature_columns()
        inputs = tf.compat.v1.feature_column.input_layer(features, feature_columns)
        batch_predictions= self.model_layers(inputs)
        predictions = tf.squeeze(batch_predictions)
        if labels is not None:
            loss = tf.compat.v1.losses.absolute_difference(labels, predictions)
        # regression function’s training code
        if mode == tf.estimator.ModeKeys.TRAIN:
            # to keep track of total number of training steps during different training runs
            global_step = tf.compat.v1.train.get_or_create_global_step()
            # minimize model's loss during training using ADAM optimization method
            adam = tf.compat.v1.train.AdamOptimizer()
            train_op = adam.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # regression function’s evaluation code
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        # regression function’s prediction code
        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction_info = {'predictions': batch_predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=prediction_info)
        
    def create_regression_model(self, ckpt_dir):
        # configuration for training logs the loss and global step values every 5000 training steps
        config = tf.estimator.RunConfig(log_step_count_steps=5000)
        regression_model = tf.estimator.Estimator(self.regression_fn, config=config, model_dir=ckpt_dir)
        return regression_model
    
    def run_regression_training(self, ckpt_dir, batch_size, num_training_steps=None):
        # Setting steps to None will run training until is manually terminated
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda:create_tensorflow_dataset('train.tfrecords', batch_size)
        regression_model.train(input_fn, steps=num_training_steps)
        
    def run_regression_eval(self, ckpt_dir):
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda:create_tensorflow_dataset('eval.tfrecords', 50, training=False)
        return regression_model.evaluate(input_fn)
    
    def run_regression_predict(self, ckpt_dir, data_file):
        regression_model = self.create_regression_model(ckpt_dir)
        input_fn = lambda:create_tensorflow_dataset(data_file, 1, training=False, has_labels=False)
        predictions = regression_model.predict(input_fn)
        pred_list = []
        for pred_dict in predictions:
            pred_list.append(pred_dict['predictions'][0])
        return pred_list
    


# ## ⛏ Result Summary

# In[64]:


# TODO: Execute it!!

# run_regression_predict('test.tfrecords')
# # Making Predictions
# input_fn = lambda:create_tensorflow_dataset('test.tfrecords', 1, training=False, has_labels=False)
# predictions = regression_model.predict(input_fn)

ckpt_dir = './storeSalesForecasting/'
batch_size = 50
data_file = 'eval.tfrecords'

nn = SalesModel(2)
# nn.run_regression_training(ckpt_dir, batch_size, num_training_steps=None)
# eval_list = nn.run_regression_eval(ckpt_dir)
# pred_list = nn.run_regression_predict(ckpt_dir, data_file)


# In[ ]:





# In[ ]:





# In[ ]:


# (fast) ml_examples >> tensorboard --logdir=~/anaconda3/workingDir/learnings/ml_examples/

# NOTE: Using experimental fast data loading logic. To disable, pass
#     "--load_fast=false" and report issues on GitHub. More details:
#     https://github.com/tensorflow/tensorboard/issues/4784

# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.8.0 at http://localhost:6006/ (Press CTRL+C to quit)


# In[ ]:





# - https://medium.com/nerd-for-tech/walmart-sales-time-series-forecasting-using-deep-learning-e7a5d47c448b
# - https://github.com/abhinav-bhardwaj/Walmart-Sales-Time-Series-Forecasting-Using-Machine-Learning/blob/master/Walmart_Time_Series_Forecast.ipynb
# 

# In[ ]:




