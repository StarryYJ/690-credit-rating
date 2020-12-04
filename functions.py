import pandas as pd
from numpy import nan
import os


def read_data(path, names):
	"""
	Read all csv files under given path, and store to global variables with coresponding names

	:param path: address of a folder stores and only stores data files to be read
	:param names: names of all global variables
	:return: variable names of read_in data
	"""
	files = os.listdir(path)
	filenames = [filename[:-4] for filename in files]
	for filename in files:
		df = pd.read_csv(path + '/' + filename)
		names[filename[:-4]] = df
	return filenames


def summary(df: pd.DataFrame):
	"""
	Will print column names in the data frame, and show the first five lines of it.

	:param df: a pandas data frame
	:return: None
	"""
	print('Columns inside: ')
	n = len(df.columns)
	text = ''
	for i in range(n):
		text = text + df.columns[i]
		if i != 0 & i % 8 == 0:
			print(text)
			text = ''
	print('The first five lines of the data frame:')
	print(df.iloc[:5, :])


def missing_detect(df: pd.DataFrame):
	"""
	Detect and return distribution of missing value of the input data frame(in a descending order)

	:param df: a pandas data frame
	:return: a pandas frame describing missing value situation about the input df
	"""
	missing = df.isnull().sum()
	missing_pct = missing / len(df)
	out_df = pd.concat([missing, missing_pct], axis=1)
	out_df.columns = ['Missing value', 'Percent of missing value']
	out_df = out_df[out_df['Missing value'] != 0]
	out_df = out_df.sort_values('Percent of missing value', ascending=False)
	return out_df


def quality_quantity_classify(df: pd.DataFrame):
	qualitative = []
	quantitative = []
	for column in df.columns:
		i = 0
		while df[column].iloc[i] is nan:
			i += 1
		if type(df[column].iloc[i]) is str:
			qualitative.append(column)
		else:
			quantitative.append(column)
	return qualitative, quantitative


def similarity(a, b):
	count = 0
	for item in a:
		if item in b:
			count += 1
	return count/len(b)


















