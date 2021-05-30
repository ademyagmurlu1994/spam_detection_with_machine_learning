import operator
import numpy as np
import utils

class Tokenizer(object):
	def __init__(self, num_words):
		self.num_words = num_words
	
	def fit_on_texts(text:str):
		pass
		
	def texts_to_sequences(self, arrayData):
		text = ''
		for msg in arrayData:
			text +=msg[0] + ' '
			
		words = text.split(' ')
		sequence_dict = {}
		
		for i in words:
			sequence_dict[i]=  0
			
		for i in words:
			sequence_dict[i] +=  1
			
		return sequence_dict 
	
	def sort_dict(self, dict):
		sorted_words = sorted(dict, key=dict.get, reverse=True)
		sorted_dict={}
		
		for w in sorted_words:
			sorted_dict[w] = dict[w]
		return sorted_dict
		
	def get_feature_with_dict(self, dict, max_top):
		feature_dict = {}
		a = 0
		for w in dict:
			if max_top > a:
				feature_dict[w] = dict[w]
				a = a +1
			else:
				break
		return feature_dict
	
	def manipulate_raw_dataset(self, message_list, feature_array):
		manipulated_dataset = np.array([[]])
		'''feature_vektor_values =  []
		feature_vektor_values = [row[0] for row in feature_array]
		feature_vektor_values.append('spam')
		manipulated_dataset.append(feature_vektor_values)'''
		result_of_condition = np.array([])
		index = 0
		for item in message_list:
			feature_of_item = []
			for j in feature_array:
				if utils.is_exist(item[0], j[0]):
					feature_of_item.append(1)
				else:
					feature_of_item.append(0)
			result_of_condition = np.append(result_of_condition, item[1])# spam olup olmadığını belirten değer
			if index == 0:
				manipulated_dataset = np.array(feature_of_item)
				index = index +1
			else:
				manipulated_dataset = np.vstack([manipulated_dataset, feature_of_item])
		return manipulated_dataset, result_of_condition


		
		


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	