import numpy as np

class Perceptron(object):
	"""
	params
	
	eta:float
	n_iter:int
	
	Attributes
	
	w_:1d_array
	errors_: list
	"""
	
	def __init__(self,eta=0.01,n_iter=10):
		self.eta=eta
		self.n_iter=n_iter
	
	def fit(self,X,y):
		""" Fit training data
		Param
		X: [n_samples,n_features]
		y: [n_sampels]
		
		returns
		
		self: object
		"""
		self.w_=np.zeros(1+X.shape[1])
		self.errors_=[]
		
		for _ in range(self.n_iter):
			errors=0
			for xi,target in zip(X,y):
				update=self.eta*(target -self.predict(xi))
				self.w_[1:]+=update *xi
				self.w_[0]+=update
				errors+=int(update != 0.0)
			self.errors_.append(errors)
		return self
	
	def net_input(self,X):
		""" calculate net input """
		return np.dot(X,self.w_[1:]) + self.w_[0]
		
	def predict(self,X):
		"""Return class label after unit step"""
		return np.where(self.net_input(X) >= 0.0,1,-1)
		
		

import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()


		