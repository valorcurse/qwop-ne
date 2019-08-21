import networkx as nx
import rpyc
import json
from plotly.utils import PlotlyJSONEncoder

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from neat.phenotypes import CNeuralNet

class Visualize():

	def __init__(self) -> None:
		self.queue = rpyc.connect("localhost", 18811)

	def update(self, neuralNetwork: CNeuralNet):
		self.neuralNetwork = neuralNetwork

		edges = {"x": [], "y": []}
		for edge in [e for n in self.neuralNetwork.neurons for e in n.linksIn]:
			edges['x'].append(tuple([edge.fromNeuron.x, edge.toNeuron.x, None]))
			edges['y'].append(tuple([edge.fromNeuron.y, edge.toNeuron.y, None]))
		
		# print(edges)
		
		neurons = {"x": [], "y": []}
		for neuron in self.neuralNetwork.neurons:
			neurons['x'].append(neuron.x)
			neurons['y'].append(neuron.y)

		dataToSend = json.dumps([neurons, edges])
		self.queue.root.send_message(dataToSend)

