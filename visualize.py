import networkx as nx
import rpyc
import json
from plotly.utils import PlotlyJSONEncoder

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from phenotypes import CNeuralNet

class Visualize():

	def __init__(self) -> None:
		self.queue = rpyc.connect("localhost", 18811)

	def update(self, neuralNetwork: CNeuralNet):
		self.neuralNetwork = neuralNetwork

		edges = {"x": [], "y": []}
		for edge in [e for n in self.neuralNetwork.neurons for e in n.linksIn]:
			edges['x'].append(tuple([int(edge.fromNeuron.posX), int(edge.toNeuron.posX), None]))
			edges['y'].append(tuple([int(edge.fromNeuron.posY), int(edge.toNeuron.posY), None]))

		neurons = {"x": [], "y": []}
		for neuron in self.neuralNetwork.neurons:
			neurons['x'].append(int(neuron.posX))
			neurons['y'].append(int(neuron.posY))

		dataToSend = json.dumps([neurons, edges])
		self.queue.root.send_message(dataToSend)

