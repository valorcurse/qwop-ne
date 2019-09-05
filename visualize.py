import networkx as nx
import rpyc
import json
from plotly.utils import PlotlyJSONEncoder

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from gephistreamer import graph
from gephistreamer import streamer

import networkx as nx
from netwulf import visualize

import matplotlib.pyplot as plt

from neat.phenotypes import Phenotype
from neat.utils import Singleton

# class Visualize():

# 	def __init__(self) -> None:
# 		self.queue = rpyc.connect("localhost", 18811)

# 	def update(self, neuralNetwork: Phenotype):
# 		self.neuralNetwork = neuralNetwork

# 		edges = {"x": [], "y": []}
# 		for edge in [e for n in self.neuralNetwork.neurons for e in n.linksIn]:
# 			edges['x'].append(tuple([edge.fromNeuron.x, edge.toNeuron.x, None]))
# 			edges['y'].append(tuple([edge.fromNeuron.y, edge.toNeuron.y, None]))
		
# 		# print(edges)
		
# 		neurons = {"x": [], "y": []}
# 		for neuron in self.neuralNetwork.neurons:
# 			neurons['x'].append(neuron.x)
# 			neurons['y'].append(neuron.y)

# 		print("Visualize -> neurons: {}".format(len(neurons)))

# 		dataToSend = json.dumps([neurons, edges])
# 		self.queue.root.send_message(dataToSend)


class Visualize(metaclass=Singleton):

	def update(self, neuralNetwork: Phenotype):
		# G = nx.Graph()
		G = nx.DiGraph()
		fig, ax = plt.subplots()

		for neuron in neuralNetwork.neurons:
			G.add_node(neuron.ID, pos=(neuron.x, neuron.y))

		for edge in [e for n in neuralNetwork.neurons for e in n.linksIn]:
			G.add_edge(edge.fromNeuron.ID, edge.toNeuron.ID, edge_color=edge.weight)
		
		pos = nx.get_node_attributes(G,'pos')
		
		nx.draw_networkx_nodes(G, pos, ax=ax)
		nx.draw_networkx_edges(G, pos, ax=ax, edge_vmin=-1.0, edge_vmax=1.0)
		ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

		print(nx.adjacency_matrix(G).todense())

		plt.show()
		# visualize(G)

		

