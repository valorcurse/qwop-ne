import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import asyncio
import rpyc
from rpyc.utils.server import ThreadedServer
import multiprocessing
from multiprocessing import Process, Queue
import multiprocessing.managers as managers

import json
from plotly.utils import PlotlyJSONEncoder

class QueueManager(managers.BaseManager):
    pass # Pass is really enough. Nothing needs to be done here.

def MyService(queue):
    class Service(rpyc.Service):

        # My service
        def exposed_send_message(self, message):
            queue.put(message)
            return queue.qsize()
    return Service


def startRpyC(queue):
    QueueService = MyService(queue)
    server = ThreadedServer(QueueService, port = 18811)
    server.listener.settimeout(1)
    server.start()

def plotlyfromjson(data):
    v = json.loads(data)

    return go.Figure(data=v['data'], layout=v['layout'])

queue = None

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions']=True

app.layout = html.Div([
    html.Div(id='my-output-interval'),
        dcc.Graph(id='live-update-graph', figure=go.Figure()),
    dcc.Interval(id='my-interval', interval=1*1000),
])

@app.callback(Output('live-update-graph', 'figure'), [Input('my-interval', 'n_intervals')],  [State('live-update-graph', 'figure')])
def update_graph_live(n, fig):
    if (queue.qsize() > 0):
        msg = queue.get()
        data = json.loads(msg)

        node_trace = go.Scatter(
            x=data[0]['x'],
            y=data[0]['y'],
            # text=[],
            mode='markers',
            # hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))

        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')

        for e in data[1]['x']:
            edge_trace['x'] += tuple(e)

        for e in data[1]['y']:
            edge_trace['y'] += tuple(e)


        return go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='<br>Network Graph of '+str(len(data[0]))+' neurons',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return fig

import datetime
@app.callback(Output('my-output-interval', 'children'), [Input('my-interval', 'n_intervals')])
def display_output(n):
    now = datetime.datetime.now()
    return '{} intervals have passed. It is {}:{}:{}'.format(
        n,
        now.hour,
        now.minute,
        now.second
)

if __name__ == '__main__':
    queueManager = QueueManager()
    queueManager.start()
    queue = multiprocessing.Manager().Queue()

    p = Process(target=startRpyC, args=(queue,))
    p.start()

    app.run_server(debug=False)