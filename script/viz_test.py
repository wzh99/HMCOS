from graphviz import Digraph

graph = Digraph(name='test', node_attr={
                'shape': 'box', 'style': 'rounded', 'fontname': 'Segoe UI'})
graph.node('1', label='one')
graph.node('2', label='two')
graph.edge('1', '2')
graph.render(directory='out')
