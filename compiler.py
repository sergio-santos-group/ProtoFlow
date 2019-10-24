from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum
import json
import os


DIR_NAME = os.path.abspath(os.path.dirname(__file__))

jenv = Environment(
    loader=FileSystemLoader(os.path.join(DIR_NAME, 'templates'))
    # , trim_blocks=True
    # , line_statement_prefix = '#'
)

def render(context, **kwargs):
    template = context['template']
    return template.render(**context, **kwargs)

jenv.filters['render'] = render

# def call(context):
#     return (c() for c in context)
jenv.filters['call'] = lambda ctx: (c() for c in ctx)


class OutputType(Enum):
    STREAM = 1
    SINGLE = 2


class NodeType(Enum):
    MODULE = 1
    CALLBACK = 2
    COMPONENT = 3


class PortType(Enum):
    CHANNEL = 1
    CALLBACK = 2
    COMPONENT = 3


class Port(object):
    CHANNELS = {}
    VARIABLES = {}

    def __init__(self, is_input, parent, **kwargs):
        self.id = kwargs.get('id')
        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self.category = kwargs.get('category')
        self.cardinality = kwargs.get('cardinality')
        # self.connections = kwargs.get('connections', [])
        self.static = kwargs.get('static', False)
        
        self.category = getattr(PortType, kwargs.get('category').upper())

        self.is_input = is_input
        self.parent = parent
        self.others = []

    def __str__(self):
        return f'Port<{self.id}>'

    def __repr__(self):
        return str(self)

    def is_valid(self):
        nc = len(self.others)
        return (self.cardinality['min'] <= nc <= self.cardinality['max'])

    
    def get_variable_name(self):
        return self.get_name() + self.parent.id

    def get_name(self):
        if self.is_input:
            return 'in_'+self.name
        return 'out_'+self.name

    def get_type(self):
        return self.type
    
    def get_channel_name(self):
        return self.get_name() + '_channel'

    def get_channel_instance(self):
        return f'channel_{self.parent.id}_{self.name}'
    
    def validate(self, errors=None):
        if errors is None:
            errors = []
        if self.is_input and self.static and len(self.others) > 1:
            errors.append('static input ports can only have a single connection')
        return errors



class Node(object):

    @staticmethod
    def _filter_ports(container, categories):
        assert all(isinstance(x, PortType) for x in categories)
        return (p for p in container if not categories or p.category in categories)

    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.name = kwargs.get('name')
        self.config = kwargs.get('config', [])
        self.type = getattr(NodeType, kwargs.get('type').upper())
        
        self.produce_type = OutputType.SINGLE
        self.iports = []
        self.oports = []
        

    def __str__(self):
        return f'Node<{self.id}:{self.name}>'
    
    def __repr__(self):
        return str(self)

    def add_inport(self, port):
        self.iports.append(port)
    
    def add_outport(self, port):
        self.oports.append(port)
    
    def validate(self):
        errors = []
        for port in self.iports:
            port.validate(errors)
        return errors
    
    def get_inports(self, *categories):
        return Node._filter_ports(self.iports, categories)

    def get_outports(self, *categories):
        return Node._filter_ports(self.oports, categories)

    def get_instance_name(self):
        return f'{self.name.replace(" ","_")}_{self.id}'
    
    def get_scope(self):
        scope = {d['name']:d['value'] for d in self.config}
        scope['template'] = jenv.get_template(self.name + '.j2')
        scope['name'] = self.name
        scope['id'] = self.id
        return scope

    def get_callbacks_scope(self):
        scope = {}
        for port in self.get_inports(PortType.CALLBACK):
            scope[port.name] = [
                p.parent.get_scope()
                for p in port.others
            ]
        return scope

    def get_components_scope(self):
        scope = {}
        for port in self.get_inports(PortType.COMPONENT):
            source = port.others[0]
            item_scope = source.parent.get_scope()
            item_scope[source.get_name()] = source.get_variable_name()
            item_scope[  port.get_name()] = source.get_variable_name()
            scope[port.name] = item_scope
        return scope
    
    def get_channels_scope(self):
        scope = {
            p.get_name():p.get_variable_name()
            for p in self.get_inports(PortType.CHANNEL)
        }
        scope.update({
            p.get_name():p.get_variable_name()
            for p in self.oports
        })
        return scope

    # def get_template(self):
    #     return jenv.get_template(self.name + '.j2')

    def identify_runtype(self):
        if self.type != NodeType.MODULE:
            self.produce_type = None
        elif self.produce_type == OutputType.SINGLE:
            ports = list(self.get_inports(PortType.CHANNEL))
            if len(ports) > 0 and any(len(p.others)>1 for p in ports):
                self.produce_type = OutputType.STREAM
        return self.produce_type
    

    def get_runtype(self):
        return self.produce_type



class Compiler(object):
    
    def __init__(self):
        self.dg = nx.DiGraph()
        self.nodes = {}
        self.ports = {}

    def new_port(self, is_input, parent, **kwargs):
        port = Port(is_input, parent, **kwargs)
        self.ports[port.id] = port
        return port

    def new_node(self, **kwargs):
        node = Node(**kwargs)
        self.nodes[node.id] = node

        # only add MODULE nodes to the graph
        if node.type == NodeType.MODULE:
            self.dg.add_node(node.id)
        
        return node
    
    def new_connection(self, **kwargs):
        connection = kwargs
        
        source_id = connection.get('source')['nodeId']
        target_id = connection.get('target')['nodeId']
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        
        if source.type == NodeType.MODULE and target.type == NodeType.MODULE:
            self.dg.add_edge(source_id, target_id)
        
        source_port,target_port = connection['id'].split('-')
        self.ports[source_port].others.append( self.ports[target_port])
        self.ports[target_port].others.append( self.ports[source_port])
        return connection

    
    def validate(self):
        errors = []
        # count number of subgraphs
        # if nx.number_connected_components(graph.dg.to_undirected()) > 1:
        if nx.number_weakly_connected_components(self.dg) > 1:
            errors.append('no subgraphs are allowed')
        for node in self.nodes.values():
            errors.extend(node.validate())
            # node.validate()
        return errors
    

    def compile(self):

        nodes = [node
                    for node in self.nodes.values()
                    if node.type == NodeType.MODULE]
        nodes.sort(key=lambda node: int(node.id))
        
        # identify taskable functions (one per node)
        functions = []
        for node in nodes:
            node.identify_runtype()

            scope = node.get_scope()
            scope.update(node.get_channels_scope())
            scope.update(node.get_callbacks_scope())
            scope.update(node.get_components_scope())

            inputs  = [p for p in node.get_inports( PortType.CHANNEL)]
            outputs = [p for p in node.get_outports(PortType.CHANNEL) for _ in p.others]
            
            template = jenv.get_template('taskable.j2')
            functions.append(
                template.render(
                    forever=node.get_runtype()==OutputType.STREAM, block=scope,
                    node=node, inputs=inputs, outputs=outputs)
            )

        # identify channels
        channels = [p for node in nodes for p in node.get_inports(PortType.CHANNEL)]
        
        # identify tasks
        tasks = []
        for node in nodes:
            args = list(node.get_inports(PortType.CHANNEL))
            args.extend(x for p in node.get_outports(PortType.CHANNEL) for x in p.others)
            tasks.append((node, args))
        
        template = jenv.get_template('app.j2')
        return template.render(functions=functions, channels=channels, tasks=tasks)
        
            
    @staticmethod
    def from_file(fname):

        with open(fname) as fin:
            data = json.load(fin)
        
        graph = Compiler()
        for node in data.get('nodes', []):
            n = graph.new_node(**node)
            ports = node.get('ports', {})
            for port in ports.get('input', []):
                p = graph.new_port(True, n, **port)
                n.add_inport(p)
            for port in ports.get('output', []):
                p = graph.new_port(False, n, **port)
                n.add_outport(p)
        
        for connection in data.get('connections', []):
            graph.new_connection(**connection)
            
        return graph



if __name__ == '__main__':

    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        prog='compile',
        # description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'src', type=Path,
        help='flow file (json)')
    parser.add_argument(
        '-l', '--lib', type=Path,
        help='directory containing code templates')
    parser.add_argument(
        '--draw', action="store_true")
    
    args = parser.parse_args()
    if args.lib is not None:
        jenv.loader.searchpath.insert(0, str(args.lib))
    
    compiler = Compiler.from_file(args.src)
    errors = compiler.validate()
    if errors:
        for error in errors:
            print(error)
    else:
        jl = compiler.compile()
        print(jl)
    
    if args.draw:
        pos = nx.spring_layout(compiler.dg)
        nx.draw(compiler.dg, pos=pos, with_labels=True)
        plt.show()

