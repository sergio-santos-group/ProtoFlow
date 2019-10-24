from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum
import json
import os
# from jinja2 import Template

# import templates

DIR_NAME = os.path.abspath(os.path.dirname(__file__))

jenv = Environment(
    loader=FileSystemLoader(os.path.join(DIR_NAME, 'templates')),
    # trim_blocks=True
)

def render(context, **kwargs):
    template = context['template']
    return template.render(**context, **kwargs)

jenv.filters['render'] = render


class OutputType(Enum):
    STREAM = 1
    SINGLE = 2


class NodeType(Enum):
    MODULE = 1
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
        self.connections = kwargs.get('connections', [])
        self.static = kwargs.get('static', False)
        
        self.is_input = is_input
        self.parent = parent
        self.others = []

    def __str__(self):
        return f'Port<{self.id}>'

    def __repr__(self):
        return str(self)

    def is_valid(self):
        nc = len(self.connections)
        return (self.cardinality['min'] <= nc <= self.cardinality['max'])

    # def get_channel_name(self):
    #     return Port.format_channel(self.id)
        
    # def get_inchannel_name(self):
    #     return Port.format_channel(self.id)
        
    # def get_outchannel_names(self):
    #     return [Port.format_channel(c.split('-')[1]) for c in self.connections]
    
    @staticmethod
    def format_channel(port_id):
        '''
        Generate an unique channel name for the given port ID

        >>> Port.format_channel("NODEID:inport")
        channel_NODEID_inport
        '''
        if port_id not in Port.CHANNELS:
            nodeid, varname = port_id.split(':')
            Port.CHANNELS[port_id] = f'channel_{nodeid}_{varname}'
        return Port.CHANNELS[port_id]
    
    @staticmethod
    def format_varname(port_id):
        '''
        Generate an unique variable name for the given port ID

        >>> Port.format_varname("NODEID:inport")
        var_NODEID_inport
        '''
        if port_id not in Port.VARIABLES:
            nodeid, varname = port_id.split(':')
            Port.VARIABLES[port_id] = f'var_{nodeid}_{varname}'
        return Port.VARIABLES[port_id]


    def get_varname(self):
        '''
         NODE_J         NODE_I
           ───┐       ┌────────┐
           out┾ ╌╌╌╌╌ ┽in   out┾
              │       │ (self) │
           ───┘       └────────┘
        
        Return the variable name associated with this port's variable
        
        If this is an output port, then return variable name corresponding
        to this port's ID; otherwise, identify the port connected to this port,
        and return the variable associated with that port.

        >>> # self.is_input = False
        >>> self.get_varname()
        var_NODE_I_out

        >>> # self.is_input = True
        >>> self.get_varname()
        var_NODE_J_out
        '''
        if not self.is_input:
            return Port.format_varname(self.id)
    
        # if len(self.connections) == 0:
        #     raise Exception('Cannot get variable name for unconnected ports')
        # elif len(self.connections) > 1:
        #     raise Exception('Cannot get variable name for ports with multiple connections')
        
        other = self.others[0]
        return Port.format_varname(other.id)


   

    #-------------
    def get_name(self):
        if self.is_input:
            return 'in_'+self.name
        return 'out_'+self.name

    def get_channel_name(self):
        return self.get_name() + '_channel'

    def get_channel_type(self):
        return self.type



class Node(object):

    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self.config = kwargs.get('config', [])
        self.parent = kwargs.pop('parent')
        self.iports = []
        self.oports = []
        self.raw = kwargs

        self.produce_type = OutputType.SINGLE
        self.type_ = getattr(NodeType, self.type.upper())
        #CODE[self.name]['produce_type']

    def get_config(self):
        return {d['name']:d['value'] for d in self.config}

    def is_valid(self):
        for port in self.iports:
            if not port.is_valid():
                return False
        return True
        
    def get_instance_name(self):
        return f'{self.name.replace(" ","_")}_{self.id}'

    def get_instance_type(self):
        return self.name.replace(' ','_')

    # def get_config_values(self):
    #     pairs = []
    #     for cfg in self.config:
    #         if cfg['type'] in ('number','range','checkbox'):
    #             pairs.append(f'{cfg["name"]} = {cfg["value"]}')
    #         else:
    #             pairs.append(f'{cfg["name"]} = "{cfg["value"]}"')
    #     return pairs
    #     # return ((k['name'],k['value']) for k in self.config)
    #     # return (k['name']+' = '+str(k['value']) for k in self.config)

    # def _get_inchannels(self):
    #     channels = []
    #     for port in self.iports:
    #         if port.category == 'channel' and len(port.connections) > 0:
    #             ch = port.get_inchannel_name()
    #             channels.append((port, ch))
    #     return channels
    
    # def _get_outchannels(self):
    #     channels = []
    #     for port in self.oports:
    #         if port.category == 'channel' and len(port.connections) > 0:
    #             for ch in port.get_outchannel_names():
    #                 channels.append((port, ch))
    #     return channels
    
    # def _get_components(self):
    #     components = []
    #     for port in filter(lambda p: p.category=='component' and len(p.connections)==1, self.iports):
    #         connection = self.parent.connection_from_id(port.connections[0])
    #         component = self.parent.node_from_id(connection['source']['nodeId'])
    #         components.append((port, component))
    #     return components

    # def _get_callbacks(self):
    #     callbacks = []
    #     for port in filter(lambda p: p.category=='callback' and len(p.connections)>0, self.iports):
    #         for connectionID in port.connections:
    #             connection = self.parent.connection_from_id(connectionID)
    #             component = self.parent.node_from_id(connection['source']['nodeId'])
    #             callbacks.append(component)
    #     return callbacks

    def __str__(self):
        return f'Node<{self.id}:{self.name}>'
        
    def __repr__(self):
        return str(self)

    def add_inport(self, port):
        self.iports.append(port)
    
    def add_outport(self, port):
        self.oports.append(port)

    def _filter_ports(self, container, categories):
        return (p for p in container if not categories or p.category in categories)

    def get_inports(self, *categories):
        return self._filter_ports(self.iports, categories)
        # return (p for p in self.iports if type is None or p.category==type)

    def get_outports(self, *categories):
        return self._filter_ports(self.oports, categories)
        # return (p for p in self.oports if type is None or p.category==type)

    def get_callbacks_scope(self):
        scope = {}
        for port in self.get_inports('callback'):
            callbacks = []
            for source in port.others:
                node = source.parent
                item_scope = node.get_config()
                item_scope['template'] = node.get_template()
                callbacks.append(item_scope)
            scope[port.name] = callbacks
        return scope

    def get_components_scope(self):
        scope = {}
        for port in self.get_inports('component'):
            source = port.others[0]
            node = source.parent
            item_scope = node.get_config()
            item_scope[source.get_name()] = source.get_varname()
            # item_scope.update({p.get_name():p.get_varname() for p in node.oports if p.is_valid()})
            item_scope[port.get_name()] = port.get_varname()
            item_scope['template'] = node.get_template()
            scope[port.name] = item_scope
        return scope
    
    def get_channels_scope(self):
        scope = {p.get_name():p.get_varname() for p in self.get_inports('channel')}
        scope.update({p.get_name():p.get_varname() for p in self.oports})
        return scope

    def get_channels_scope2(self):
        scope = {p.get_name():p.get_name() for p in self.get_inports('channel')}
        scope.update({p.get_name():p.get_name() for p in self.oports})
        return scope

    def get_template(self):
        return jenv.get_template(self.name + '.j2')

    def identify_runtype(self):
        if self.type_ != NodeType.MODULE:
            self.produce_type = None
        elif self.produce_type == OutputType.SINGLE:
            ports = list(self.get_inports('channel'))
            if len(ports) > 0 and any(len(p.others)>1 for p in ports):
                self.produce_type = OutputType.STREAM
        return self.produce_type
    
    def get_runtype(self):
        return self.produce_type

    # def count_in_channels(self):
    #     return sum(1 for p in self.iports if p.category=='channel')
    # def julia(self):
    #     t = Node.TEMPLATES.get(self.type, None)
    #     return None if t is None else t.render(node=self)
    
    # def finalize(self, graph):
    #     self.callbacks = self._get_callbacks()
    #     self.components = self._get_components()
    #     # self.inchannels = self._get_inchannels()
    #     self.outchannels = self._get_outchannels()
    #     # print(self.callbacks)
        
    #     inchannels = self._get_inchannels()
    #     self.inchannels = [(p,ch) for p,ch in inchannels if not p.static]
    #     self.staticchannels = [(p,ch) for p,ch in inchannels if p.static]



class ExecutionGraph(object):
    def __init__(self):
        self.nodes = {}
        self.ports = {}
        self.connections = {}
        self.g = nx.Graph()
        self.dg = nx.DiGraph()

    def new_port(self, is_input, parent, **kwargs):
        port = Port(is_input, parent, **kwargs)
        self.ports[port.id] = port
        return port

    def new_node(self, **kwargs):
        node = Node(parent=self, **kwargs)
        self.nodes[node.id] = node
        # self.g.add_node(node.id)
        if node.type_ == NodeType.MODULE:
            self.dg.add_node(node.id, node=node)
        return node
    
    def new_connection(self, **kwargs):
        connection = kwargs
        # self.connections[connection['id']] = connection
        source_id = connection.get('source')['nodeId']
        target_id = connection.get('target')['nodeId']
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        if source.type_ == NodeType.MODULE and target.type_ == NodeType.MODULE:
            # self.g.add_edge(source['nodeId'], target['nodeId'])
            self.dg.add_edge(source_id, target_id)
        # self.dg.add_edge(source_id, target_id)
        return connection

    def node_from_id(self, id):
        return self.nodes.get(id)
    
    def connection_from_id(self, id):
        return self.connections.get(id)
    
    def finalize(self):
        # for node in self.nodes.values():
        #     node.finalize(self)
        return self
    
    def validate(self):
        # print('Ntriggers =', len(filter(lambda n: n.type=='trigger', self.nodes.values())))
        print(nx.number_connected_components(self.g))
        # nx.dag.descendants(g1,'zflmdo79x')
        # nx.number_connected_components()
    #     errors = []
    #     self.g
    
    # def identify_streaming_nodes(self):
    #     streaming = set()
    #     for node in self.nodes.values():
    #         # skip non-module nodes: those will be inlined!
    #         if node.type != 'module':
    #             continue

    #         inports = list(node.get_inports('channel'))
    #         if node.produce_type != OutputType.STREAM and \
    #             len(inports) > 0 and \
    #             any(len(p.others)>1 for p in inports):
    #             node.produce_type = OutputType.STREAM
            
    #         if node.produce_type == OutputType.STREAM:
    #             streaming.add(node)

    #     for node in streaming:
    #         for source,target in nx.edge_bfs(graph.dg, source=node.id):
    #             target = self.nodes[target]
    #             if target.type == 'module':
    #                 target.produce_type = OutputType.STREAM

    #     for node in self.nodes.values():
    #         print(node.id, node.produce_type)
    #     print(streaming)
    #     return streaming

    # def julia2(self):
    #     self.identify_streaming_nodes()

    def julia(self):
        #self.julia2()
        #print("--------- END JULIA2------------")

        # *** IDENTIFY STREAMING NODES ***
        # find all streaming nodes: these are all nodes
        # explicitly marked OutputType.STREAM or having
        # at least one input channel with more than one connection
        multisource = set()
        for node in self.nodes.values():
            runtype = node.identify_runtype()
            if runtype == OutputType.STREAM:
                multisource.add(node)

        # all downstream nodes from streaming nodes are
        # marked as streaming nodes as well, because they
        # will be consuming streams!
        #
        # TODO: identify nodes that can be packed inside
        #  the streaming node.
        #
        for node in multisource:
            for source,target in nx.edge_bfs(graph.dg, source=node.id):
                self.nodes[target].produce_type = OutputType.STREAM
        
        # *** IDENTIFY ENDPOINTS ***
        # identify all nodes predecessing each multisource node;
        # these nodes should output their results to channels.
        endpoints = set()
        for node in multisource:
            for predecessor in self.dg.predecessors(node.id):
                other = self.nodes[predecessor]
                if other.get_runtype() == OutputType.SINGLE:
                    endpoints.add(predecessor)
        print(endpoints)

        # downstream nodes without output connections are also endpoints
        for node in self.nodes.values():
            if node.get_runtype() == OutputType.SINGLE and self.dg.out_degree(node.id) == 0:
                endpoints.add(node.id)
        print(endpoints)

        # from the set of all identified potential endpoint nodes,
        # some may be reacheable from other endpoint nodes. Hence,
        # only those not reacheable from other endpoints should be retained.
        uniq_endpoints = []
        for node in endpoints:
            if sum(nx.has_path(self.dg, node, other) for other in endpoints) == 1:
                uniq_endpoints.append(node)
        endpoints = uniq_endpoints
        

        # write code for each upstream path starting
        # at each endpoint node
        for endpoint in endpoints:
            path = [endpoint]
            for edge in nx.edge_dfs(self.dg, source=endpoint, orientation='reverse'):
                path.insert(0, edge[0])
            
            print('# ===================================================')
            print('# ', ' → '.join(path))
            print('# ===================================================')
            for nid in path:
                node = self.nodes[nid]
                
                # scope.update({p.get_name():p.get_varname() for p in node.get_inports('channel') if p.is_valid()})
                # scope.update({p.get_name():p.get_varname() for p in node.oports if p.is_valid()})
                
                scope = node.get_config()
                scope.update(node.get_channels_scope())
                scope.update(node.get_callbacks_scope())
                scope.update(node.get_components_scope())

                # callbacks_scope = node.get_callbacks_scope()
                # scope.update(callbacks_scope)
                
                # components_scope = node.get_components_scope()
                # scope.update(components_scope)
                
                code = node.get_template()
                print(code.render(**scope))

            print('')
        
        # Generate taskable functions
        tasks = []
        for node in self.nodes.values():
            if node.get_runtype() == OutputType.SINGLE or node.type_ != NodeType.MODULE:
                continue
            
            scope = {}
            inchannels = []
            outchannels = []

            for port in node.get_inports('channel'):
            
                # if this port has a single connection, and it comes
                # from a serial node, then it is a variable
                if len(port.connections) == 1:
                    source_node = port.others[0].parent
                    # source,target = port.connections[0].split('-')
                    # source_node = self.ports[source].parent
                    
                    if source_node.produce_type == OutputType.SINGLE:
                        scope[port.get_name()] = Port.format_varname(source)
                    else:
                        scope[port.get_name()] = port.get_name()
                        inchannels.append((port, Port.format_channel(port.id)))
                else:
                    scope[port.get_name()] = port.get_name()
                    inchannels.append((port, Port.format_channel(port.id)))
            # scope.update({p.get_name():p.get_varname() for p in node.iports if p.category!='callback' and p.is_valid()})
            
            for port in node.get_outports('channel'):
                # for target in port.others:
                #     outchannels.append((port, Port.format_channel(target.id)))
                for connection in port.connections:
                    source,target = connection.split('-')
                    outchannels.append((port, Port.format_channel(target)))
                scope[port.get_name()] = port.get_name()
            
            # code = CODE[node.name]['code']
            code = node.get_template()
            scope.update(node.get_components_scope())
            scope.update(node.get_callbacks_scope())
            scope['scope'] = scope
            scope.update(node.get_config())
            body = code.render(**scope)#, **node.get_config())
            
            print( '# ===================================================')
            print(f'# {node.name} [ID: {node.id}]')
            print( '# ===================================================')
            print(jenv.get_template('task.j2').render(node=node, inchannels=inchannels, outchannels=outchannels, body=body))
            print('')
            
            args = ','.join(c for p,c in inchannels+outchannels)
            task = f'task_{node.id} = @task {node.get_instance_name()}({args})'
            tasks.append((f'task_{node.id}', task))
            
        
        print('')
        for taskname,task in tasks:
            print(task)
        print('')
        for taskname,task in tasks:
            print(f'schedule({taskname})')
        

        # dump multisource nodes
        for node in multisource:

            for port in node.get_inports('channel'):
                
                # data fed through ports with a single connection are
                # also inlined if the connection source is single_sourced
                if len(port.connections) == 1:
                    continue
                
                for connection in port.connections:
                    source,target = connection.split('-')
                    source_node = self.ports[source].parent
                    if source_node.produce_type != OutputType.SINGLE:
                        continue
                    channel = Port.format_channel(port.id)
                    print("put!({channel}, {var})".format(channel=channel, var=Port.format_varname(source)))


        # # identify all nodes bearing channel ports with a single input source
        # single_source = set()
        # for node in self.nodes.values():
            
        #     if all(len(p.connections)==1 for p in node.iports if p.category=='channel'):
        #         single_source.add(node.id)
        
        # print(single_source)
        # for nodeid in single_source:
        #     node = self.nodes[nodeid]
        #     if len(node.iports) == 0:
        #         for descendant in self.dg.descendants(node.id):
        #             if descendant in single_source:
        #                 single_source.discard()
        #         print(node.id, "TRIGGER")


        
        # for key in self.dg.nodes:
        #     node = self.dg.nodes[key]['node']
            
        #     predecessors = list(self.dg.predecessors(key))
        #     if len(predecessors) == 0:
        #         print(key, node.name, node.type, 'IS TRIGGER')

            
        # nodes = filter(lambda n: n.type=='driver', self.nodes.values())
        # nodes = list(nodes)
        # for node in nodes:
        #     print(node.julia())
        
        # for node in nodes:
        #     for port,channel in node.inchannels:
        #         print(f'const {channel} = Channel{{{port.type}}}()')
        # print('')
        
        # for n,node in enumerate(nodes, start=1):
        #     channels = node.inchannels + node.outchannels
        #     s = ','.join(channel for port,channel in channels)
        #     print(f'task_{n} = @task {node.get_instance_name()}({s})')
        # print('')
        
        # for n,node in enumerate(nodes, start=1):
        #     print(f'schedule(task_{n})')
        # print('')

        # for node in filter(lambda n: n.type=='trigger', self.nodes.values()):
        #     print(node.julia())

        # print('yield()')
    
    def julia1(self):
        for nid in self.dg.nodes:
            node = self.nodes[nid]
            node.identify_runtype()

            code = node.get_template()
            scope = node.get_config()
            scope.update(node.get_channels_scope2())
            scope.update(node.get_callbacks_scope())
            scope.update(node.get_components_scope())

            body = code.render(**scope)
            inchannels = []
            for port in node.get_inports('channel'):
                inchannels.append((port, port.get_channel_name()))
            
            outchannels = []
            for port in node.get_outports('channel'):
                for target in port.others:
                    outchannels.append((port, target.get_channel_name()))
            
            print(
                jenv.get_template('task.j2').render(
                    forever=node.get_runtype()==OutputType.STREAM,
                    node=node, inchannels=inchannels, outchannels=outchannels, body=body)
            )
            

    @staticmethod
    def from_file(fname):
        with open(fname) as fin:
            data = json.load(fin)
        graph = ExecutionGraph()
        for node in data.get('nodes', []):
            n = graph.new_node(**node)
            ports = node.get('ports', {})
            for port in ports.get('input', []):
                p = graph.new_port(True, n, **port)
                n.add_inport(p)
                # n.add_inport(Port(True, n, **port))
            for port in ports.get('output', []):
                p = graph.new_port(False, n, **port)
                n.add_outport(p)
                # n.add_outport(Port(False, n, **port))

        for connection in data.get('connections', []):
            source,target = connection['id'].split('-')
            source_port = graph.ports[source]
            target_port = graph.ports[target]
            graph.new_connection(**connection)
            # graph.new_connection(source_port, target_port, **connection)
        
        for port in graph.ports.values():
            for connection in port.connections:
                source,target = connection.split('-')
                if port.is_input:
                    p = graph.ports[source]
                else:
                    p = graph.ports[target]
                port.others.append(p)
            
        return graph.finalize()


import sys

# graph = ExecutionGraph.from_file('../flows/f1.json')
graph = ExecutionGraph.from_file(sys.argv[1])
graph.julia1()
# graph.validate()

# print(g)
# pos = nx.spring_layout(graph.dg)
# nx.draw(graph.dg, pos=pos, with_labels=True)
# plt.show()
