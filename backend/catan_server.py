"""Local rules-only bridge. Language-model inference runs in the browser."""
import argparse
import json
import math
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

from catanatron import Game, Color, RandomPlayer
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS, NodeRef, ActionType
from catanatron.models.board import get_edges
from catanatron.models.tiles import Port
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import player_key

REVISION = 'ecf931181b9a65bb4116a2153fb78c16f1438e00'
RESOURCE_SHORT = {'WOOD':'wood','BRICK':'brick','SHEEP':'sheep','WHEAT':'wheat','ORE':'ore'}
REFS = [NodeRef.NORTH,NodeRef.NORTHEAST,NodeRef.SOUTHEAST,NodeRef.SOUTH,NodeRef.SOUTHWEST,NodeRef.NORTHWEST]


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return value


class CatanBridge:
    def __init__(self, seed=42, max_turns=160, auto_forced=True):
        self.seed=integer(seed,'seed',0,2**31-1)
        self.max_turns=integer(max_turns,'maxTurns',1,1000)
        if type(auto_forced) is not bool:
            raise ValueError('autoForced must be boolean')
        self.auto_forced=auto_forced
        self.game=Game([RandomPlayer(Color.RED),*[WeightedRandomPlayer(c) for c in [Color.BLUE,Color.WHITE,Color.ORANGE]]],seed=seed)
        self.game_id=uuid.uuid4().hex
        self.revision=0
        self.stats={'engineActions':0,'llmDecisions':0,'forcedActions':0,'opponentActions':0,'redActionCounts':{}}
        self.events=[]
        self.history=[]
        self._advance()

    @property
    def version(self):
        return f'{self.game_id}:{self.revision}'

    def finished(self):
        return self.game.winning_color() is not None or self.game.state.num_turns >= self.max_turns or self.stats['engineActions'] >= 5000

    def _execute(self, action, source):
        label=self.action_label(action)
        if action.color!=Color.RED and action.action_type==ActionType.DISCARD_RESOURCE:
            label='discard 1 resource'
        record=self.game.execute(action)
        self.stats['engineActions']+=1
        if source=='forced':self.stats['forcedActions']+=1
        if source=='opponent':self.stats['opponentActions']+=1
        if action.color==Color.RED:
            counts=self.stats['redActionCounts']; name=action.action_type.value
            counts[name]=counts.get(name,0)+1
        # Never expose stolen resources or opponents' purchased development cards.
        event=f'{action.color.value}: {label}'
        if action.action_type==ActionType.ROLL:
            event+=f' → {sum(record.result)}'
        self.events.append(event)
        self.events=self.events[-12:]
        self.history.append({'actor':action.color.value,'type':action.action_type.value,'label':label,'source':source,'turn':self.game.state.num_turns})
        self.revision+=1

    def _advance(self):
        while not self.finished():
            actions=self.game.playable_actions
            if not actions:raise RuntimeError('Engine returned no legal action before game end')
            if self.game.state.current_color()!=Color.RED:
                self._execute(self.game.state.current_player().decide(self.game,actions),'opponent')
            elif self.auto_forced and len(actions)==1:
                self._execute(actions[0],'forced')
            else:break

    def choose(self,action_id,version):
        if version!=self.version:raise ValueError('Stale state version; fetch the current state')
        if self.finished():raise ValueError('Game has ended')
        if self.game.state.current_color()!=Color.RED:raise ValueError('Not RED decision')
        integer(action_id,'actionId',0,len(self.game.playable_actions)-1)
        action=self.game.playable_actions[action_id]
        self.stats['llmDecisions']+=1
        self._execute(action,'model')
        self._advance()
        return self.snapshot()

    def node_yield(self,node):
        return '/'.join(f'{RESOURCE_SHORT[t.resource]}{t.number}' if t.resource else 'desert' for t in self.game.state.board.map.land_tiles.values() if node in t.nodes.values())

    def action_label(self,action):
        kind,value=action.action_type,action.value
        if kind in [ActionType.BUILD_SETTLEMENT,ActionType.BUILD_CITY]:
            return f'{kind.value.lower()} N{value} ({self.node_yield(value)})'
        if kind==ActionType.BUILD_ROAD:
            return f'build_road N{value[0]}–N{value[1]}'
        if kind==ActionType.MOVE_ROBBER:
            tile=self.game.state.board.map.land_tiles[value[0]]
            return f'move_robber T{tile.id} {tile.resource or "desert"}{tile.number or ""}; steal {value[1].value if value[1] else "nobody"}'
        if kind==ActionType.MARITIME_TRADE:
            giving=[r for r in value[:-1] if r]
            return f'trade {len(giving)} {giving[0].lower()} for 1 {value[-1].lower()}'
        if kind==ActionType.DISCARD_RESOURCE:return f'discard 1 {value.lower()}'
        if kind==ActionType.PLAY_YEAR_OF_PLENTY:return 'year_of_plenty '+','.join(value)
        if kind==ActionType.PLAY_MONOPOLY:return 'monopoly '+value
        if value is not None:return kind.value.lower()+' '+str(value)
        return kind.value.lower()

    def board_data(self):
        board=self.game.state.board
        nodes={};tiles=[]
        for coordinate,tile in board.map.land_tiles.items():
            x=math.sqrt(3)*(coordinate[0]+coordinate[2]/2);y=1.5*coordinate[2]
            ids=[]
            for i,ref in enumerate(REFS):
                node_id=tile.nodes[ref];ids.append(node_id)
                angle=math.radians(-90+i*60)
                building=board.buildings.get(node_id)
                nodes[node_id]={'id':node_id,'x':round(x+math.cos(angle),5),'y':round(y+math.sin(angle),5),'owner':building[0].value if building else None,'building':building[1] if building else None}
            tiles.append({'id':tile.id,'x':round(x,5),'y':round(y,5),'resource':tile.resource or 'DESERT','number':tile.number,'nodes':ids,'robber':coordinate==board.robber_coordinate})
        edges=[{'a':a,'b':b,'owner':board.roads[(a,b)].value if (a,b) in board.roads else None} for a,b in get_edges(board.map.land_nodes)]
        ports=[]
        for tile in board.map.tiles.values():
            if isinstance(tile,Port):
                ids=sorted(tile.nodes[ref] for ref in PORT_DIRECTION_TO_NODEREFS[tile.direction])
                ports.append({'resource':tile.resource or 'ANY','nodes':ids})
        return {'tiles':tiles,'nodes':sorted(nodes.values(),key=lambda n:n['id']),'edges':edges,'roads':[e for e in edges if e['owner']], 'robber':board.map.land_tiles[board.robber_coordinate].id,'ports':ports}

    def snapshot(self):
        state=self.game.state; p=state.player_state; key=player_key(state,Color.RED)
        hand={r:p[f'{key}_{r}_IN_HAND'] for r in RESOURCES}
        dev={r:p[f'{key}_{r}_IN_HAND'] for r in DEVELOPMENT_CARDS}
        players=[]
        for color in state.colors:
            k=player_key(state,color)
            players.append({'color':color.value,'points':p[f'{k}_ACTUAL_VICTORY_POINTS'] if color==Color.RED else p[f'{k}_VICTORY_POINTS'],'roads':15-p[f'{k}_ROADS_AVAILABLE'],'settlements':5-p[f'{k}_SETTLEMENTS_AVAILABLE'],'cities':4-p[f'{k}_CITIES_AVAILABLE'],'resourceCount':sum(p[f'{k}_{r}_IN_HAND'] for r in RESOURCES),'developmentCount':sum(p[f'{k}_{r}_IN_HAND'] for r in DEVELOPMENT_CARDS),'longestRoad':p[f'{k}_LONGEST_ROAD_LENGTH'],'playedKnights':p[f'{k}_PLAYED_KNIGHT']})
        done=self.finished();winner=self.game.winning_color()
        actions=[] if done else [{'id':i,'label':self.action_label(a),'type':a.action_type.value} for i,a in enumerate(self.game.playable_actions)]
        board=self.board_data()
        result={'observationVersion':'catan-compact-v2','gameId':self.game_id,'version':self.version,'seed':self.seed,'turn':state.num_turns,'phase':state.current_prompt.value,'maxTurns':self.max_turns,'done':done,'winner':winner.value if winner else None,'reason':'winner' if winner else 'turn limit' if state.num_turns>=self.max_turns else 'action limit' if self.stats['engineActions']>=5000 else None,'actions':actions,'players':players,'hand':hand,'development':dev,'board':board,'stats':dict(self.stats),'events':list(self.events),'engine':{'name':'Catanatron','version':'3.3.0','commit':REVISION,'opponents':'3 WeightedRandomPlayer bots; seeded game RNG','vpsToWin':10,'playerTradeOffers':False,'autoForced':self.auto_forced}}
        result['prompt']=self.observation_prompt(result)
        return result

    def observation_prompt(self,view):
        # Public information plus RED's own hand only. Every legal move is shown.
        robber=next(tile for tile in view['board']['tiles'] if tile['robber'])
        robber_nodes=','.join('N'+str(node) for node in robber['nodes'])
        lines=['You are RED playing standard 4-player Catan. Reach 10 victory points. Choose ONE legal move by its actionId. No explanations.',
               'Costs: road=wood+brick; settlement=wood+brick+sheep+wheat; city=2wheat+3ore; development=sheep+wheat+ore. Settlement=1 point; city=2. Longest road and largest army=2 each.',
               'Development effects: knight moves robber and steals 1 random resource; year_of_plenty takes 2 chosen bank resources; monopoly takes all of a chosen resource from opponents; road_building grants up to 2 free roads; victory_point adds 1 private point.',
               'Ports require your settlement/city on a listed node. ANY port trades 3 identical resources for 1 other resource; a resource port trades 2 of its named resource for 1 other resource. Without a matching port, bank trade is 4:1.',
               'Dice likelihood: 6/8 most common, then 5/9, 4/10, 3/11, 2/12. Robber blocks its tile. Nodes list their adjacent resource+dice tiles; roads join node IDs. You cannot build on occupied or adjacent-to-building nodes. Legal moves already enforce rules.',
               f'Turn {view["turn"]}/{view["maxTurns"]}; phase {view["phase"]}.',
               f'Robber blocks T{robber["id"]} {robber["resource"]} dice={robber["number"] if robber["number"] is not None else "none"} nodes={robber_nodes}.',
               'Your resources: '+json.dumps(view['hand'],separators=(',',':')),
               'Your development cards: '+json.dumps(view['development'],separators=(',',':')),
               'Players: '+'; '.join(f'{p["color"]}={p["points"]}VP,{p["resourceCount"]}resources,{p["developmentCount"]}dev,{p["playedKnights"]}knights,roadLength{p["longestRoad"]}' for p in view['players'])]
        adjacency={n['id']:[] for n in view['board']['nodes']}
        for edge in view['board']['edges']:
            adjacency[edge['a']].append(edge['b']);adjacency[edge['b']].append(edge['a'])
        lines.append('BOARD N=node, next=connected neighbors:')
        for node in view['board']['nodes']:
            owner=f'{node["owner"]}:{node["building"]}' if node['owner'] else 'empty'
            lines.append(f'N{node["id"]} {self.node_yield(node["id"])} next={",".join(map(str,sorted(adjacency[node["id"]])))} {owner}')
        lines.append('Roads: '+';'.join(f'{e["a"]}-{e["b"]}:{e["owner"]}' for e in view['board']['roads']))
        lines.append('Ports: '+';'.join(f'{p["resource"]}@{",".join(map(str,p["nodes"]))}' for p in view['board']['ports']))
        lines.append('Recent actions: '+'; '.join(view['events'][-4:]))
        lines.append('LEGAL MOVES (IDs are valid for this decision only):')
        lines.extend(f'{a["id"]}: {a["label"]}' for a in view['actions'])
        lines.append('Return only {"actionId": NUMBER} selecting a listed move.')
        return '\n'.join(lines)


class Handler(BaseHTTPRequestHandler):
    bridge=None
    def log_message(self,*args):pass
    def allowed(self):
        port=self.server.server_port
        if self.headers.get('Host') not in [f'127.0.0.1:{port}',f'localhost:{port}']:return False
        origin=self.headers.get('Origin')
        return not origin or origin in ['http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8001','http://localhost:8001']
    def respond(self,code,value):
        body=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
        if self.headers.get('Origin') and self.allowed():self.send_header('Access-Control-Allow-Origin',self.headers['Origin']);self.send_header('Vary','Origin')
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_OPTIONS(self):
        if not self.allowed():return self.respond(403,{'error':'Local origin required'})
        self.send_response(204);self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin','http://127.0.0.1:5173'));self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS');self.send_header('Access-Control-Allow-Headers','Content-Type');self.end_headers()
    def do_GET(self):
        if not self.allowed():return self.respond(403,{'error':'Local origin required'})
        if urlsplit(self.path).path=='/api/catan/state':
            return self.respond(200,Handler.bridge.snapshot()) if Handler.bridge else self.respond(404,{'error':'Create a game first'})
        if urlsplit(self.path).path=='/api/catan/health':return self.respond(200,{'ok':True,'engine':'Catanatron 3.3.0','commit':REVISION})
        self.respond(404,{'error':'Unknown endpoint'})
    def do_POST(self):
        if not self.allowed():return self.respond(403,{'error':'Local origin required'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=16384:raise ValueError('Request body must be 1–16384 bytes')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('JSON object required')
            path=urlsplit(self.path).path
            if path=='/api/catan/new':
                Handler.bridge=CatanBridge(data.get('seed',42),data.get('maxTurns',160),data.get('autoForced',True));return self.respond(200,Handler.bridge.snapshot())
            if path=='/api/catan/action':
                if not Handler.bridge:raise ValueError('Create a game first')
                return self.respond(200,Handler.bridge.choose(data.get('actionId'),data.get('version')))
            return self.respond(404,{'error':'Unknown endpoint'})
        except (ValueError,TypeError,KeyError) as error:return self.respond(400,{'error':str(error)})
        except Exception as error:return self.respond(500,{'error':f'{type(error).__name__}: {error}'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8001);args=parser.parse_args()
    print(f'Catanatron bridge http://127.0.0.1:{args.port}',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
