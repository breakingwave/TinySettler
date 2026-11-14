"""Information-safe two-player BASE adapter; no search or private opponent inputs."""
import functools
import hashlib
import random
import numpy as np
from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.models.enums import ActionPrompt, ActionType, Action, DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.board import get_edges
from catanatron.models.map import number_probability
from catanatron.features import create_sample, get_feature_ordering
from catanatron.state_functions import player_key
from catanatron.gym.envs.action_space import get_action_array, to_action_space, from_action_space
from catanatron.players.weighted_random import WEIGHTS_BY_ACTION_TYPE

COLORS = (Color.RED, Color.BLUE)
FEATURES = tuple(f for f in get_feature_ordering(2) if not f.startswith('BANK_'))
TABLE = get_action_array(COLORS, 'BASE')
INDEX = {a:i for i,a in enumerate(TABLE)}
MAPPING_HASH = hashlib.sha256(('actor-relative-v2:'+repr(TABLE)).encode()).hexdigest()
for i in range(len(TABLE)):
    assert to_action_space(from_action_space(i, COLORS[0], COLORS, 'BASE'), COLORS, 'BASE') == i

def relative_value(action_type, value, color):
    # In the fixed table RED denotes self and BLUE denotes the opponent.
    # Swapping colors is its own inverse for encoding and decoding.
    if action_type == ActionType.MOVE_ROBBER and color == Color.BLUE:
        coordinate, victim = value
        if victim is not None:
            victim = Color.RED if victim == Color.BLUE else Color.BLUE
        return coordinate, victim
    return value

def action_index(action, color):
    return INDEX[(action.action_type, relative_value(action.action_type, action.value, color))]

def decode(index, color):
    kind, value = TABLE[index]
    return Action(color, kind, relative_value(kind,value,color))

@functools.lru_cache(maxsize=8)
def topology(m):
    x = []
    # Explicit fixed topology; sorted node-to-tile incidence and edge endpoints.
    for node in sorted(m.land_nodes):
        x.extend(float(tile in m.adjacent_tiles[node]) for _,tile in sorted(m.tiles_by_id.items()))
    for edge in sorted(get_edges(m.land_nodes)):
        x.extend(v / 53 for v in edge)
    return tuple(x)

def observe(game, color):
    s = game.state
    sample = create_sample(game, color)
    x = [float(sample[f]) for f in FEATURES]
    m = s.board.map
    for _, tile in sorted(m.tiles_by_id.items()):
        x.extend(float(tile.number == n) for n in range(13))
    x.extend(topology(m))
    x.extend(float(s.current_prompt == p) for p in ActionPrompt)
    x.extend([s.current_color() == color, s.colors[s.current_turn_index] == color,
              s.is_initial_build_phase, s.is_discarding, s.is_moving_knight,
              s.is_road_building, s.free_roads_available, s.is_resolving_trade,
              s.num_turns / 1000, 10, 7])
    own = s.colors.index(color)
    x.extend(s.discard_counts[(own+i)%2] for i in range(2))
    key = player_key(s, color)
    x.extend(float(s.player_state[key+'_'+c+'_OWNED_AT_START']) for c in DEVELOPMENT_CARDS if c != 'VICTORY_POINT')
    return np.asarray(x, dtype=np.float32)

def opponent_action(game, kind, rng):
    actions = game.playable_actions
    if kind == 'random': return rng.choice(actions)
    if kind == 'weighted':
        return rng.choices(actions, weights=[WEIGHTS_BY_ACTION_TYPE.get(a.action_type,1) for a in actions])[0]
    # Audited heuristic uses only action descriptors and public tile production.
    priority = {'BUILD_CITY': 100, 'BUILD_SETTLEMENT': 90, 'BUY_DEVELOPMENT_CARD': 40,
                'BUILD_ROAD': 20, 'END_TURN': 0, 'MARITIME_TRADE': -1}
    def score(a):
        val = priority.get(a.action_type.value, 10)
        if a.action_type.value in ('BUILD_CITY','BUILD_SETTLEMENT'):
            val += sum(number_probability(t.number) for t in game.state.board.map.adjacent_tiles[a.value] if t.resource)
        return val
    best = max(map(score, actions))
    return rng.choice([a for a in actions if score(a) == best])

class Env:
    def __init__(self, seed=0, seat=0, opponent=None, max_actions=20000, max_turns=1000):
        self.rng = random.Random(seed+981723)
        self.opponent = opponent or self.rng.choice(['random','weighted','heuristic'])
        self.game = Game([Player(c) for c in COLORS], seed=seed, discard_limit=7, vps_to_win=10)
        self.color = self.game.state.colors[seat]
        self.actions = 0
        self.max_actions, self.max_turns = max_actions, max_turns
        self.advance()
    def execute(self,a):
        assert a in self.game.playable_actions
        self.game.execute(a); self.actions += 1
    def advance(self):
        # Reach a learner decision even across an administrative boundary, so
        # bootstrapping uses the correct player's continuation value.
        count = 0
        while self.game.winning_color() is None and self.game.state.current_color() != self.color:
            self.execute(opponent_action(self.game,self.opponent,self.rng)); count += 1
            if count > 20000: raise RuntimeError('opponent advancement guard')
    def obs(self): return observe(self.game,self.color)
    def mask(self):
        mask = np.zeros(len(TABLE),dtype=bool)
        if self.game.winning_color() is None:
            assert self.game.state.current_color() == self.color
            for a in self.game.playable_actions:
                mask[action_index(a,self.color)] = True
        return mask
    def step(self,a):
        assert self.mask()[a], 'illegal learner action'
        before = self.actions
        self.execute(decode(int(a),self.color))
        self.advance()
        winner = self.game.winning_color()
        terminal = winner is not None
        truncated = not terminal and (self.actions >= self.max_actions or self.game.state.num_turns >= self.max_turns)
        reward = 0 if not terminal else (1 if winner == self.color else -1)
        return self.obs(), reward, terminal, truncated, self.actions-before
    def points(self):
        s = self.game.state
        return s.player_state[player_key(s,self.color)+'_ACTUAL_VICTORY_POINTS']
