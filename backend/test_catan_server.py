import json
import random
import unittest
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from catanatron.state_functions import player_key
from backend.catan_server import CatanBridge

class BridgeTests(unittest.TestCase):
    def test_complete_game_legal_action_loop(self):
        bridge=CatanBridge(seed=42,max_turns=1000)
        rng=random.Random(5)
        while not bridge.finished():
            state=bridge.snapshot()
            self.assertEqual(bridge.game.state.current_color(),Color.RED)
            self.assertGreater(len(state['actions']),1)
            bridge.choose(rng.randrange(len(state['actions'])),state['version'])
        result=bridge.snapshot()
        self.assertTrue(result['done'])
        self.assertEqual(result['reason'],'winner')
        self.assertIsNotNone(result['winner'])
        self.assertFalse(result['actions'])
        self.assertGreater(result['stats']['engineActions'],100)
        self.assertGreater(result['stats']['forcedActions'],0)
        self.assertEqual(result['stats']['llmDecisions'],sum(h['source']=='model' for h in bridge.history))

    def test_stale_invalid_and_cross_game_action_do_not_mutate(self):
        bridge=CatanBridge(3)
        state=bridge.snapshot()
        for action in [-1,len(state['actions']),True,0.2,None]:
            with self.assertRaises(ValueError):bridge.choose(action,state['version'])
            self.assertEqual(state['version'],bridge.version)
        with self.assertRaises(ValueError):bridge.choose(0,CatanBridge(3).version)
        bridge.choose(0,state['version'])
        with self.assertRaises(ValueError):bridge.choose(0,state['version'])

    def test_opponent_private_hand_and_dev_cards_not_serialized(self):
        bridge=CatanBridge(42)
        state=bridge.game.state
        blue=player_key(state,Color.BLUE)
        state.player_state[f'{blue}_ORE_IN_HAND']=91
        state.player_state[f'{blue}_VICTORY_POINT_IN_HAND']=3
        state.player_state[f'{blue}_ACTUAL_VICTORY_POINTS']=9
        snapshot=bridge.snapshot()
        blue_view=next(p for p in snapshot['players'] if p['color']=='BLUE')
        self.assertEqual(blue_view['resourceCount'],91)
        self.assertEqual(blue_view['developmentCount'],3)
        self.assertNotEqual(blue_view['points'],9)
        self.assertNotIn('ORE',blue_view)
        self.assertNotIn('VICTORY_POINT',blue_view)
        self.assertNotIn('91ore',snapshot['prompt'])
        self.assertNotIn('"ORE":91',snapshot['prompt'])

    def test_robber_prompt_identifies_blocked_resource_and_nodes(self):
        bridge=CatanBridge(42)
        board=bridge.game.state.board
        coordinate,tile=next((coordinate,tile) for coordinate,tile in board.map.land_tiles.items() if tile.resource is not None)
        board.robber_coordinate=coordinate
        snapshot=bridge.snapshot()
        rendered=next(item for item in snapshot['board']['tiles'] if item['robber'])
        nodes=','.join('N'+str(node) for node in rendered['nodes'])
        expected=f'Robber blocks T{tile.id} {tile.resource} dice={tile.number} nodes={nodes}.'
        self.assertIn(expected,snapshot['prompt'])
        self.assertEqual(snapshot['observationVersion'],'catan-compact-v2')
        self.assertIn('ANY port trades 3 identical resources',snapshot['prompt'])
        self.assertIn('monopoly takes all of a chosen resource',snapshot['prompt'])

    def test_board_coordinates_topology_and_seed(self):
        a,b=CatanBridge(17).snapshot(),CatanBridge(17).snapshot()
        self.assertEqual(a['board'],b['board'])
        self.assertEqual(a['actions'],b['actions'])
        board=a['board'];nodes={n['id']:n for n in board['nodes']}
        self.assertEqual(len(nodes),54);self.assertEqual(len(board['tiles']),19);self.assertEqual(len(board['edges']),72)
        self.assertEqual(len(board['ports']),9)
        for port in board['ports']:self.assertEqual(len(port['nodes']),2)
        for tile in board['tiles']:
            for nid in tile['nodes']:
                n=nodes[nid];distance=((n['x']-tile['x'])**2+(n['y']-tile['y'])**2)**0.5
                self.assertAlmostEqual(distance,1,places=4)
        self.assertLess(len(a['prompt']),12000)

if __name__=='__main__':unittest.main()
