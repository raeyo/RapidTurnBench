import unittest
import numpy as np
import yaml
from pathlib import Path
from rapidturn.isaac_metrics import evaluate
from rapidturn.submission import validate_action

class IsaacMetricsTests(unittest.TestCase):
 def setUp(self):
  self.c=yaml.safe_load((Path(__file__).parents[1]/'configs/benchmark/isaac_quickcheck.yaml').read_text())
 def log(self,completion=.5,stationary=False,body_turn=True):
  t=np.arange(0,6.001,.02);yaw=np.where(t>=3,np.pi/2,0)
  actual=np.where(t>=3+completion,np.pi/2,0)
  v=np.c_[np.cos(actual),np.sin(actual),np.zeros_like(t)]
  if stationary:v[t>=3]=0
  q=np.zeros((len(t),4));q[:,0]=np.cos((actual if body_turn else np.zeros_like(t))/2);q[:,3]=np.sin((actual if body_turn else np.zeros_like(t))/2)
  return dict(timestamp=t,root_linear_velocity=v,root_position=np.tile([0,0,.8],(len(t),1)),
              torso_quaternion=q,command_yaw=yaw,command_velocity_xy=np.c_[np.cos(yaw),np.sin(yaw)],standing_root_height=.8)
 def test_known_response_includes_causal_filter(self):
  r=evaluate(self.log(),self.c)
  self.assertTrue(r['turn_reached']);self.assertAlmostEqual(r['turn_time_s'],.64,places=2)
 def test_stationary_is_not_a_turn(self):self.assertFalse(evaluate(self.log(stationary=True),self.c)['turn_reached'])
 def test_sideways_translation_without_facing_is_not_a_turn(self):self.assertFalse(evaluate(self.log(body_turn=False),self.c)['turn_reached'])
 def test_incomplete_transient_not_aggregated(self):
  log=self.log();log={k:(v[:180] if hasattr(v,'shape') else v) for k,v in log.items()}
  self.assertFalse(evaluate(log,self.c)['transient_window_complete'])
 def test_nonfinite_and_shape_actions_rejected(self):
  for a in [np.zeros(28),np.full(29,np.nan)]:
   with self.assertRaises(ValueError):validate_action(a,np.ones(29))
 def test_torque_limit_is_shared(self):
  a,n=validate_action(np.full(29,2),np.ones(29));self.assertEqual(n,29);np.testing.assert_equal(a,np.ones(29))
if __name__=='__main__':unittest.main()
