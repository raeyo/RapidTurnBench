"""Submission contract example. Holds a posture; it is not a locomotion model."""
import numpy as np
# Per-joint PD gains: legs, waist, arms (Unitree order).
KP=np.array([100,100,100,150,40,40]*2+[150,40,40]+[40,40,40,40,20,20,20]*2,dtype=float)
KD=np.array([2,2,2,4,2,2]*2+[4,2,2]+[1,1,1,1,.5,.5,.5]*2,dtype=float)
class Policy:
    def load(self, directory, context):
        self.nominal=np.asarray(context['asset_profile']['nominal_joint_position'])
    def reset(self,seed): pass
    def act(self, observation, command):
        return self.control(observation)
    def control(self, observation):
        # Optional 200 Hz hook: recompute PD torque every physics step.
        return KP*(self.nominal-observation['joint_position'])-KD*observation['joint_velocity']
    def close(self): pass
