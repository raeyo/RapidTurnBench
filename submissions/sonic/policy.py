"""Frozen official SONIC deployment connected to Isaac state via DDS, without MuJoCo."""
from pathlib import Path
import os
import sys
import time
import numpy as np

class Policy:
    def load(self,directory,context):
        self.context=context
        root=Path(context['project_root'])
        sys.path.append(str(root/'.runtime/lib/python3.11/site-packages'))
        upstream=root/'third_party/GR00T-WholeBodyControl'
        sys.path.insert(0,str(upstream/'external_dependencies/unitree_sdk2_python'))
        sys.path.insert(0,str(upstream))
        os.environ['CYCLONEDDS_HOME']=str(root/'.native/cyclonedds-full')
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
        from rapidturn.sonic import SonicAdapter
        ChannelFactoryInitialize(61,'lo')
        self.bridge=UnitreeSdk2Bridge(dict(ROBOT_TYPE='g1_29dof',NUM_MOTORS=29,NUM_HAND_MOTORS=0,USE_SENSOR=False))
        self.adapter_class=SonicAdapter
        self.adapter=None
        self.ready=False
        self.clock=0
        self.last_command=None

    def reset(self,seed):
        if self.adapter: self.adapter.stop()
        self.bridge.reset()
        self.adapter=self.adapter_class(Path(self.context['episode_dir']),seed,61)
        root=Path(self.context['project_root'])
        old=os.environ.get('LD_LIBRARY_PATH','')
        os.environ['LD_LIBRARY_PATH']=':'.join(str(root/p) for p in ['.runtime/lib','.native/tensorrt/lib','.native/onnxruntime-linux-x64-1.22.0/lib','.native/cyclonedds-full/lib','.runtime/lib/python3.11/site-packages/nvidia/cuda_runtime/lib'])+':'+old
        try:self.adapter.start()
        finally:os.environ['LD_LIBRARY_PATH']=old
        self.ready=False
        self.last_command=None
        # DDS tick must be strictly increasing within each policy process.
        self.clock=0

    def _publish(self,observation):
        b=self.bridge
        wire=dict(time=(self.clock+.1)/1000,body_q=observation['joint_position'],
                  body_dq=observation['joint_velocity'],body_ddq=np.zeros(29),
                  body_tau_est=observation['previous_action'],
                  floating_base_pose=np.r_[observation['root_position'],observation['root_quaternion']],
                  floating_base_vel=np.r_[observation['root_linear_velocity'],observation['body_angular_velocity']],
                  floating_base_acc=np.zeros(6),secondary_imu_quat=observation['torso_quaternion'],
                  secondary_imu_vel=np.r_[np.zeros(3),observation['torso_angular_velocity']],
                  left_hand_q=[],right_hand_q=[])
        b.PublishLowState(wire)
        limit=time.monotonic()+(600 if not b.low_cmd_received else 15)
        resend=time.monotonic()+.02
        while True:
            with b.low_cmd_lock:
                if b.low_cmd_received and b.low_cmd.reserve[0]==self.clock: break
            if self.adapter.process.poll() is not None:
                raise RuntimeError('SONIC exited; see episode sonic.log')
            if time.monotonic()>limit: raise TimeoutError('SONIC DDS tick timed out')
            if time.monotonic()>=resend:
                b.PublishLowState(wire);resend=time.monotonic()+.02
            time.sleep(.0002)
        self.clock+=20

    def act(self,observation,command):
        if not self.ready:
            if self.adapter.log_contains('Init Done') and self.last_command is None:
                self.adapter.set_command(0,0,start=False,acknowledge=False)
                self.last_command=(0,0)
            if self.adapter.log_contains('Planner initialized successfully!'):
                self.adapter.set_command(0,0,start=True,acknowledge=False)
                self.ready=True
        if self.ready:
            key=(float(np.linalg.norm(command['target_velocity_xy'])),float(command['facing_yaw']))
            if key!=self.last_command:
                self.adapter.set_command(*key,acknowledge=False);self.last_command=key
        self._publish(observation)
        with self.bridge.low_cmd_lock:
            motors=self.bridge.low_cmd.motor_cmd[:29]
            kp=np.array([m.kp for m in motors]);kd=np.array([m.kd for m in motors])
            q=np.array([m.q for m in motors]);dq=np.array([m.dq for m in motors]);tau=np.array([m.tau for m in motors])
        self.motor_targets=(kp,kd,q,dq,tau)
        return self.control(observation)

    def control(self,observation):
        # Official deployment retains motor targets for the 200 Hz PD loop.
        kp,kd,q,dq,tau=self.motor_targets
        return kp*(q-observation['joint_position'])+kd*(dq-observation['joint_velocity'])+tau

    def close(self):
        if self.adapter:self.adapter.stop()
