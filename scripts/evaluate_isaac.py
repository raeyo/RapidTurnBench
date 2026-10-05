#!/usr/bin/env python3
"""Evaluate reviewed local model submissions in real Isaac Lab / PhysX."""
import argparse
from pathlib import Path
import sys
import json
import os
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--submission',type=Path,default=ROOT/'submissions/sonic')
parser.add_argument('--profile',type=Path,default=ROOT/'configs/benchmark/isaac_quickcheck.yaml')
parser.add_argument('--output',type=Path,default=ROOT/'results/isaac/quickcheck')
parser.add_argument('--angles',type=float,nargs='+')
parser.add_argument('--speeds',type=float,nargs='+')
parser.add_argument('--seeds',type=int,nargs='+')
parser.add_argument('--video',action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.video:args.enable_cameras=True
launcher=AppLauncher(args);app=launcher.app
import numpy as np
import torch
import yaml
import isaaclab.sim as s
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply_inverse
from rapidturn.isaac_asset import prepare_asset,JOINT_NAMES
from rapidturn.submission import load_submission,validate_action
from rapidturn.isaac_metrics import evaluate,euler
from rapidturn.isaac_report import overlay,write_report


def main():
    config=yaml.safe_load(args.profile.read_text())
    if config['simulation']['backend']!='isaaclab':raise ValueError('Only Isaac Lab profiles accepted')
    args.output.mkdir(parents=True,exist_ok=True)
    urdf,profile=prepare_asset()
    (ROOT/'configs/robot/isaac_g1.json').write_text(json.dumps(profile,indent=2))
    limits=np.asarray(profile['effort_limits'])
    dt=config['simulation']['physics_dt'];control_dt=1/config['simulation']['control_frequency']
    every=round(control_dt/dt)
    sim=s.SimulationContext(s.SimulationCfg(dt=dt,device=args.device,render_interval=every))
    ground=s.GroundPlaneCfg(physics_material=s.RigidBodyMaterialCfg(static_friction=1,dynamic_friction=1,restitution=0))
    ground.func('/World/Ground',ground)
    light=s.DomeLightCfg(intensity=2200,color=(.85,.9,1.));light.func('/World/Light',light)
    cfg=ArticulationCfg(prim_path='/World/G1',spawn=s.UrdfFileCfg(asset_path=str(urdf),fix_base=False,merge_fixed_joints=True,
        joint_drive=s.UrdfConverterCfg.JointDriveCfg(target_type='none',gains=s.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0,damping=0)),
        rigid_props=s.RigidBodyPropertiesCfg(linear_damping=0,angular_damping=0,max_depenetration_velocity=1),
        articulation_props=s.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,profile['standing_root_height']),joint_pos=dict(zip(JOINT_NAMES,profile['nominal_joint_position']))),
        actuators={'all':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0,damping=0,effort_limit_sim=200,armature=profile['armature'])})
    robot=Articulation(cfg)
    camera=None
    if args.video:
        from isaaclab.sensors import Camera,CameraCfg
        camera=Camera(CameraCfg(prim_path='/World/Camera',update_period=0,height=540,width=720,data_types=['rgb'],
                                spawn=s.PinholeCameraCfg(focal_length=24,horizontal_aperture=28,clipping_range=(.05,100))))
    sim.reset()
    indices=torch.tensor([robot.joint_names.index(n) for n in JOINT_NAMES],device=sim.device)
    torso=robot.body_names.index('torso_link')
    previous=np.zeros(29)
    def observation(timestamp):
        q=robot.data.root_quat_w
        ang=quat_apply_inverse(q,robot.data.root_ang_vel_w)[0]
        torso_q=robot.data.body_quat_w[:,torso]
        torso_ang=quat_apply_inverse(torso_q,robot.data.body_ang_vel_w[:,torso])[0]
        gravity=quat_apply_inverse(q,torch.tensor([[0.,0.,-1.]],device=sim.device))[0]
        def array(x):return x.detach().cpu().numpy().copy()
        return dict(timestamp=float(timestamp),dt=control_dt,root_position=array(robot.data.root_pos_w[0]),
                    root_quaternion=array(q[0]),root_linear_velocity=array(robot.data.root_lin_vel_w[0]),
                    root_angular_velocity=array(robot.data.root_ang_vel_w[0]),body_angular_velocity=array(ang),
                    projected_gravity=array(gravity),torso_quaternion=array(torso_q[0]),torso_angular_velocity=array(torso_ang),
                    joint_position=array(robot.data.joint_pos[0,indices]),joint_velocity=array(robot.data.joint_vel[0,indices]),
                    previous_action=previous.copy())
    context=dict(project_root=str(ROOT),asset_profile=profile,device=sim.device,episode_dir=str(args.output/'episodes/initial'))
    policy,manifest,hashes=load_submission(args.submission,context)
    config['task']['implementation_status']='experimental'
    config['asset_profile']=profile
    angles=args.angles or config['command']['turn_angles'];speeds=args.speeds or [config['command']['speed']];seeds=args.seeds or config['evaluation']['seeds']
    config['run_conditions']=dict(angles=angles,speeds=speeds,seeds=seeds)
    (args.output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ['torch','isaaclab','isaacsim']}
    provenance=dict(backend='isaaclab',model=manifest,artifact_hashes=hashes,asset_profile=profile,versions=versions,
                    profile_version=config['task']['profile_version'],metric_version='isaac-response-v1',
                    source_sha256={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'rapidturn/isaac_metrics.py',ROOT/'rapidturn/isaac_asset.py']})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    records=[]
    nominal_pose=robot.data.default_root_state[:,:7].clone();nominal_vel=torch.zeros((1,6),device=sim.device)
    switch=config['timing']['standing_duration']+config['timing']['approach_duration'];duration=switch+config['timing']['post_switch_duration']
    videos=args.output/'videos';videos.mkdir(exist_ok=True)
    def set_action(action):
        nonlocal previous
        previous,clipped=validate_action(action,limits)
        effort=torch.zeros_like(robot.data.joint_pos);effort[0,indices]=torch.as_tensor(previous,device=sim.device,dtype=effort.dtype)
        robot.set_joint_effort_target(effort);robot.write_data_to_sim()
        return clipped
    def physics():
        # A submitted motor controller may recompute torque at physics frequency,
        # using targets from its most recent 50 Hz policy action.
        if hasattr(policy,'control'):set_action(policy.control(observation(-1)))
        sim.step(render=False);robot.update(dt)
    zero_command=dict(target_velocity_xy=np.zeros(2),facing_yaw=0.)
    try:
      for speed in speeds:
       for angle in angles:
        for seed in seeds:
            eid=f'v{speed:g}_turn{angle:+g}_seed{seed}'
            directory=args.output/'episodes'/eid;directory.mkdir(parents=True,exist_ok=True)
            context['episode_dir']=str(directory)
            record=dict(model=manifest['model_name'],episode_id=eid,requested_speed_m_s=float(speed),turn_angle_deg=float(angle),seed=int(seed),status='completed')
            rows=[];writer=None;clipping=0;inference=[]
            try:
                robot.write_root_pose_to_sim(nominal_pose);robot.write_root_velocity_to_sim(nominal_vel)
                robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.default_joint_vel));robot.reset();previous=np.zeros(29)
                policy.reset(seed)
                # Identical root pose support during startup; never during measured task.
                initialization=[]
                for n in range(round(4/control_dt)):
                    obs=observation(-5+n*control_dt);set_action(policy.act(obs,zero_command))
                    initialization.append(np.r_[obs['timestamp'],obs['root_position'],obs['torso_quaternion'],obs['joint_position']])
                    for _ in range(every):
                        robot.write_root_pose_to_sim(nominal_pose);robot.write_root_velocity_to_sim(nominal_vel);physics()
                for n in range(round(1/control_dt)):
                    set_action(policy.act(observation(-1+n*control_dt),zero_command))
                    obs=observation(-1+n*control_dt)
                    initialization.append(np.r_[obs['timestamp'],obs['root_position'],obs['torso_quaternion'],obs['joint_position']])
                    for _ in range(every):physics()
                np.savez_compressed(directory/'initialization.npz',trajectory=np.array(initialization))
                obs=observation(0)
                height=float(obs['root_position'][2]);tilt=float(np.rad2deg(np.linalg.norm(euler(obs['torso_quaternion'])[:2])))
                if height<.5 or tilt>30 or np.linalg.norm(obs['root_linear_velocity'][:2])>.4:
                    raise RuntimeError(f'initialization_failed: height={height:.3f}, tilt={tilt:.1f}')
                start_yaw=float(euler(obs['torso_quaternion'])[2]);target_yaw=start_yaw;cue_yaw=start_yaw
                render_episode=args.video and seed==config['output']['representative_seed']
                if render_episode:
                    import imageio.v2 as imageio
                    writer=imageio.get_writer(videos/f'{eid}.mp4',fps=25,codec='libx264',macro_block_size=2)
                for n in range(round(duration/control_dt)+1):
                    t=n*control_dt;obs=observation(t)
                    if abs(t-switch)<1e-8:
                        cue_yaw=float(euler(obs['torso_quaternion'])[2]);target_yaw=cue_yaw+np.deg2rad(angle)
                    command_speed=float(speed) if t>=config['timing']['standing_duration'] else 0.
                    command=dict(target_velocity_xy=command_speed*np.array([np.cos(target_yaw),np.sin(target_yaw)]),facing_yaw=target_yaw)
                    row={k:obs[k] for k in ['timestamp','root_position','root_quaternion','root_linear_velocity','root_angular_velocity','torso_quaternion','joint_position','joint_velocity','previous_action']}
                    row.update(command_yaw=target_yaw,command_velocity_xy=command['target_velocity_xy'].copy());rows.append(row)
                    if render_episode and n%2==0:
                        pos=obs['root_position'];yaw=start_yaw if t<switch else cue_yaw
                        eye=pos+np.array([-3*np.cos(yaw),-3*np.sin(yaw),1.7]);look=pos+np.array([0,0,.4])
                        camera.set_world_poses_from_view(torch.tensor(eye[None],device=sim.device,dtype=torch.float32),torch.tensor(look[None],device=sim.device,dtype=torch.float32))
                        sim.render();camera.update(.04,force_recompute=True)
                        rgb=camera.data.output['rgb'][0].cpu().numpy()
                        frame=overlay(rgb,t,angle,speed,float(np.linalg.norm(obs['root_linear_velocity'][:2])),manifest['model_name'],switch)
                        writer.append_data(frame)
                        if n==round((switch+1)/control_dt):__import__('PIL.Image',fromlist=['Image']).fromarray(frame).save(videos/f'{eid}.png')
                    tilt=float(np.rad2deg(np.linalg.norm(euler(obs['torso_quaternion'])[:2])))
                    if obs['root_position'][2]<height*config['failure']['min_root_height_ratio'] or tilt>config['failure']['max_torso_tilt_deg']:
                        record['status']='fall';break
                    if n==round(duration/control_dt):break
                    before=time.monotonic();action=policy.act(obs,command);inference.append(time.monotonic()-before);clipping+=set_action(action)
                    for _ in range(every):physics()
                log={k:np.asarray([r[k] for r in rows]) for k in rows[0]};log['standing_root_height']=height
                np.savez_compressed(directory/'episode.npz',**log)
                record.update(evaluate(log,config))
                if record['status']=='completed' and not record['approach_ready']:record['status']='approach_failed'
                record['torque_clipped_joint_ticks']=clipping;record['mean_inference_wall_s']=float(np.mean(inference)) if inference else None
            except Exception as exc:
                record['status']='invalid_action' if 'invalid_action' in str(exc) else 'initialization_failed' if 'initialization_failed' in str(exc) else 'policy_error'
                record['error']=str(exc)
                (directory/'error.txt').write_text(traceback.format_exc())
                print('EPISODE ERROR',eid,str(exc),flush=True)
                # Preserve partial physical state when policy execution fails.
                if rows and not (directory/'episode.npz').exists():
                    log={k:np.asarray([r[k] for r in rows]) for k in rows[0]};log['standing_root_height']=height
                    np.savez_compressed(directory/'episode.npz',**log);record.update(evaluate(log,config))
            finally:
                if writer:writer.close()
            (directory/'metrics.json').write_text(json.dumps(record,indent=2));records.append(record)
            print('EPISODE',json.dumps(record),flush=True)
            # Normalize keys for CSV, including failures.
            keys=set().union(*(r.keys() for r in records))
            for r in records:
                for k in keys:r.setdefault(k,None)
            write_report(args.output,records,config)
    finally:policy.close()
    clips=sorted(p for p in videos.glob('*.mp4') if p.name!='demo.mp4')
    if clips:
        import imageio.v2 as imageio
        with imageio.get_writer(videos/'demo.mp4',fps=25,codec='libx264',macro_block_size=2) as merged:
            for clip in clips:
                reader=imageio.get_reader(clip)
                for frame in reader:merged.append_data(frame)
                reader.close()
    print('REPORT',args.output/'report.html',flush=True)

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    # Kit's app.close() can hang after headless runs; all outputs are already written.
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
