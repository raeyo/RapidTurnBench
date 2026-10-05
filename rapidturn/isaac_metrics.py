"""Versioned rapid-turn metrics from logged Isaac states, independent of policy."""
import numpy as np

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def euler(q):
    w,x,y,z=np.asarray(q).T
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))]).T

def causal_average(t,v,window):
    out=np.empty_like(v,dtype=float)
    for i,time in enumerate(t):
        start=np.searchsorted(t,time-window+1e-8)
        out[i]=v[start:i+1].mean(axis=0)
    return out

def evaluate(log,c):
    t=np.asarray(log['timestamp']);v=np.asarray(log['root_linear_velocity'])[:,:2]
    switch=c['timing']['standing_duration']+c['timing']['approach_duration']
    cfg=c['direction_response'];post=t>=switch-1e-8
    before=(t>=switch-c['speed_sweep']['pre_cue_actual_speed_window_s'])&(t<switch-1e-8)
    pre_speed=float(np.linalg.norm(v[before],axis=1).mean()) if before.any() else 0.
    filtered=causal_average(t,v,cfg['filter_window_s'])
    speed=np.linalg.norm(filtered,axis=1)
    target=np.asarray(log['command_yaw'])
    direction_err=np.abs(wrap(np.arctan2(filtered[:,1],filtered[:,0])-target))
    pose=euler(log['torso_quaternion'])
    facing_err=np.abs(wrap(pose[:,2]-target))
    tilt=np.rad2deg(np.linalg.norm(pose[:,:2],axis=1))
    min_speed=max(cfg['minimum_speed_m_s'],pre_speed*cfg['minimum_speed_fraction_of_pre_cue'])
    valid=post&(direction_err<np.deg2rad(cfg['direction_threshold_deg']))&(facing_err<np.deg2rad(cfg['facing_threshold_deg']))&(speed>=min_speed)
    start=None;settle=None
    approach=pre_speed>=c['speed_sweep']['minimum_approach_speed_m_s']
    if approach:
        for i in np.flatnonzero(post):
            if valid[i]:
                if start is None:start=i
                # 50 Hz samples represent [timestamp, timestamp+dt).
                if t[i]-t[start]+.02>=cfg['hold_duration_s']-1e-8:
                    settle=float(t[start]-switch);break
            else:start=None
    fall=(np.asarray(log['root_position'])[:,2]<float(log['standing_root_height'])*c['failure']['min_root_height_ratio'])|(tilt>c['failure']['max_torso_tilt_deg'])
    window=c['transient']['window_s'];transient=post&(t<switch+window-1e-8)
    full_window=bool(t[-1]>=switch+window-1e-8)
    desired=np.asarray(log['command_velocity_xy'])
    err=float(np.linalg.norm(v[transient]-desired[transient],axis=1).mean()) if transient.any() else None
    final=post&(t>=switch+c['timing']['post_switch_duration']-cfg['final_direction_error_window_s']-1e-8)
    return dict(approach_ready=bool(approach),pre_cue_actual_speed_m_s=pre_speed,
                turn_reached=settle is not None,turn_time_s=settle,
                transient_velocity_error_m_s=err,transient_window_complete=full_window,
                peak_tilt_deg=float(tilt[post].max()) if post.any() else None,
                fall=bool(fall.any()),
                final_direction_error_deg=float(np.rad2deg(direction_err[final]).mean()) if final.any() else None,
                minimum_speed_ratio=float(speed[post].min()/pre_speed) if post.any() and approach else None)
