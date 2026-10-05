"""Load a reviewed local submission and validate its common action contract."""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
import yaml
from rapidturn.isaac_asset import JOINT_NAMES

def load_submission(directory, context):
    directory = Path(directory).resolve()
    manifest = yaml.safe_load((directory/'manifest.yaml').read_text())
    if manifest.get('contract_version') != 'rapidturn-policy-v1':
        raise ValueError('Unsupported submission contract_version')
    if manifest.get('action_type') != 'joint_effort_nm':
        raise ValueError('action_type must be joint_effort_nm')
    if manifest.get('joint_names') != JOINT_NAMES:
        raise ValueError('joint_names must exactly match configs/robot/isaac_g1.json')
    file, class_name = manifest['entrypoint'].split(':')
    source = (directory/file).resolve()
    if not source.is_relative_to(directory):
        raise ValueError('Entrypoint must be inside the submission directory')
    artifacts = [source, directory/'manifest.yaml']
    for name in manifest.get('artifacts',[]):
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Invalid artifact: {name}')
        artifacts.append(path)
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    spec=importlib.util.spec_from_file_location('rapidturn_submitted_policy',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy=getattr(module,class_name)()
    policy.load(directory,context)
    return policy,manifest,hashes

def validate_action(action, limits):
    a=np.asarray(action,dtype=np.float64)
    if a.shape!=(29,) or not np.isfinite(a).all():
        raise ValueError('invalid_action: expected 29 finite joint torques')
    clipped=np.clip(a,-limits,limits)
    return clipped,int(np.count_nonzero(clipped!=a))
