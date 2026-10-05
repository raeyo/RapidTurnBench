"""Fixed G1 asset contract; no simulator import is needed to inspect it."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/GR00T-WholeBodyControl'
JOINT_NAMES = [f'{side}_{name}_joint' for side in ('left','right')
               for name in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
JOINT_NAMES += ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
JOINT_NAMES += [f'{side}_{name}_joint' for side in ('left','right')
                for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
NOMINAL = [-.1,0,0,.3,-.2,0]*2 + [0,0,0] + [.2,.2,0,.6,0,0,0] + [.2,-.2,0,.6,0,0,0]

def prepare_asset():
    source = UPSTREAM / 'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf'
    if not source.exists():
        raise FileNotFoundError('Fetch pinned upstream G1 URDF/meshes; see README.md')
    tree = ET.parse(source)
    joint_map = {j.attrib['name']:j for j in tree.getroot().findall('joint') if j.attrib['type'] != 'fixed'}
    if set(joint_map) != set(JOINT_NAMES):
        raise ValueError('G1 asset must have exactly the canonical 29 body joints')
    digest = hashlib.sha256(source.read_bytes())
    for mesh in tree.getroot().iter('mesh'):
        path = source.parent / mesh.attrib['filename']
        if not path.exists() or path.read_bytes()[:30].startswith(b'version https://git-lfs'):
            raise FileNotFoundError(f'Missing mesh or LFS pointer: {path}')
        digest.update(path.read_bytes())
        mesh.set('filename', str(path.resolve()))
    directory = ROOT / '.cache/isaac_asset'
    directory.mkdir(parents=True,exist_ok=True)
    urdf = directory/'g1_29dof.urdf'
    text = ET.tostring(tree.getroot(),encoding='unicode')
    if not urdf.exists() or urdf.read_text()!=text:
        urdf.write_text(text)
    profile = dict(name='sonic-g1-urdf-effort-v1', joint_names=JOINT_NAMES,
                   nominal_joint_position=NOMINAL, standing_root_height=.793,
                   effort_limits=[float(joint_map[n].find('limit').attrib['effort']) for n in JOINT_NAMES],
                   asset_sha256=digest.hexdigest(), self_collisions=False, armature=.01,
                   ground_friction=1.0)
    (directory/'profile.json').write_text(json.dumps(profile,indent=2))
    return urdf, profile
