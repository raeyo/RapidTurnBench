"""Project-local SONIC deployment process adapter (command file + lockstep DDS)."""
from pathlib import Path
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SONIC = ROOT / 'third_party/GR00T-WholeBodyControl'
DEPLOY = SONIC / 'gear_sonic_deploy'


class SonicAdapter:
    def __init__(self, directory, seed, domain_id=0):
        self.directory = Path(directory)
        self.command_file = self.directory / 'command.txt'
        self.sequence = 0
        self.process = None
        self.seed = seed
        self.domain_id = domain_id
        self.stdout = None

    def set_command(self, speed, yaw, start=True, acknowledge=True):
        self.sequence += 1
        temp = self.command_file.with_suffix('.tmp')
        temp.write_text(f'{self.sequence} {2 if speed else 0} {speed:.12g} {yaw:.12g} -1 {int(start)}\n')
        temp.replace(self.command_file)
        if not acknowledge:
            return None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                seq, tick = map(int, Path(str(self.command_file) + '.ack').read_text().split())
                if seq == self.sequence:
                    return tick
            except (OSError, ValueError):
                pass
            if self.process.poll() is not None:
                raise RuntimeError(f'SONIC exited: {self.process.returncode}. See {self.directory / "sonic.log"}')
            time.sleep(.001)
        raise TimeoutError('SONIC did not acknowledge the command')

    def start(self):
        binary = DEPLOY / 'target/release/g1_deploy_onnx_ref'
        if not binary.exists():
            raise FileNotFoundError('Build SONIC first: bash scripts/build_sonic.sh')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.set_command(0, 0, start=-1, acknowledge=False)
        self.stdout = (self.directory / 'sonic.log').open('w')
        env = dict(os.environ, RAPIDTURN_COMMAND_FILE=str(self.command_file.resolve()), RAPIDTURN_SEED=str(self.seed), RAPIDTURN_LOCKSTEP='1', RAPIDTURN_DOMAIN_ID=str(self.domain_id))
        args = [str(binary), 'lo', 'policy/sonic_v1_1/model_decoder.onnx', 'reference/example',
                '--encoder-file', 'policy/sonic_v1_1/model_encoder.onnx',
                '--obs-config', 'policy/sonic_v1_1/observation_config.yaml',
                '--planner-file', 'planner/target_vel/V2/planner_sonic.onnx',
                '--input-type', 'keyboard', '--disable-crc-check', '--planner-precision', '32', '--output-type', 'zmq', '--zmq-out-port', str(5557+self.domain_id)]
        self.process = subprocess.Popen(args, cwd=DEPLOY, env=env, stdin=subprocess.DEVNULL,
                                        stdout=self.stdout, stderr=subprocess.STDOUT, start_new_session=True)

    def log_contains(self, message):
        return message in (self.directory / 'sonic.log').read_text(errors='replace')

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.stdout:
            self.stdout.close()
