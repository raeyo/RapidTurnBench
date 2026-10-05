"""Apply a small, reversible input adapter to the pinned official deployment.

CSV playback is clocked by an independent input thread (not simulation time),
and cannot start CONTROL itself. This adapter adds command acknowledgement;
it does not modify inference, planning, gains, safety, or control logic.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp'
text = source.read_text()
if '// RAPIDTURN_INPUT_ADAPTER' not in text:
    needle = '      // Update input interface (poll for new data)'
    block = r'''
      // RAPIDTURN_INPUT_ADAPTER: optional sim-only acknowledged command input.
      if (const char* path = std::getenv("RAPIDTURN_COMMAND_FILE")) {
        std::ifstream stream(path);
        long long sequence; int mode, start;
        double speed, yaw, height;
        if (stream >> sequence >> mode >> speed >> yaw >> height >> start) {
          if (planner_) {
            planner_->planner_state_.enabled = (start >= 0);
            MovementState movement(mode, {std::cos(yaw), std::sin(yaw), 0.0},
                                   {std::cos(yaw), std::sin(yaw), 0.0}, speed, height);
            if (speed == 0.0) movement.movement_direction = {0.0, 0.0, 0.0};
            movement_state_buffer_.SetData(movement);
            operator_state.start = (start == 1);
            operator_state.play = true;
          }
          static long long previous = -1;
          if (sequence != previous) {
            auto state = low_state_buffer_.GetDataWithTime().data;
            std::ofstream ack(std::string(path) + ".ack");
            ack << sequence << " " << (state ? state->tick() : 0) << std::endl;
            std::cout << "[RapidTurn] command " << sequence << " mode=" << mode
                      << " speed=" << speed << " yaw=" << yaw << std::endl;
            previous = sequence;
          }
        }
        return;
      }
'''
    assert needle in text
    text = text.replace(needle, block + '\n' + needle, 1)
    needle = '        planner_config.model_path = planner_path;'
    assert needle in text
    text = text.replace(needle, needle + '\n        if (const char* seed = std::getenv("RAPIDTURN_SEED")) planner_config.initial_random_seed = std::stoi(seed);', 1)
    # The simulator has no hardware motion-switch RPC server. Keep the official
    # service logic, but avoid spending 5 s waiting on an absent robot service.
    text = text.replace('msc_->SetTimeout(5.0f);', 'msc_->SetTimeout(std::getenv("RAPIDTURN_COMMAND_FILE") ? 0.01f : 5.0f);', 1)
    source.write_text(text)
audio = root / 'third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/audio_thread/audio_thread.cpp'
audio_text = audio.read_text()
if 'RAPIDTURN_COMMAND_FILE' not in audio_text:
    audio_text = '#include <cstdlib>\n' + audio_text
    audio_text = audio_text.replace('client_.SetTimeout(10.0f);', 'client_.SetTimeout(std::getenv("RAPIDTURN_COMMAND_FILE") ? 0.01f : 10.0f);', 1)
    audio_text = audio_text.replace('client_.SetVolume(100);', 'if (!std::getenv("RAPIDTURN_COMMAND_FILE")) client_.SetVolume(100);', 1)
    audio.write_text(audio_text)
text = source.read_text()
text = text.replace('ChannelFactory::Instance()->Init(0, networkInterface);', 'ChannelFactory::Instance()->Init(std::getenv("RAPIDTURN_DOMAIN_ID") ? std::stoi(std::getenv("RAPIDTURN_DOMAIN_ID")) : 0, networkInterface);', 1)
if '// RAPIDTURN_LOCKSTEP' not in text:
    needle = '    ThreadPtr input_thread_ptr_, command_writer_ptr_, control_thread_ptr_, planner_thread_ptr_;'
    assert needle in text
    text = text.replace(needle, needle + '\n    std::jthread benchmark_thread_;', 1)
    begin = text.index('      // create threads\n')
    finish = text.index('      SetThreadPriority();', begin)
    original_threads = text[begin:finish]
    schedule = r'''
      // RAPIDTURN_LOCKSTEP: sim-time scheduling; inference/control methods are unchanged.
      if (std::getenv("RAPIDTURN_LOCKSTEP")) {
        benchmark_thread_ = std::jthread([this](std::stop_token stop) {
          int64_t previous_tick = -1;
          while (!stop.stop_requested() && !operator_state.stop) {
            auto state = low_state_buffer_.GetDataWithTime().data;
            if (state && int64_t(state->tick()) > previous_tick && state->tick() % 20 == 0) {
              const auto tick = state->tick();
              Input();
              if (planner_ && tick % 100 == 0) Planner();
              Control();
              LowCommandWriter();
              previous_tick = tick;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        });
      } else {
'''
    text = text[:begin] + schedule + original_threads + '\n      }\n' + text[finish:]
    needle = '      operator_state.stop = true;\n\n      if (control_thread_ptr_)'
    assert needle in text
    text = text.replace(needle, '      operator_state.stop = true;\n      if (benchmark_thread_.joinable()) { benchmark_thread_.request_stop(); benchmark_thread_.join(); }\n\n      if (control_thread_ptr_)', 1)
    needle = '        dds_low_command.crc() = Crc32Core'
    assert needle in text
    text = text.replace(needle, '        if (std::getenv("RAPIDTURN_LOCKSTEP")) {\n          auto state = low_state_buffer_.GetDataWithTime().data;\n          if (state) dds_low_command.reserve()[0] = state->tick();\n        }\n' + needle, 1)
    source.write_text(text)
source.write_text(text)
print('SONIC input adapter applied (inference/control/planner algorithms unchanged).')
