import os
import time
import json
import logging
from datetime import datetime

class PerformanceLogger:
    def __init__(self, log_dir="logs"):
        # Create logs directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamp for this session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up standard Python logger for general events
        self.logger = logging.getLogger("performance_log")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for main log
        main_log_file = os.path.join(self.log_dir, f"main_{self.timestamp}.log")
        file_handler = logging.FileHandler(main_log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Create voice-specific log files
        self.voice_logs = {}
        for voice in ['voice1', 'voice2', 'voice3', 'voice4', 'voice5', 'solo', 'drone']:
            log_file = os.path.join(self.log_dir, f"{voice}_{self.timestamp}.txt")
            self.voice_logs[voice] = open(log_file, 'w')
            self.voice_logs[voice].write(f"# {voice} OSC data log - {self.timestamp}\n")
            if voice == 'voice4':
                self.voice_logs[voice].write("# timestamp,type,frequency,amplitude,duration,noise_factor\n")
            elif voice == 'voice5':
                self.voice_logs[voice].write("# timestamp,type,frequency,amplitude,duration,noise_factor,drum_type\n")
            else:
                self.voice_logs[voice].write("# timestamp,frequency,amplitude,gate,pan\n")
            self.voice_logs[voice].flush()
        
        # Create comprehensive audio analysis log
        self.audio_log_file = os.path.join(self.log_dir, f"audio_analysis_{self.timestamp}.csv")
        self.audio_log = open(self.audio_log_file, 'w')
        self.audio_log.write("# Audio input and analysis log\n")
        self.audio_log.write("timestamp,instant_pitch,avg_pitch,onset_detected,onset_rate,activity_level,energy_level\n")
        self.audio_log.flush()
        
        # Create comprehensive system events log  
        self.system_log_file = os.path.join(self.log_dir, f"system_events_{self.timestamp}.csv")
        self.system_log = open(self.system_log_file, 'w')
        self.system_log.write("# System events and triggers log\n")
        self.system_log.write("timestamp,event_type,voice,trigger_reason,activity_level,phase,additional_data\n")
        self.system_log.flush()
        
        # Create musical decision log (for transparency and trust)
        self.decision_log_file = os.path.join(self.log_dir, f"decisions_{self.timestamp}.csv")
        self.decision_log = open(self.decision_log_file, 'w')
        self.decision_log.write("# Musical decision transparency log\n")
        self.decision_log.write("timestamp,mode,voice_type,trigger_midi,trigger_note,context_tokens,context_consonance,context_melody_tendency,request_primary,request_secondary,pattern_match_score,generated_notes,reasoning,confidence\n")
        self.decision_log.flush()
        
        # Create MIDI output log (all MIDI messages sent)
        self.midi_log_file = os.path.join(self.log_dir, f"midi_output_{self.timestamp}.csv")
        self.midi_log = open(self.midi_log_file, 'w')
        self.midi_log.write("# MIDI output log - all messages sent\n")
        self.midi_log.write("timestamp,message_type,port,channel,note,velocity,duration,cc_number,cc_value,pitch_bend,pressure,additional_data\n")
        self.midi_log.flush()

    def log_voice_event(self, voice_num, frequency, amplitude, state, gate=1, pan=0):
        """Log a voice event with OSC parameters"""
        if isinstance(voice_num, str):
            voice_key = voice_num  # If voice_num is already a string like 'voice1'
        else:
            voice_key = f"voice{voice_num}"
        
        timestamp = time.time()
        
        # Log to the voice-specific file
        if voice_key in self.voice_logs:
            log_line = f"{timestamp:.6f},{frequency:.3f},{amplitude:.3f},{gate},{pan}\n"
            self.voice_logs[voice_key].write(log_line)
            self.voice_logs[voice_key].flush()
        
        # Also log to main logger
        self.logger.info(f"Voice event: {voice_key}, freq={frequency:.2f}, amp={amplitude:.2f}, state={state}")

    def log_osc_event(self, event_data):
        """Log an OSC event to the appropriate voice file"""
        voice_key = event_data.get('voice', 'unknown')
        timestamp = event_data.get('timestamp', time.time())
        frequency = event_data.get('frequency', 0)
        amplitude = event_data.get('amplitude', 0)
        gate = event_data.get('gate', 0)
        pan = event_data.get('pan', 0)
        
        # Log to the voice-specific file
        if voice_key in self.voice_logs:
            log_line = f"{timestamp:.6f},{frequency:.3f},{amplitude:.3f},{gate},{pan}\n"
            self.voice_logs[voice_key].write(log_line)
            self.voice_logs[voice_key].flush()
        
        # Also log to main logger
        event_type = event_data.get('event_type', 'unknown')
        self.logger.info(f"OSC {event_type}: {voice_key}, freq={frequency:.2f}, amp={amplitude:.2f}")

    def log_rhythm_event(self, event_type, voice, bpm, beat_interval, time_interval, pattern, timestamp):
        """Log a rhythm event"""
        self.logger.info(f"Rhythm {event_type}: {voice}, bpm={bpm:.1f}, interval={time_interval:.2f}s")

    def log_performance_event(self, event_data):
        """Log a general performance event"""
        self.logger.info(f"Performance event: {json.dumps(event_data)}")

    def log_drone_event(self, event_data):
        """Log a drone event to the drone log file"""
        timestamp = event_data.get('timestamp', time.time())
        frequency = event_data.get('frequency', 0)
        amplitude = event_data.get('amplitude', 0)
        
        # Log to the drone file
        if 'drone' in self.voice_logs:
            log_line = f"{timestamp:.6f},{frequency:.3f},{amplitude:.3f},1,-0.5\n"
            self.voice_logs['drone'].write(log_line)
            self.voice_logs['drone'].flush()
        
        # Also log to main logger
        self.logger.info(f"Drone event: freq={frequency:.2f}, amp={amplitude:.2f}")

    def log_percussion_event(self, event_data):
        """Log a percussion event to the voice4 log file"""
        timestamp = event_data.get('timestamp', time.time())
        sound_type = event_data.get('type', 'unknown')
        frequency = event_data.get('frequency', 0)
        amplitude = event_data.get('amplitude', 0)
        duration = event_data.get('duration', 0)
        noise_factor = event_data.get('noise_factor', 0)
        
        # Log to the voice4 file
        if 'voice4' in self.voice_logs:
            log_line = f"{timestamp:.6f},{sound_type},{frequency:.1f},{amplitude:.3f},{duration:.3f},{noise_factor:.2f}\n"
            self.voice_logs['voice4'].write(log_line)
            self.voice_logs['voice4'].flush()
        
        # Also log to main logger
        self.logger.info(f"Percussion: {sound_type} at {frequency:.1f}Hz, amp={amplitude:.3f}")

    def log_voice5_event(self, event_data):
        """Log a voice5 drum event"""
        timestamp = event_data.get('timestamp', time.time())
        sound_type = event_data.get('type', 'unknown')
        frequency = event_data.get('frequency', 0)
        amplitude = event_data.get('amplitude', 0)
        duration = event_data.get('duration', 0)
        noise_factor = event_data.get('noise_factor', 0)
        drum_type = event_data.get('drum_type', 0)
        
        # Log to the voice5 file
        if 'voice5' in self.voice_logs:
            log_line = f"{timestamp:.6f},{sound_type},{frequency:.1f},{amplitude:.3f},{duration:.3f},{noise_factor:.2f},{drum_type}\n"
            self.voice_logs['voice5'].write(log_line)
            self.voice_logs['voice5'].flush()
        
        # Also log to main logger
        self.logger.info(f"Voice5 Drum: {sound_type} (type {drum_type}) at {frequency:.1f}Hz, amp={amplitude:.3f}")

    def log_voice3_event(self, event_data):
        """Log a voice3 bass event"""
        timestamp = event_data.get('timestamp', time.time())
        sound_type = event_data.get('type', 'unknown')
        frequency = event_data.get('frequency', 0)
        amplitude = event_data.get('amplitude', 0)
        duration = event_data.get('duration', 0)
        noise_factor = event_data.get('noise_factor', 0)
        bass_type = event_data.get('bass_type', 0)
        
        # Log to the voice3 file
        if 'voice3' in self.voice_logs:
            log_line = f"{timestamp:.6f},{sound_type},{frequency:.1f},{amplitude:.3f},{duration:.3f},{noise_factor:.2f},{bass_type}\n"
            self.voice_logs['voice3'].write(log_line)
            self.voice_logs['voice3'].flush()
        
        # Also log to main logger
        self.logger.info(f"Voice3 Bass: {sound_type} (type {bass_type}) at {frequency:.1f}Hz, amp={amplitude:.3f}")

    def log_audio_analysis(self, instant_pitch, avg_pitch, onset_detected, onset_rate, activity_level, energy_level):
        """Log audio input analysis data"""
        timestamp = time.time()
        # Safety checks for None values
        onset_detected = onset_detected if onset_detected is not None else False
        instant_pitch = instant_pitch if instant_pitch is not None else 0.0
        avg_pitch = avg_pitch if avg_pitch is not None else 0.0
        onset_rate = onset_rate if onset_rate is not None else 0.0
        activity_level = activity_level if activity_level is not None else 0.0
        energy_level = energy_level if energy_level is not None else 0.0
        
        log_line = f"{timestamp:.6f},{instant_pitch:.1f},{avg_pitch:.1f},{int(onset_detected)},{onset_rate:.2f},{activity_level:.3f},{energy_level:.6f}\n"
        self.audio_log.write(log_line)
        self.audio_log.flush()

    def log_system_event(self, event_type, voice, trigger_reason, activity_level, phase, additional_data=""):
        """Log system events and triggers"""
        timestamp = time.time()
        # Escape commas in additional_data
        additional_data_clean = str(additional_data).replace(',', ';')
        log_line = f"{timestamp:.6f},{event_type},{voice},{trigger_reason},{activity_level:.3f},{phase},{additional_data_clean}\n"
        self.system_log.write(log_line)
        self.system_log.flush()
    
    def log_musical_decision(self, decision_explanation):
        """
        Log a musical decision with full context for transparency
        
        Args:
            decision_explanation: DecisionExplanation object from MusicalDecisionExplainer
        """
        timestamp = decision_explanation.timestamp
        mode = decision_explanation.mode
        voice_type = decision_explanation.voice_type
        
        # Extract trigger info
        trigger_midi = decision_explanation.trigger_event.get('midi', 0)
        trigger_note = self._midi_to_note_name(trigger_midi)
        
        # Extract context
        context = decision_explanation.recent_context
        tokens = context.get('gesture_tokens', [])
        tokens_str = str(tokens[-3:]) if tokens else "[]"
        consonance = context.get('avg_consonance', 0.0)
        melody_tendency = context.get('melodic_tendency', 0.0)
        
        # Extract request parameters
        request = decision_explanation.request_params
        request_primary = self._format_request_param(request.get('primary', request))
        request_secondary = self._format_request_param(request.get('secondary', {}))
        
        # Pattern matching
        pattern_score = decision_explanation.pattern_match_score if decision_explanation.pattern_match_score is not None else ""
        
        # Generated output
        notes = decision_explanation.generated_notes
        notes_str = str(notes[:8]) if len(notes) <= 8 else str(notes[:8]) + "..."
        
        # Reasoning and confidence
        reasoning = decision_explanation.reasoning.replace(',', ';')  # Escape commas
        confidence = decision_explanation.confidence
        
        # Write log line
        log_line = (f"{timestamp:.6f},{mode},{voice_type},{trigger_midi},{trigger_note},"
                   f"{tokens_str},{consonance:.2f},{melody_tendency:.2f},"
                   f"{request_primary},{request_secondary},{pattern_score},"
                   f"{notes_str},{reasoning},{confidence:.2f}\n")
        
        self.decision_log.write(log_line)
        self.decision_log.flush()
    
    def _format_request_param(self, request_dict):
        """Format request parameter for CSV logging"""
        if not request_dict:
            return ""
        
        param = request_dict.get('parameter', '')
        req_type = request_dict.get('type', '')
        value = request_dict.get('value', '')
        weight = request_dict.get('weight', '')
        
        return f"{param}:{req_type}:{value}:{weight}".replace(',', ';')
    
    def _midi_to_note_name(self, midi):
        """Convert MIDI number to note name"""
        if midi <= 0:
            return "silence"
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = note_names[midi % 12]
        return f"{note}{octave}"

    def log_osc_message_sent(self, voice, message_data):
        """Log when OSC messages are sent to SuperCollider"""
        self.log_system_event(
            event_type="osc_sent",
            voice=voice,
            trigger_reason="message_dispatch",
            activity_level=0.0,
            phase="unknown", 
            additional_data=str(message_data)
        )
    
    def log_midi_message(self, message_type, port="", channel=0, note=0, velocity=0, 
                        duration=0.0, cc_number=0, cc_value=0, pitch_bend=0, 
                        pressure=0, additional_data=""):
        """
        Log all MIDI messages sent to external devices
        
        Args:
            message_type: Type of MIDI message (note_on, note_off, cc, pitch_bend, etc.)
            port: MIDI port name
            channel: MIDI channel (1-16)
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            duration: Note duration in seconds (for note_on)
            cc_number: Control change number (0-127)
            cc_value: Control change value (0-127)
            pitch_bend: Pitch bend value (-8192 to 8191)
            pressure: Aftertouch pressure (0-127)
            additional_data: Any extra information (comma-escaped)
        """
        timestamp = time.time()
        additional_data_clean = str(additional_data).replace(',', ';')
        
        log_line = (f"{timestamp:.6f},{message_type},{port},{channel},"
                   f"{note},{velocity},{duration:.4f},{cc_number},{cc_value},"
                   f"{pitch_bend},{pressure},{additional_data_clean}\n")
        
        self.midi_log.write(log_line)
        self.midi_log.flush()

    def close(self):
        """Close all log files"""
        for file in self.voice_logs.values():
            file.close()
        self.audio_log.close()
        self.system_log.close()
        self.decision_log.close()
        self.midi_log.close()
