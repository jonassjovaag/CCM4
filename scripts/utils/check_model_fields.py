import json

with open('JSON/Itzama_271025_1709.json') as f:
    model = json.load(f)
    
frames = model.get('audio_frames', {})
print(f"Total frames: {len(frames)}")

if frames:
    sample_frame = list(frames.values())[0]
    print(f"Sample frame type: {type(sample_frame)}")
    
    if isinstance(sample_frame, dict):
        audio_data = sample_frame.get('audio_data', sample_frame)
        print(f"Audio data keys: {list(audio_data.keys())[:30]}")
