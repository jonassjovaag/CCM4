# Autonomous Behavior Architecture

## 🎯 Problem Identified

We've been **duplicating logic** in `MusicHal_9000.py` that should:
1. Be **learned from GPT-OSS insights**
2. Use **existing agent components** (`DensityController`, `BehaviorScheduler`)
3. Be **reusable** across different scripts

## 📁 New Module: `agent/autonomous_behavior_config.py`

### Components:

#### 1. **`AutonomousBehaviorConfig`** (Data class)
Holds all behavioral parameters:
- `silence_timeout`: When to switch to autonomous
- `autonomous_interval`: How often to generate
- `bass_accompaniment_probability`: Bass density while human plays
- `melody_silence_when_active`: Melody behavior
- `density_when_active/autonomous`: Density settings
- `give_space_when_active/autonomous`: Space-giving settings

#### 2. **`GPTOSSBehaviorParser`** (Parser)
Parses GPT-OSS text insights:
- `parse_silence_strategy()`: Extracts timing and silence behavior
- `parse_role_development()`: Extracts voice role relationships
- `create_config_from_gpt_oss()`: Creates full config from insights

**Example:**
```python
# GPT-OSS says: "responds quickly after 1-2 seconds of silence"
# Parser extracts: silence_timeout = 1.5

# GPT-OSS says: "bass provides sparse accompaniment"
# Parser extracts: bass_accompaniment_probability = 0.6
```

#### 3. **`AutonomousBehaviorManager`** (Controller)
Integrates with existing agent:
- `update_from_event()`: Tracks human activity
- `is_in_autonomous_mode()`: Determines current mode
- `get_voice_filter_decision()`: Decides if voice should play
- `get_density_parameters()`: Returns (density, give_space) for agent
- `should_generate_autonomous()`: Triggers autonomous generation

## 🔄 Integration Flow

```
1. STARTUP
   ├─ Load GPT-OSS insights from JSON
   ├─ GPTOSSBehaviorParser.create_config_from_gpt_oss()
   ├─ Create AutonomousBehaviorManager(config)
   └─ Configure agent.scheduler & agent.density_controller

2. AUDIO EVENT
   ├─ manager.update_from_event(event, time)
   ├─ density, give_space = manager.get_density_parameters(time)
   ├─ agent.scheduler.set_density_level(density)
   ├─ agent.scheduler.set_give_space_factor(give_space)
   ├─ decisions = agent.process_event(...)
   └─ filtered = [d for d in decisions if manager.get_voice_filter_decision(d.voice_type, time)]

3. MAIN LOOP (Autonomous Generation)
   ├─ if manager.should_generate_autonomous(time, last_gen):
   │     ├─ Create synthetic event
   │     ├─ decisions = agent.process_event(...)
   │     └─ Send MIDI
   └─ sleep(0.1)
```

## ✅ Benefits

### **Before (Current):**
- ❌ Hardcoded parameters in `MusicHal_9000.py`
- ❌ Bypasses existing `DensityController` and `BehaviorScheduler`
- ❌ No learning from GPT-OSS insights
- ❌ Not reusable

### **After (With New Module):**
- ✅ Parameters learned from GPT-OSS insights
- ✅ Uses existing agent components properly
- ✅ Reusable across scripts (MusicHal, main.py, etc.)
- ✅ Centralized behavioral logic
- ✅ Easy to test and refine

## 🎵 GPT-OSS Learning Examples

### **Example 1: Singer-Songwriter Style**
**GPT-OSS Insight:**
> "The performer uses extended silences (3-4 seconds) for reflection. 
> Bass provides minimal harmonic foundation. Melody stays completely 
> silent during vocals."

**Parsed Config:**
```python
silence_timeout = 3.5
bass_accompaniment_probability = 0.3
melody_silence_when_active = True
```

### **Example 2: Jazz Combo Style**
**GPT-OSS Insight:**
> "Quick call-and-response (1-2 seconds). Bass walks continuously. 
> Melody adds sparse commentary during solos."

**Parsed Config:**
```python
silence_timeout = 1.5
bass_accompaniment_probability = 0.8
melody_silence_when_active = False
autonomous_interval = 2.0
```

### **Example 3: Duet Style**
**GPT-OSS Insight:**
> "Roles alternate frequently. Both voices active but respect space.
> Responds immediately after pauses."

**Parsed Config:**
```python
silence_timeout = 1.0
bass_accompaniment_probability = 0.5
melody_silence_when_active = True
autonomous_interval = 2.5
```

## 🚀 Next Steps

1. **Refactor `MusicHal_9000.py`** to use new module
2. **Test GPT-OSS parsing** with real training data
3. **Add CLI overrides** (keep `--bass-accompaniment` etc for testing)
4. **Extend to `main.py`** for consistency
5. **Add more parsing patterns** as we discover them

---

**Result:** Behavioral intelligence is now **learned** not **hardcoded**! 🧠
