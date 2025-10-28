# Autonomous Generation - Adjustable Parameters

## 🎛️ New Behavior

### **Default Settings:**
```bash
python MusicHal_9000.py
```

- **While you sing:** 
  - 🎤 Melody: **Silent** (gives you space)
  - 🎸 Bass: **50% probability** (sparse accompaniment)
  
- **When you stop (1.5s):**
  - 🎤 Melody: **Active** (every ~1.5s)
  - 🎸 Bass: **Active** (every ~1.5s)

---

## 🔧 Adjustable Parameters

### 1. **Bass Accompaniment Density**
```bash
# No bass while you sing (complete silence)
python MusicHal_9000.py --bass-accompaniment 0.0

# More bass (75% probability)
python MusicHal_9000.py --bass-accompaniment 0.75

# Bass always plays (100%)
python MusicHal_9000.py --bass-accompaniment 1.0
```

### 2. **Melody Response Speed**
```bash
# Slower melody response (every 2 seconds when you're quiet)
python MusicHal_9000.py --autonomous-interval 4.0

# Faster melody response (every 1 second when you're quiet)
python MusicHal_9000.py --autonomous-interval 2.0

# Very active melody (every 0.75s when you're quiet)
python MusicHal_9000.py --autonomous-interval 1.5
```

### 3. **Melody While You Sing**
```bash
# Allow sparse melody while you sing (20% probability)
python MusicHal_9000.py --melody-while-active

# Default: melody is completely silent while you sing
```

---

## 🎵 Example Configurations

### **More Accompaniment (Jazz Combo Feel)**
```bash
python MusicHal_9000.py \
  --bass-accompaniment 0.8 \
  --melody-while-active \
  --autonomous-interval 2.5
```
- Bass plays 80% while you sing
- Melody plays sparse accents (20%)
- Quick response when you stop (1.25s)

### **More Space (Singer-Songwriter Feel)**
```bash
python MusicHal_9000.py \
  --bass-accompaniment 0.3 \
  --autonomous-interval 4.0
```
- Minimal bass while you sing (30%)
- No melody while you sing
- Moderate response when you stop (2s)

### **Very Active Response (Duet Feel)**
```bash
python MusicHal_9000.py \
  --bass-accompaniment 0.6 \
  --melody-while-active \
  --autonomous-interval 2.0
```
- Bass plays 60% while you sing
- Melody adds sparse responses
- Fast response when you stop (1s)

---

## 📊 How It Works

```
YOU SING (active)
├─ Melody: Silent (unless --melody-while-active, then 20% probability)
├─ Bass: Plays with bass_accompaniment probability (default 50%)
└─ Status: 👂 LISTEN

YOU STOP (1.5s silence)
├─ Immediate response triggered
├─ Melody: Generates every autonomous_interval * 0.5 seconds
├─ Bass: Generates every autonomous_interval * 0.5 seconds
└─ Status: 🤖 AUTO

YOU START AGAIN
├─ Melody: Stops immediately
├─ Bass: Reduces to accompaniment mode
└─ Status: 👂 LISTEN
```

---

## 💡 Recommended Starting Points

**If melody is too shy:**
```bash
python MusicHal_9000.py --autonomous-interval 2.0
```

**If you want more bass groove:**
```bash
python MusicHal_9000.py --bass-accompaniment 0.7
```

**If you want fuller arrangement:**
```bash
python MusicHal_9000.py --melody-while-active --bass-accompaniment 0.7
```

**If you want maximum space:**
```bash
python MusicHal_9000.py --bass-accompaniment 0.2 --autonomous-interval 5.0
```
