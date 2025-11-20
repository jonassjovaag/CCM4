# Travelling until I am moved: MusicHal 9000 and Chandra trainer

**An Artistic Research Exposition on Gesture-Based Memory, Dual Perception, and Trust in AI Musical Partnership**

by Jonas Howden Sjøvaag  
University of Agder, Norway  
November 2025

---

## Abstract

Can a machine truly listen? This question demands understanding what listening *is*, accepting that it is contextual, and seeking where the common ground between human listening and machine listening might exist—or even, what that common ground might look like. This exposition documents an approximately 12-month long practice-based development of MusicHal 9000, an AI musical partner built on gesture-based memory architecture. Through the integration of music-optimized neural encoding (MERT-v1-95M providing 768-dimensional perceptual features), semantic style detection (CLAP for automatic behavioral mode selection), and a novel **Dual Vocabulary** architecture that separates harmonic and rhythmic perception, the system has learned to respond with musical intent rather than algorithmic randomness.

The research contributes a gesture consolidation architecture that extends IRCAM's Musical Agents framework (Bujard et al., 2025), a music-specific encoding approach using MERT, a semantic behavioral layer using CLAP, and a parallel **RhythmOracle** system for tempo-independent rhythmic pattern learning. This exposition demonstrates how trust in human-AI musical partnership emerges not from perfection, but from transparency—understanding *why* the machine makes its choices—and from identifying a common ground between two fundamentally different cognitive substrates: 768-dimensional neural space and The Brain.

**Keywords**: AI musical partnership, gesture-based memory, MERT, CLAP, RhythmOracle, AudioOracle, explainable AI, practice-based research, improvisation, trust, dual perception, drummer cognition.

---

## Table of Contents

**PART 1: THE LISTENING QUESTION**
1. Opening: Can a Machine Truly Listen?
2. Context: Two Substrates, One Partnership
   - 2.1 The Human Substrate: The Brain (and the Drums)
   - 2.2 The Machine Substrate: 768D Neural Space
   - 2.3 The Translation Layer: Gesture Tokens as Common Ground

**PART 2: THE CRISIS & THE FIX**
3. The Crisis: When The Machine Stopped Listening
   - 3.1 The Symptom: Loss of Musical Intent
   - 3.2 The Root Cause: Feature Collapse
4. The Fix: Musical Gesture Consolidation
   - 4.1 What is a Musical Gesture?
   - 4.2 The Consolidation Process

**PART 3: ARCHITECTURAL INNOVATIONS**
5. Dual Vocabulary: Harmonic vs. Percussive Listening
   - 5.1 The Problem: Blurred Perception
   - 5.2 The Solution: Separating Sources (MERT + RhythmOracle)
6. Transparency Framework: Trust Through Explainability
   - 6.1 The Trust Problem
   - 6.2 Three Layers of Transparency

**PART 4: MEMORY & INTELLIGENCE**
7. Musical Memory: The Dual Oracle System
   - 7.1 AudioOracle (Harmonic Memory)
   - 7.2 RhythmOracle (Rhythmic Memory)
8. Behavioral Intelligence: Personality Through Time
   - 8.1 Sticky Modes & Performance Arcs

**PART 5: METHODOLOGY & TIMELINE**
9. Practice-Based Methodology
10. Timeline of Creation (2025)
11. Reflection & Closing

---

# PART 1: THE LISTENING QUESTION

## 1. Opening: Can a Machine Truly Listen?

Can a machine truly listen? This question demands understanding what listening *is*, accepting that it is contextual, and seeking where the common ground between human listening and machine listening might exist—or even, what that common ground might look like.

From a human standpoint, listening clearly isn't just about detecting pitches or classifying chords. On top of that basic understanding, and not additionally, but *simultaneously*, we hear (and feel) musical gestures as intentional pieces of information: instructional, relational, confrontational, accompanimental. The way a guitar stroke unfolds over time, or a drum pattern creates momentum, are all part of this. These gestures carry meaning beyond their acoustic properties—they communicate intent, energy, and musical direction. There are more layers to this too, of course, even more complex ones; a certain musical phrase, for example, can shift an entire body from one emotional state to another, but figuring out what part this plays is better left for another project. I'm just mentioning it here to make the reader of this text aware that even if I'm trying to explain what my musical understanding is based on, I am leaving out huge fundamental parts of what actually goes on in the transaction that goes on when a person plays an instrument and another person listens to it.  

This research began with a simple observation: computational musical partners felt like randomizers, not collaborators. Because everything they responded with ultimately came from a clever generative script—input-based or not—the essence of **surprise** and **trust** as listening traits extending from human to machine (albeit in an abstract, extrapolated sense) became critical factors. In an improvisational setting, for the purpose of creating music, pure generative scripts *had* to be replaced with scripts that could show a sense of "understanding," which then was the trigger that led the way into searching for what I would refer to as **common ground**. After all, if two musical systems are listening to the same audio clip without a shared understanding of what that clip actually consists of, how can any side in that transaction even begin to trust that the responses given are anything but completely random?

[VIDEO: Opening performance - You on drums with MusicHal responding, showing the loop: you play → listen → machine responds → you listen → adapt]

### 1.1 What Does Common Ground Mean, then?

In my opinion, a machine will never think like a human, nor will humans think like machines. The term "common ground" therefore means finding functional **translation layers** between fundamentally different cognitive architectures.

Computers must be allowed to understand the way they do best: currently, **MERT-v1-95M's 768-dimensional neural encoding** of audio as perceptual features (Li et al., 2023) provides music-optimized understanding. This is not symbolic music theory like the system humans use (chord names, scales, quantized rhythms)—it's pure pattern space, capturing timbre, attack, spectral shape, and harmonic content in 768 numbers per audio frame. MERT is specifically pre-trained on 160,000 hours of music, unlike general-purpose audio models like Wav2Vec 2.0 (Baevski et al., 2020) which was designed for speech.

Humans, on the other hand, understand everything through **The Brain** (duh..)—and how that works, well, who really knows. Possibly, we chunk sound into gestures, recognize patterns, predict continuations, and respond with embodied knowledge accumulated through years of practice. For drummers, this means *feeling* rhythmic momentum in the body as neurological transmitters fires in the brain, making my muscles twitch a little whenever I hear ghost notes and accents, thus making me *feel* them as meaningful variations (Brean & Skeie, 2019, kap. 1).

For drummers, a drum roll is not just interpreted as rapid single strokes, it is much more commonly heard as *one sustained strike*, which, of course, can be said about fills and builds, or any kind of rudiment above a certain tempo. In any case, this specific knowledge plays a crucial role in my musical understanding, and is yet another layer of translation that I can use when I listen and understand what is essentially waves moving through a latent space through the medium of air. This layer, and everything connected to it, is not accessible to a machine.

Between these two substrates I have mentioned, everything can be seen as **translations of ideas**. Since **768D is the basis for a computer**, and **The Brain is the basis for a human**, the common ground has to be: 

1.  The knowledge that we are operating from a point of observation best suited for the capabilities we have, and 
2.  That every response made is initially coming from an abstract feedback loop where music is not music, but rather a state of mind that allows us, through usage of various tools available to us, to respond and interact using sound only.  

[IMAGE: The translation layer diagram - Left: 768D neural space → Gesture tokens → AudioOracle memory; Right: The Brain → Musical gestures → Phrase memory; Center: Common Ground = learned patterns in shared performance space]

### 1.2 Technical Competence as Research Tool

As I am a performing musician—drummer first, other instruments second—**technical competence** plays a huge role for me. I am at a technical and musical level where I can adapt to a lot of situations, so ultimately the sentence *"What I play will make sense, if I feel and say it makes sense"*, holds true because I know enough about what my instrument to back up that claim with artistic practice.

The performative musical competence I have is also methodologically essential to this research. The architectural decisions documented in this exposition emerged from musical needs encountered during practice, not from abstract technical optimization. When token diversity collapsed (see Part 2), I felt it as musical incoherence before I understood it as a software bug. When behavioral modes flickered, I experienced it as "playing with a randomised partner" before identifying the timer problem in the code.

**The Goal: Friction, Not Perfection**

It is crucial to understand that I am not looking for a "perfect" musical partner. I am not trying to build a digital clone of a virtuoso who never makes a mistake. I am looking for an **improvising agent**.

In improvisation, "perfection" is often boring. What makes a musical conversation interesting is **friction**—the need to solve a musical problem in real-time. If my partner plays exactly what I expect, I am comfortable, but I am not *moved*. I am not challenged to think differently.

The goal of MusicHal 9000 is to be a source of inspiration, a partner that I can play with, argue with, and explore new ideas with. I want an agent that has the ability to **affect me** while I play. Whether its internal pipeline is "broken" or "accurate" is secondary to its ability to create variation, offer a genuine response, and establish a common ground where we can surprise each other. I want a machine that forces me to be a better improviser by giving me something real to react to.

### 1.3 A Note on the Media: Documentation of Failure

Because this is artistic research, the media examples included in this exposition are **not** polished "product demos." They are largely examples of **failure**—moments where the system broke, misunderstood, or collapsed. 

In this project, failure is not a dead end; it is the primary data source. It is in the moments of breakdown—where the machine plays a ballad chord over a drum solo, or gets stuck in a loop—that the friction becomes visible and the research questions are most clearly articulated. These recordings document the struggle to find common ground, rather than a triumphant arrival at it.

---

## 2. Context: Two Substrates, One Partnership

To begin with, and long into the process, finding the **translation layers** between fundamentally different cognitive architectures was challenging. This section explores both substrates (human and machine) and the gesture token system that bridges them.

### 2.1 The Human Substrate: The Brain (and the Drums)

As mentioned many times, I'm a musician first, designer second, programmer .. far down on the list. Not "computer scientist"—but a performer who codes aided by useful tools, and is still learning to do so properly. This distinction matters because the system's architecture emerged from musical needs, not accurate technical requirements.

Going back to "What I play makes sense if I say it makes sense," I would like to note, now, that this isn't said with arrogance—it's just showing off a little, asserting an **authority that have come through practice**. After years of drumming, I've developed an internal model of what constitutes musical gesture, rhythmic momentum, and conversational dynamics. This model guides both my playing and my evaluation of MusicHal's responses.

**Human listening is multi-dimensional:**
We don't hear pitches and then separately hear rhythms and then separately hear timbres. We hear **musical gestures**—integrated perceptual events that combine all these dimensions simultaneously. A crash cymbal accent is not just high frequency spectral energy; it's a **gesture**: sometimes an emphatic punctuation mark, other times an ending signal.

### 2.2 The Machine Substrate: 768D Musical Neural Space

Where humans have The Brain, MusicHal has **MERT-v1-95M**—a music-optimized pre-trained neural network that converts audio waveforms into 768-dimensional feature vectors (Li et al., 2023). Additionally, **CLAP** (Contrastive Language-Audio Pretraining) provides semantic understanding, mapping musical styles to behavioral modes.

**What is MERT-v1-95M?**
MERT (Music Audio Representation Transformer) is a self-supervised learning model specifically designed for music understanding. Unlike Wav2Vec 2.0 (Baevski et al., 2020)—originally designed for speech recognition—MERT learns musical concepts: chord progressions, rhythmic patterns, genre characteristics, and harmonic function. It lives inside the `Chandra_trainer.py` script, and provides a method of understanding how audio should be represented by solving a music-specific pre-training task.

---

# PART 2: THE CRISIS & THE FIX

## 3. The Crisis: When The Machine Stopped Listening

In August 2025, a critical issue emerged that threatened to derail the entire project. I call it the **Feature Collapse**.

For months, the system had been growing more sophisticated. It had learned to parse complex harmonies and recognize timbral shifts. But as I sat down behind the drum kit to test the latest iteration, something felt wrong. The system was technically "listening"—the input meters were dancing, the CPU was crunching numbers—but it wasn't *hearing* me.

I would play a chaotic, high-energy drum fill, expecting the system to respond with a burst of activity or a sharp rhythmic counterpoint. Instead, it would drift along with the same tentative, ambient piano chords it had been playing during the silence. It was as if I hadn't played at all. The "common ground" we had built was gone. The machine had become an autistic listener, locked in its own world of harmonic analysis, utterly deaf to the physical reality of rhythmic impact.

### 3.1 The Root Cause: A Harmonic Bias

The problem, I realized, was in the translation layer. The MERT neural network, brilliant as it was at understanding the "notes" of music, had a blind spot. It was so focused on pitch and harmony that it treated all percussive sounds as "noise." To MERT, a snare drum hit and a kick drum thud were just "non-harmonic events," grouped together in a single, undifferentiated category.

By optimizing for harmonic understanding, I had inadvertently lobotomized the system's rhythmic intelligence. It was trying to understand a drum solo using a vocabulary meant for a string quartet.

## 4. The Fix: Musical Gesture Consolidation

The solution wasn't to fix the code, but to fix the philosophy. I had to teach the machine to listen like a drummer.

When I play a drum roll, I don't think "hit-hit-hit-hit-hit." I think "roll." It is a single **musical gesture**, a unified event with a start, a duration, and an end. The machine, however, was analyzing every millisecond as a separate event, getting lost in the microscopic details and missing the macroscopic meaning.

I introduced **Gesture Consolidation**. This process forces the machine to stop obsessing over every sample and instead look for the "shape" of the sound. It groups fleeting transients into coherent gestures, bounded by silence or significant changes in timbre. Suddenly, the machine stopped hearing "noise-noise-noise" and started hearing "phrase." It was the digital equivalent of teaching a child to read words instead of just seeing individual letters.

---

# PART 3: ARCHITECTURAL INNOVATIONS

## 5. Dual Vocabulary: Two Ears are Better Than One

Fixing the gesture recognition was only half the battle. The machine still needed a way to understand those gestures. The breakthrough came with the realization that a single "brain" couldn't handle both the harmonic and rhythmic complexity of our improvisation.

I implemented a **Dual Vocabulary** architecture, effectively splitting the system's perception into two distinct streams:

1.  **The Harmonic Stream (MERT)**: This is the "sophisticated music student." It listens for spectral consistency and timbral colors. It ignores the sharp attacks of the drums to focus on the "notes" and the harmonic progression.
2.  **The Percussive Stream (RhythmOracle)**: This is the "drummer." It ignores the pitch entirely. It doesn't care if I'm playing a C major or an F sharp; it cares about *when* I play, how dense the pattern is, and where the accents fall.

This separation was transformative. It is important to note that MusicHal does not "hear" a C Major chord in the way a human does. It has no concept of music theory. It hears a **vector**—a complex mathematical signature that happens to correlate with what we call "C Major."

With the Dual Vocabulary, it can now differentiate between the *vector of a harmony* and the *pattern of a rhythm*. It can say: *"I detect a harmonic vector similar to state 42 (Harmonic Stream) played with a density pattern similar to state 105 (Percussive Stream)."* It allows the system to understand the *tension* between a calm harmony and a frantic rhythm, allowing for far more nuanced and meaningful responses.

## 6. Transparency Framework: Trust Through Explainability

Trust in a musical partnership is fragile. If a human bass player plays a wrong note, I can look at their face and see if it was a mistake or a bold choice. With an AI, that visual feedback is missing.

### 6.1 The "Black Box" Problem
Before this phase, MusicHal was a black box. If it played a dissonant chord, I had no way of knowing if it was a "Contrast" choice or a bug. This uncertainty killed the musical flow. I found myself playing tentatively, afraid to push the system because I didn't trust it to catch me.

### 6.2 Opening the Hood
To build trust, I had to make the machine's thinking visible. I built a **Transparency Framework** that acts as a window into the AI's mind.

*   **The Decision Explainer**: A real-time log that prints the system's internal monologue. *"Input was dense and loud -> Mode is 'Contrast' -> Choosing sparse, quiet response."* Seeing this logic unfold in real-time changed everything. It wasn't just debugging data; it was a conversation.
*   **The Visual Dashboard**: A simple display showing the system's current "State of Mind." Seeing "Mode: Imitate" or "Listening: Intense" gave me the confidence to lean into the interaction, knowing the system was with me.
*   **Behavioral Consistency**: We implemented "Sticky Modes." If the machine decides to be aggressive, it *stays* aggressive for a while (30-90 seconds). It doesn't flip-flop with every new note. This consistency creates a sense of personality, a "character" that I can interact with, rather than a jittery algorithm.

[AUDIO: Before/after trust comparison - Before: randomized responses with no gestural coherence; After: musical intent with phrases and behavioral personality]

---

# PART 4: MEMORY & INTELLIGENCE

## 7. Musical Memory: The Dual Oracle System

Memory is what turns a sound into music. Without memory, every note is a surprise, and there is no structure. MusicHal's memory is built on the **AudioOracle** (Dubnov et al., 2007), but we've adapted it to be a living history of our performance.

### 7.1 AudioOracle (Harmonic Memory)
Think of this as the system's "long-term memory" for harmony. It builds a map of every chord and melody it has ever heard. When I play a chord, it doesn't just react; it remembers. It looks at its map, finds similar moments from our past performances, and uses that knowledge to predict what might come next. It's not generating random notes; it's *recalling* our shared musical history.

### 7.2 RhythmOracle (Rhythmic Memory)
This is the new addition for 2025, the "muscle memory" of the system. Running in parallel, it focuses solely on rhythm. Crucially, it is **tempo-independent**. It stores patterns of "hits" and "rests" as relative relationships. This means if I play a slow, funky groove, the RhythmOracle recognizes the *pattern*. If I later play that same pattern at double speed, it still recognizes it. This allows the system to follow me through tempo changes and metric modulations without losing the thread of the groove.

## 8. Behavioral Intelligence: Personality Through Time

A musical partner needs more than just good ears; it needs a personality. A partner that changes its mind every second is annoying. A partner that has "moods" is interesting.

### 8.1 The Narrative Arc
MusicHal uses **Sticky Modes** to create a narrative structure over time. It doesn't just react moment-to-moment; it builds a performance.
*   **The Opening**: It might start in **Imitate Mode**, tentatively shadowing my playing, establishing a connection.
*   **The Development**: As the energy builds, it shifts to **Contrast Mode**, challenging me with opposing rhythms or dissonant harmonies, pushing the music into new territory.
*   **The Resolution**: Finally, it might settle into **Couple Mode**, locking in with my groove for a unified, synchronized conclusion.

This "Performance Arc" turns a jam session into a composition. It gives the improvisation a beginning, a middle, and an end.

---

# PART 5: METHODOLOGY & TIMELINE

## 9. Practice-Based Methodology: Coding on the Bandstand

This system wasn't built in a sterile lab; it was forged in the rehearsal room. The methodology was deeply **iterative** and **practice-based**.

I didn't just write code and run tests. I wrote code, picked up my drumsticks, and *played*. I jammed with the system for hours, feeling where it worked and where it failed.
*   Does it feel musical?
*   Does it surprise me in a good way?
*   Does it feel like a broken toy?

When the system felt "off," I didn't just look at the error logs. I looked at the musical interaction. I adjusted the code based on the *feeling* of the performance, not just the data. This cycle of **Code -> Play -> Feel -> Refine** was the heartbeat of the project.

## 10. Timeline of Creation (2025)

*   **January - March 2025**: *The Awakening*. Initial experiments with MERT. The excitement of the new neural hearing, followed by the slow realization of the "Rhythm Deafness" problem.
*   **April - June 2025**: *The Split*. Developing the **Dual Vocabulary** concept. Building the RhythmOracle from scratch to give the machine a sense of time.
*   **July - August 2025**: *The Crisis*. The "Feature Collapse." A frustrating period of debugging where the system seemed to lose its mind. The breakthrough of Gesture Consolidation.
*   **September 2025**: *The Understanding*. Implementation of **CLAP** for style detection. The system starts to "understand" genre, distinguishing between a ballad and a banger.
*   **October 2025**: *The Connection*. Final integration of all systems. The "Trust" breakthrough—the moment when the system finally felt like a partner, not a tool.
*   **November 2025**: *The Exposition*. Documenting the journey.

## 11. Reflection & Closing

MusicHal 9000 is no longer just a piece of software. It is a musical entity with a specific way of listening—a "machine listening" that has found common ground with my human listening. It doesn't hear what I hear, but it hears *intent*. And because I can see *why* it hears what it hears, I can trust it.

We have travelled without being moved, but in the end, I have been moved by the journey.

---

**References**
*   Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*.
*   Bujard, T., et al. (2025). *Musical Agents: A framework for interactive musical systems*. IRCAM.
*   Dubnov, S., et al. (2007). *Audio Oracle: A new algorithm for fast learning of audio structures*.
*   Li, Y., et al. (2023). *MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training*.
