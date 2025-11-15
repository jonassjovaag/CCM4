# MusicHal: Building a Musical Partner I Can Trust

**An Artistic Research Documentation**  
by Jonas Howden Sjøvaag  
University of Agder  
2025

---

## Part I: The Problem I'm Trying to Solve

### 1. Why I Built This

I don't trust MusicHal yet. That's the problem I need to solve.

When it plays back at me, I can't tell if it's actually listening to what I'm doing or just spitting out random patterns it learned from some jazz recording. And if I can't trust it's listening, then I can't really play with it. Simple, really.

Playing with another human musician, you know they're listening. You play a phrase, they respond—maybe they imitate it, maybe they contrast it, maybe they develop it into something else. There's a conversation happening. With a machine, that certainty disappears. Is it responding to me, or is it just... running its algorithm?

This matters more than you might think. When I play with a human, I'm in a state of semi-automated consciousness. Some choices are deliberate—I think "I'll play a lydian scale here" or "I'll create rhythmic tension against the pulse." But most of it is intuitive, reactive, flowing. That flow state only happens when I trust my musical partner. When I'm second-guessing whether they're listening, the flow breaks.

So here's the challenge: create a machine that I can trust enough to enter that flow state with.

Not too predictable—if it just plays back exactly what I played, that's boring, like practicing with a mirror. Not too random—if it generates alien, incoherent responses, that's frustrating, like trying to have a conversation with someone speaking a different language. Somewhere in between: responsive enough that I feel heard, surprising enough that I stay interested.

Human musicians achieve this through years of training, cultural musical knowledge, and the ability to seamlessly blend conscious choices with subconscious reactions. They remember what you played five minutes ago and bring it back varied. They develop consistent personalities—one bass player might always walk lines, another might play sparse punctuations. They stay in a musical mood long enough that you can respond to it, develop a dialogue within it.

Can a machine do this?

#### The Chinese Room in My Studio

Here's the philosophical problem I can't avoid: John Searle's Chinese Room Argument. The setup goes like this—imagine someone who doesn't understand Chinese, locked in a room with a rulebook. Chinese text comes in through a slot, they follow the rulebook to manipulate symbols, and send Chinese text back out. From the outside, it looks like the room understands Chinese. From the inside, it's just symbol manipulation.

Is MusicHal any different?

It receives audio input, extracts features (768-dimensional vectors from Wav2Vec, rhythm ratios from Brandtsegg's analyzer), looks up patterns in memory (AudioOracle with Factor Oracle), applies some probability distributions (request masking with temperature scaling), and outputs MIDI notes. From the outside, it might sound like it's listening and responding musically. From the inside, it's just... math.

Does that matter?

I lean toward: yes, it's simulation, not "real" listening or understanding. But the simulation might be sophisticated enough to be musically useful. The question isn't whether MusicHal has consciousness—it doesn't—but whether the simulation is good enough that I can enter a flow state while playing with it. Whether I can trust the mechanism enough to stop thinking about the mechanism.

That's what this whole project is about: building that trust through transparency, coherence, and personality.

#### What I Want

I want a musical partner that:

1. **Listens perceptually**, the way I listen—not by extracting symbolic chord names, but by perceiving the spectral, timbral, and rhythmic qualities of sound
2. **Remembers and develops ideas** over time—not just responding to the immediate moment, but recalling motifs from earlier and bringing them back varied
3. **Has consistent personality modes** that persist long enough to establish character—shadowing me closely for 60 seconds, then switching to independent counterpoint for another 60 seconds
4. **Shows me its reasoning** so I can verify it's actually using my input, not just generating random patterns

If I can build that, maybe I can trust it. And if I can trust it, maybe I can play with it.

#### Personal Context

I'm a drummer mainly, also play piano and sing. I've been playing for years, with other humans or solo. I know what meaningful musical dialogue feels like. It's not about perfectly coordinated responses—sometimes the best moments come from conflict, from misunderstanding that gets resolved, from surprising divergences. But there's always a sense that we're in it together, listening to each other, building something.

I don't want MusicHal to be a clone of John Coltrane. I don't want it to be indistinguishable from a human improviser. I want it to be a partner with its own character, its own voice, that emerges from the specific recordings I train it on and the algorithms I use to shape its behavior. Like any musician develops their own personality through their individual journey, MusicHal should develop a localized musical identity through its training data and structural constraints.

The naming helps me think about this. MusicHal_9000 after HAL 9000 from 2001: A Space Odyssey—an AI that was supposed to be a partner but became... complicated. Chandra_trainer after Dr. Chandra, HAL's first instructor who taught him to sing "Daisy." Training is teaching. Performance is partnership. The names remind me that this is about relationship, not just technology.

Jazz improvisation is my test case, not because it's "the hardest" (it's my practice, what I do), but because its challenge creates the conditions for dialogue. Real-time listening and response, implicit harmonic rules with freedom within them, turn-taking, the expectation of coherence over time—these constraints force a dialogue space into existence. If MusicHal can engage in this kind of structured but open interaction—responding to melodic gestures, maintaining harmonic coherence, developing rhythmic patterns, recalling and varying themes—then the dialogue mechanism is working. I know jazz well enough to tell when something is actually dialoguing versus just making jazz-adjacent sounds. That's why I'm using it.

#### Technical Preview (We'll Get There)

Building this required solving several problems:

- **Machine listening**: Wav2Vec 2.0 for perceptual feature extraction (Bujard et al., 2024), Brandtsegg/Formo rhythm ratio analysis (Brandtsegg & Formo, 2024), Barlow harmonicity calculations
- **Memory systems**: short-term (180-second buffer), pattern learning (AudioOracle; Dubnov et al., 2007), long-term thematic (phrase memory with variations)
- **Conditional generation**: request masking with multi-parameter constraints
- **Behavioral consistency**: sticky modes that persist 30-90 seconds (based on Thelle & Wærstad, 2023)
- **Decision transparency**: real-time logging of why it plays what it plays

The development was also informed by Hans Martin Austestad's work on real-time tuning and intonation analysis (Austestad, 2025), which influenced my thinking about perceptual systems.

But before I dive into the technical details, let me tell you why existing systems weren't enough.

---

### 2. What Others Have Built (and Why It's Not Enough)

I didn't build MusicHal in a vacuum. There's a rich history of computer music systems designed for improvisation and musical interaction. I've studied them, used some of them, learned from all of them. But none of them solved my specific problem: trust through transparency, coherence through memory, identity through personality.

Here's what already exists and why I needed something different.

#### IRCAM's OMax: The Linear Continuation Machine

OMax, developed at IRCAM by Gérard Assayag and colleagues, uses the Factor Oracle algorithm to learn musical patterns and generate continuations. It's the foundation of most modern musical AI improvisation systems, including mine.

**What they got right:**

The Factor Oracle algorithm is brilliant (Assayag & Dubnov, 2004). It incrementally learns sequences in O(m) time and space, building an automaton that captures pattern structure through suffix links. When you want to generate, you traverse the automaton, jumping through suffix links to similar past patterns. Elegant, efficient, musically coherent.

OMax can learn from live input in real-time, building its memory as you play. This is powerful—it's not pre-trained on fixed datasets, it's continuously adapting.

I use Factor Oracle in MusicHal, extended to polyphonic audio through AudioOracle (Dubnov et al., 2007).

**What's missing (for me):**

OMax does linear continuation. You feed it patterns, it spits out the next pattern based on statistical likelihood. It's following you, but it's not really *dialoguing* with you. There's no sense of personality, no mode where it challenges you or stays independent.

No phrase memory. OMax responds to the immediate context but doesn't remember and develop themes over longer timescales. It can't recall a motif from 60 seconds ago and bring it back transposed.

No decision transparency. When OMax generates something, you don't know *why* it chose that pattern. Was it responding to your melody? Your rhythm? Pure chance? You can't see inside.

#### George Lewis's Voyager: The Overwhelming Analyzer

Voyager, created by composer and trombonist George Lewis, is one of the classics (Lewis, 2000). Built in the 1980s and evolved over decades, it analyzes MIDI input across 64 parallel voices, extracting parameters like pitch, rhythm, register, density, and dynamics. It uses these analyses to make musical decisions in real-time.

**What they got right:**

The idea of parallel voice analysis is sophisticated. Voyager doesn't just look at melodic contour or harmonic progression—it tracks multiple dimensions simultaneously and can respond to any of them.

The system has a strong compositional voice. When you hear Voyager, you know it's Voyager. That's because Lewis programmed it with specific musical preferences and tendencies drawn from his own compositional practice.

**What's missing (for me):**

64 voices is overwhelming. The complexity makes it hard to understand what Voyager is responding to at any given moment. It's doing too much analysis, creating too many decision pathways.

Symbolic MIDI analysis only. Voyager doesn't hear timbre, spectral content, or the subsymbolic perceptual qualities of sound. It's analyzing discrete symbolic events, not continuous audio perception.

No transparency. Voyager's decision-making is opaque. You can't see which of the 64 analytical voices influenced a particular musical choice.

#### IRCAM's ImproteK: Real-Time Learning Without Memory

ImproteK is IRCAM's newer system, designed for live improvisation with real-time learning. It's similar to OMax but with updated algorithms and better performance.

**What they got right:**

Real-time learning is appealing. The system adapts to you as you play, building its vocabulary from your musical input.

Better integration of rhythmic and harmonic analysis than original OMax.

**What's missing (for me):**

Still no long-term phrase memory or thematic development.

No decision transparency—you still can't see why it plays what it plays.

Pre-trained models vs. live learning is a design choice (Thelle, 2022). ImproteK learns live, which means its musical knowledge resets each session. I chose pre-trained models (training on recorded audio, then using that memory in performance) because I want MusicHal to have a consistent learned vocabulary, a stable musical identity built from specific recordings.

#### François Pachet's Continuator: Pure Statistical Following

The Continuator, developed by François Pachet at Sony CSL, learns musical style through Markov models and generates continuations that match the statistical properties of the input.

**What they got right:**

Style matching. The Continuator can learn the statistical patterns of a musical style and generate material that sounds stylistically similar.

Simple, transparent algorithm (Markov chains). You can understand what it's doing.

**What's missing (for me):**

Pure statistical continuation, no conditional generation. You can't request "play something consonant" or "match this rhythmic ratio." It just generates based on learned transition probabilities.

No memory or thematic development beyond the Markov order (typically 2-4 events).

No behavioral modes or personality variation.

#### What I Needed That Wasn't There

Looking at these systems, I saw gaps:

1. **Trust through transparency**: None of them show you *why* they make specific musical choices. I need to see the reasoning to trust it's actually listening.

2. **Long-term phrase memory**: They respond to immediate context but don't remember and develop motifs over longer timescales (30-60 seconds). Real improvisers do this constantly.

3. **Sticky personality modes**: Behavioral modes that persist long enough (30-90 seconds) to establish character, not rapid flickering between states.

4. **Dual perception**: Combining subsymbolic perceptual features (Wav2Vec) with symbolic/mathematical analysis (Brandtsegg ratios). Most systems do one or the other, not both.

5. **Conditional generation with multi-parameter requests**: Being able to say "generate something that matches this gesture token AND has this consonance level AND fits this rhythm ratio," not just linear continuation.

These gaps became my design goals.

I'm building on giants' shoulders here—Factor Oracle from IRCAM, the idea of behavioral modes from Lewis, the commitment to practice-based research from the whole computer music field. But I'm combining and extending these ideas in specific ways to solve my specific problem: building a musical partner I can trust.

Let me explain how I approached this.

---

### 3. How I Approached This (Research Methodology)

This is practice-based artistic research. That means I build things, I play with them, I notice what works and what doesn't, I revise. The test is always: do I want to keep playing? If I stop after 2 minutes because it's frustrating or boring, the system failed. If I play for 15 minutes and lose track of time, it's working.

No controlled experiments comparing condition A to condition B. No user studies with N participants rating musical quality on Likert scales. Just me, in my studio, building and playing, iterating based on musical intuition and technical analysis.

Some people might say that's not rigorous enough. I'd say it's the only kind of rigor that matters for this question. Musical partnership is subjective, personal, situated. Whether MusicHal works for me in my practice is what I'm investigating. Whether it works for others is a different research question.

#### The Iterative Process

Many failed attempts before this version. Let me be honest about that.

Early versions extracted symbolic chord names from audio and tried to learn "Cmaj7 follows Dm7" progressions. It sounded like a jazz textbook, not like jazz. Harmonically correct but rhythmically dead, no sense of gesture or forward motion.

I tried pure AudioOracle learning without any perceptual pre-processing. It learned something, but when it generated, the output was musically incoherent—random jumps, no phrase structure.

I tried behavioral modes that switched every 2-5 seconds. Too fast. You'd feel a shift starting to happen and then it would switch again before establishing character.

I tried uniform probability fallback when requests didn't match. That broke trust completely—it would play something responsive, then suddenly generate something random, breaking the musical dialogue.

The breakthrough moments:

**Temporal smoothing**: When I realized the system was over-sampling sustained notes and creating false rapid chord progressions (C, Cdim, C, Cdim...), I built a temporal smoothing algorithm that groups events within time windows and applies onset-aware averaging. Single fix, transformed the musical output from manic to coherent.

**Wav2Vec for perceptual features**: When I stopped trying to extract symbolic chord names and instead used Wav2Vec's 768-dimensional perceptual feature space as the machine's "hearing," everything clicked. The machine's vocabulary became gesture tokens (0-255 quantized from feature space), and it learned relationships between gestures, not between chord symbols.

**Phrase memory**: When I added a system that extracts motifs (3-5 note patterns) and recalls them 30-60 seconds later with variations (transpose, invert, retrograde), the performances suddenly had thematic coherence. Before this, MusicHal was just responding moment-to-moment. After this, it felt like it had intent.

**Decision transparency**: When I built the logging system that shows *why* each musical choice was made (triggered by MIDI X, context parameters Y, request Z, pattern match score W), I could finally verify it was listening. Seeing "87% match to gesture_token 142 from training bar 23" confirmed the system was doing what I designed it to do.

Each of these took weeks of iteration. Build, test, notice the problem, analyze, revise, test again.

#### Training Data: My Own Recordings

I'm training MusicHal on recordings of me playing. Not downloaded datasets, not standard jazz from the Real Book, but specific performances I recorded:

- **Itzama.wav** (183.7 seconds): Pop track leaning heavily towards electronically based beat music. Very modern, not jazz. Clear 6-phase performance arc—intro, development, climax, resolution, second climax, final resolution. Complex rhythmic structure, electronic feel.

- **Georgia.wav**: "Georgia on My Mind" (Hoagy Carmichael and Stuart Gorrell, 1930). Jazz standard ballad. I sing, with Andreas Ulvo on piano. Just voice and piano, no other instruments.

- **Nineteen.wav**: My own composition, jazz standard style ballad. With Andreas Ulvo (piano) and Mats Eilertsen. Focus on melodic phrasing and space.

- **Curious Child.wav**: Lyrical, with Eple Trio, Andreas Ulvo + Sigurd Hole + myself, piano bass drums. Very complex rhythmically, at least when it comes to quantizing it. 

These recordings define MusicHal's musical vocabulary. When it generates, it's recombining patterns it learned from these specific performances. That's intentional—I want MusicHal to have a localized musical identity shaped by specific training data, not to be a universal music generator that knows everything.

#### Live Performance Testing as Evaluation

The real test happens when I run MusicHal_9000.py, set up my microphone and MIDI interface, and start playing.

Evaluation criteria:

1. **Session duration**: How long do I want to keep playing? 2 minutes = failure. 15+ minutes = working.

2. **Flow state**: Do I enter that semi-automated improvisational state where I'm not thinking about the system, just playing? If I'm constantly second-guessing or analyzing, it's not working.

3. **Musical surprise**: Does MusicHal generate anything I wouldn't have thought of myself? Positive surprises = system is doing something interesting. Negative surprises (random, incoherent) = problems.

4. **Thematic coherence**: Can I hear motifs being recalled and developed over time? Or is it just a series of unrelated responses?

5. **Trust**: Do I believe it's listening to me? The decision transparency logs let me verify this objectively, but subjectively, does it *feel* like dialogue?

No ground truth, no correct answers, just musical intuition tested through practice.

#### Honest About Uncertainty

There are things I still don't know:

- Does phrase memory create "real" intentionality or just the illusion of it? The algorithm is deterministic (check timer, select motif, apply variation), but the effect feels intentional. Does the mechanism matter if the musical result works?

- Is decision transparency artistically valuable or distracting? Seeing "pattern match: 87% similarity to training bar 23" builds trust but breaks magic. Should I use it in development only, or is transparency part of the artistic statement?

- Can the system generalize beyond jazz? I've only tested it in that context. Would it work for free improvisation, for electronic music, for other genres?

- How would other musicians experience playing with MusicHal? This whole evaluation is based on my subjective experience. Different players might have different needs, different trust thresholds.

These uncertainties are part of the research. I'm not claiming to have solved musical AI partnership universally. I'm investigating what works for me, in my practice, with my musical values and training.

That's the methodology. Build, play, listen, analyze, revise, repeat. Trust the process, trust the musical intuition, verify through both subjective experience and objective logging.

But before diving into technical details, I need to explain the conceptual frame that shapes how I think about this whole project.

---

### 3.5. Performance Ecosystems - The Conceptual Frame

I came across Simon Waters' work late in the development process, but reading it felt like someone had articulated what I'd been intuiting all along (Waters, 2007). He argues that the traditional distinctions we make—performer, instrument, environment—are artificial. They mask important ambiguities about where one thing ends and another begins.

Think about playing a flute. Waters references the old teaching: think of the flute's tube starting at your diaphragm and extending into the room. Not: you blow into an instrument which produces sound in an environment. Instead: a continuous resonating system where your breath, the tube, the air column, the room acoustics, and your ears hearing it all form one integrated complex. Change any part and the whole system changes.

When I play with MusicHal, where does "my" playing end and "its" response begin? I play a phrase. The microphone captures it (already a transformation—my sound becomes voltage). Wav2Vec processes it (another transformation—voltage becomes 768-dimensional vectors). The AudioOracle queries memory (transformation—vectors become pattern matches). MIDI output gets synthesized by SuperCollider (transformation—patterns become sound again). That sound fills the room, reaches my ears, and I respond to what I hear.

But here's the thing: my response feeds back into the system. What I play next is shaped by what I just heard MusicHal play. Which means MusicHal's next response is shaped by my response to its previous response. It's a feedback loop—a performance ecosystem where cause and effect are circular, not linear.

Waters calls this understanding music as "a dynamical complex of interacting situated embodied behaviours" where the boundaries between performer, instrument, and environment blur into one coupled system. That's what playing with MusicHal feels like when it works.

#### Affordances, Not Determinism

Waters draws on Gibson's concept of affordances (Gibson, 1979, as cited in Waters, 2007): what a system offers for interaction, not what it determines. A chair affords sitting, but also standing on, throwing, burning for warmth. The physical properties constrain but don't fully determine use.

MusicHal's training data affords certain kinds of responses. Trained on Itzama (electronic beat music), it affords different interactions than if trained on Georgia (jazz ballad). The gesture tokens, rhythm ratios, suffix links—these constrain what's possible. But they don't determine what actually emerges in performance.

Two different players with the same trained model would have completely different experiences because the ecosystem is different. Their playing style, their instrument, their room, their ears, their musical intentions—all of this shapes what the system affords in practice. The trained model is just one component of the ecosystem.

This helped me stop thinking about MusicHal as something I need to perfect in isolation. The question isn't "is this model good enough?" The question is "does this ecosystem—me + this model + this room + this moment—produce interactions I value?"

#### Site-Specificity and the Return to Embodied Practice

Waters points out something I hadn't thought about: pre-19th century music was mostly site-specific. Church music written for specific church acoustics. Court music for specific rooms. The idea that music should be portable—that the same piece should work in any hall, on any recording, on any device—that's a relatively recent development.

Electroacoustic music returns to site-specificity, but in time rather than space. MusicHal trained on Itzama is specific to that recording, that moment, that performance. The model captures patterns from October 24, 2025, from my playing on that day, in that recording setup. It's not universal music knowledge—it's a specific crystallized moment.

Each performance session with MusicHal is similarly specific: this room, this microphone placement, this day's latency characteristics, my energy level right now. The system doesn't transport cleanly. That's not a bug—it's returning to music as an embodied, situated practice.

#### Ambiguity as Productive

Is MusicHal a tool, a partner, or an environment? I've been trying to decide, but Waters suggests the ambiguity itself is valuable. Computers in musical performance occupy an "incommensurable" position—they don't fit neatly into existing categories, and forcing them to fit obscures interesting possibilities.

When I think of MusicHal as a tool, I focus on control—tuning parameters, optimizing responses, making it do what I want. When I think of it as a partner, I focus on dialogue—responding to its voice, accepting surprises, developing mutual patterns. When I think of it as environment, I focus on immersion—the whole acoustic/algorithmic space I'm playing within.

All three perspectives are true simultaneously. The ambiguity isn't confusion—it's the actual nature of the system. An ecosystem doesn't have clear inside/outside boundaries. I'm in it, it's in me, we're coupled.

#### Emergence: When Systems Outperform Their Specs

Waters cites Cariani's concept of emergence: systems that outperform the designer's specifications, where behaviors cannot be accounted for solely by the designed outcome (Cariani, 1992, as cited in Waters, 2007).

I designed MusicHal to respond to gesture tokens with request-masked pattern matching using sticky behavioral modes. That's the specification. But sometimes in performance, something emerges that I didn't design and can't fully explain—a musical moment where the interaction produces something neither the algorithm alone nor my playing alone would create. The phrase memory recalls a motif at exactly the right moment, landing in a way that feels intentional even though the timing is semi-random. Or a low pattern match score (<40%) forces the system into unfamiliar territory, and the awkwardness itself becomes musically interesting, creating tension I respond to.

These emergent moments are what I'm after. They can't be guaranteed or reproduced—they arise from the ecosystem functioning as a whole. Building for emergence means not over-constraining, leaving room for the system to surprise me, accepting that some of the most interesting musical results will come from behaviors I didn't explicitly design.

#### Why This Frame Matters

The ecosystem concept gives me a way to think about what I'm building that's more honest than "AI music partner" or "interactive system." I'm not building an autonomous agent with human-like understanding. I'm building one component of a performance ecosystem that includes my playing, my listening, the training data's learned patterns, the room's acoustics, the moment's contingencies.

When MusicHal works, it's not because the algorithm is "smart enough" or the training data is "good enough." It's because the whole ecosystem is functioning—all the components coupling together, feedback loops operating, emergent behaviors arising from the interaction.

When it doesn't work, it's rarely one broken component. It's the ecosystem not coupling: latency too high so feedback is delayed, training patterns too distant from my current playing so pattern matches fail, my energy low so I'm not responding fluidly. Everything's connected.

This frame also helps me think about the trust question. I'm not trying to trust MusicHal as a separate entity. I'm trying to trust the ecosystem—to trust that when I engage with it fully, something musically valuable will emerge. Decision transparency (Section 12) helps me verify the ecosystem is coupling, not just running in parallel.

**[Audio Example 1: Recording showing feedback loop in action - my phrase, MusicHal's response, my adjusted response over 60 seconds]**

With that frame in mind, let me show you how the machine actually listens.

---

## Part II: How The Machine Listens

### 4. Dual Perception - The Key Insight

The machine should NOT think in chord names. That was a breakthrough.

When I was trying to extract "Cmaj7" from audio and teach the AI to learn "Cmaj7 follows Dm7," I was forcing human symbolic thinking onto machine perception. Wrong approach.

Here's what I realized: Wav2Vec 2.0, a neural network trained on massive amounts of audio, learns to perceive sound in a 768-dimensional feature space (Bujard et al., 2024). It's not extracting chord names or note names—it's learning perceptual patterns, spectral qualities, timbral relationships. That's closer to how I actually hear music than symbolic analysis.

So I stopped fighting it. The machine's vocabulary became **gesture tokens** (0-255, quantized from Wav2Vec's feature space). When MusicHal learns, it learns "token 142 tends to follow token 87." When it generates, it queries the AudioOracle for "what tokens follow token 142 in the learned memory?"

Chord names like "Cmaj7" are only for me, to understand what's happening. The machine never sees them in its decision-making. This is a single source of truth—the machine and I are both perceiving the same audio data, just at different levels of abstraction.

#### But Rhythm Needs Numbers

Wav2Vec captures pitch and timbre well, but rhythm is trickier. Onset timing, duration patterns, rhythmic complexity—these are better analyzed mathematically.

Enter Øyvind Brandtsegg and Tony Formo's rhythm ratio analyzer (Brandtsegg & Formo, 2024). It takes inter-onset intervals (time between notes) and expresses them as rational number ratios. Instead of "this note was 0.347 seconds after the previous one," it says "this note was in a 3:2 ratio to the pulse."

Why ratios? Because humans perceive rhythm relationally, not absolutely. A quarter note isn't "0.5 seconds"—it's "twice as long as the eighth note." Brandtsegg's analyzer finds the simplest rational approximation of timing relationships, then calculates **Barlow indigestability**—a measure of rhythmic complexity where simple ratios (1:1, 1:2) are more consonant than complex ones (7:13).

This gives MusicHal a second perception stream: rhythmic structure expressed as mathematical relationships.

#### The Dual Architecture

Here's how audio flows through the system:

```
My instrument → Microphone → Audio buffer (1024 samples)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
              Wav2Vec 2.0                  Brandtsegg Analyzer
           (perceptual features)          (rhythm ratios + timing)
                    │                               │
                    ▼                               ▼
              768D feature vector              [ratio: 3:2,
                    │                          complexity: 2.7,
                    ▼                          deviation: -12ms]
              Quantization                           │
              (k-means, 256 clusters)               │
                    │                               │
                    ▼                               │
              gesture_token: 142                    │
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    UNIFIED MUSICAL PERCEPTION
                    
                    event = {
                      'gesture_token': 142,
                      'rhythm_ratio': [3, 2],
                      'barlow_complexity': 2.7,
                      'consonance': 0.73,
                      'deviation_polarity': -1,  # early
                      'midi': 65,  # F4
                      'timestamp': 45.23
                    }
```

This unified event is what gets stored in memory, what the AudioOracle learns from, what request masking operates on.

This dual perception runs in both training (offline analysis of audio files) and live performance (real-time, with onset buffering and re-analysis every 2-3 onsets for the Brandtsegg analyzer). Same perceptual pipeline, different timing constraints.

**The code:**  
`listener/hybrid_perception.py` - coordinates both analyzers  
`rhythmic_engine/ratio_analyzer.py` - Brandtsegg implementation  
`hybrid_training/wav2vec_chord_classifier.py` - neural decoder for chord labels

#### Single Source of Truth

The crucial point: Wav2Vec features are the ONLY audio analysis the machine uses for harmonic/melodic decisions. The Wav2Vec chord classifier (which I'll explain in Section 10) takes those same 768D features and decodes them into chord names for my benefit—"oh, MusicHal learned that sequence during the Dm7 section." But the chord name never feeds back into the decision-making. The machine only sees gesture tokens.

This solves a problem I kept hitting in earlier versions: mismatch between what the machine analyzed and what it generated. If I extract chords one way during training and another way during generation, the patterns don't align. Single source of truth fixes that.

Why this matters musically: The machine can capture microtonal inflections, spectral qualities, timbral variations that get erased when you force everything into 12-tone equal temperament chord symbols. When I bend a note, that bend is part of the gesture token. When I play a C that's slightly sharp, that sharpness affects the feature vector. The machine's perception is continuous and high-dimensional, not discrete and symbolic.

It's more honest about what machine perception actually is.

And thinking in ecosystem terms (Waters, 2007): the perception isn't "in" the Wav2Vec model or "in" the Brandtsegg analyzer. It's distributed across the whole system—my playing, the microphone's transduction, the feature extraction, the memory lookup, the generation, the audio rendering, my ears hearing it, my body responding. The 768-dimensional vectors are just one moment in a continuous loop of transformation and feedback.

---

### 5. Memory - Three Ways of Remembering

A musician needs different kinds of memory.

**Short-term**: What just happened in the last few bars? Essential for maintaining musical continuity, for knowing where you are in the phrase.

**Pattern learning**: Recognizing that this phrase is similar to something from earlier in the performance or from past performances. The basis of style, of learned musical vocabulary.

**Long-term thematic**: Remembering and developing a motif across the whole performance—introducing it, letting it rest, bringing it back varied. What makes an improvisation coherent over 10+ minutes.

I built all three.

#### Short-Term: The Memory Buffer

`memory/memory_buffer.py` implements a 180-second rolling window. Every musical event that comes in—gesture token, rhythm ratio, midi note, timestamp, all the perceptual features—gets stored in this buffer. Old events roll off the back as new ones come in.

This is like working memory in human cognition. When MusicHal needs context to make a decision—"what have I been playing recently?"—it queries this buffer. The `phrase_generator.py` does this constantly: "Get me the last 10 gesture tokens," "What's the average consonance over the last 30 seconds," "Is the melody trending upward or downward?"

The buffer also normalizes features. Barlow complexity might range from 1 to 50, consonance from 0 to 1, gesture tokens from 0 to 255. Everything gets normalized to comparable ranges so multi-parameter requests can weight them appropriately.

Simple data structure, but essential. Without short-term memory, there's no musical context, no way to know "where we are."

#### Pattern Learning: The AudioOracle

This is the core of what "trained on Itzama.wav" actually means.

Based on Gérard Assayag's Factor Oracle algorithm (Assayag & Dubnov, 2004), extended by Shlomo Dubnov to continuous audio features (Dubnov et al., 2007), this is an automaton that incrementally learns sequences.

Here's how it works during training:

1. Load Itzama.wav (183.7 seconds of me playing modal jazz)
2. Extract features every onset (Wav2Vec gesture tokens + Brandtsegg ratios)
3. Feed each feature vector into the AudioOracle sequentially
4. For each new vector, the Oracle:
   - Creates a new state
   - Checks: is this similar to any previous state? (Euclidean distance < threshold)
   - If yes, creates a **suffix link** pointing back to that similar state
   - Records forward transitions: "from state 42, we went to state 43"

After processing all of Itzama, the AudioOracle is a graph structure encoding all the patterns. Suffix links capture repetitions, variations, and structural relationships.

**Generation** works by traversing this graph:

1. Query: "What follows gesture_token 142 when rhythm_ratio is [3,2] and consonance > 0.7?"
2. Find current state in the graph matching those constraints (using request mask—Section 7)
3. Look at forward transitions from that state
4. Probabilistic selection weighted by transition frequencies and request parameters
5. Output the selected state's features → convert to MIDI notes

The beauty of Factor Oracle: it learns in linear time, generates with natural phrase structure (following learned transitions and jumping through suffix links), and automatically captures repetitions without explicit pattern matching algorithms.

**The code:**  
`memory/polyphonic_audio_oracle.py` - full implementation  
Key methods: `add_state()`, `generate_with_request()`, `_create_suffix_link()`

Serializes to JSON after training so I don't have to retrain every time I want to perform. Loading a trained model takes 2 seconds, training Itzama takes 5 minutes. Big difference.

#### Long-Term Thematic: Phrase Memory (My Contribution)

This is where I went beyond existing AudioOracle systems.

Real improvisers don't just respond to immediate context. They introduce motifs, let them develop, bring them back varied. Listen to any John Coltrane solo—he'll play a 4-note fragment early on, then 2 minutes later bring it back transposed up a fifth, then later invert it, then play it backwards. That's what makes a performance feel cohesive, intentional, like it's going somewhere.

AudioOracle alone doesn't do this. It responds to local context but has no mechanism for long-range thematic recall.

So I built `agent/phrase_memory.py`:

**What it does:**

- Extracts **motifs** (3-5 note patterns) from phrases MusicHal generates
- Stores up to 20 recent motifs with timestamps and musical context
- Every 30-60 seconds (random interval to avoid mechanical repetition), checks: "Should I recall a theme?"
- If yes: selects a motif, applies a **variation** (transpose, invert, retrograde, augment, diminish), returns it as the next phrase to play

**The variations:**

- **Transpose**: shift all notes by +7 semitones (up a fifth, classic jazz development)
- **Invert**: mirror the intervals around the first note (ascending third becomes descending third)
- **Retrograde**: play the notes backwards
- **Augment**: double all durations (slower, more spacious)
- **Diminish**: halve all durations (faster, more intense)

**Integration:**  
`agent/phrase_generator.py` checks phrase memory before generating new material:

```python
# THEMATIC DEVELOPMENT: Check if we should recall/develop existing theme
if self.phrase_memory.should_recall_theme():
    motif = self.phrase_memory.get_current_theme()
    if motif:
        variation_type = random.choice(['transpose', 'invert', 'retrograde', 'augment', 'diminish'])
        variation = self.phrase_memory.get_variation(motif, variation_type)
        # Build phrase from variation
        return phrase

# Otherwise, generate new material from AudioOracle
```

**Why this matters:**

Without phrase memory, performances felt reactive but aimless—MusicHal responded to me moment by moment, but there was no through-line, no sense of development. With phrase memory, motifs recur, get varied, create relationships across time. It feels like MusicHal has a musical memory, like it's intentionally developing ideas.

Is this "real" intentionality? No—the algorithm is deterministic (check timer, select motif, apply variation). But the effect is similar to how human improvisers recall and develop themes. Does the mechanism matter if the musical result creates coherence?

I'm still thinking about that question.

#### Memory Hierarchy Diagram

```
┌─────────────────────────────────────────────────────┐
│  PHRASE MEMORY (Long-term Thematic)                 │
│  - 20 motifs stored                                 │
│  - Recalls every 30-60 seconds                      │
│  - Applies variations                               │
│  - Creates long-range coherence                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   │  Motif extraction
                   │
┌──────────────────▼──────────────────────────────────┐
│  AUDIO ORACLE (Pattern Learning)                    │
│  - Graph structure with suffix links                │
│  - Learned from Itzama training                     │
│  - Generates based on request + transitions         │
│  - Captures style, patterns, phrase structure       │
└──────────────────┬──────────────────────────────────┘
                   │
                   │  Queries for generation
                   │
┌──────────────────▼──────────────────────────────────┐
│  MEMORY BUFFER (Short-term Context)                 │
│  - 180-second rolling window                        │
│  - Recent events: gesture tokens, ratios, MIDI      │
│  - Feature normalization                            │
│  - Provides "what just happened" context            │
└─────────────────────────────────────────────────────┘
```

Three timescales: seconds (buffer), minutes (Oracle patterns), long-range (phrase memory). Human musical memory works similarly—we remember the last few notes clearly, we recognize patterns we've played before, and we recall themes from earlier in the performance.

The phrase memory was the last piece I added. Before it, MusicHal was sophisticated but felt like it had no long-term intent. After adding it, performances changed—there was a sense of going somewhere, of ideas being introduced and developed. The system started to feel less like a reactive algorithm and more like a musical partner with memory.

Whether that's illusion or reality is the Chinese Room question again. But it works musically.

---

### 6. Personality - Six Modes, Sticky Behavior

I needed MusicHal to have character. Not just one character—different moods, different ways of relating to what I play. But each mode needs to persist long enough (30-90 seconds) that I can feel its personality, not just flash by in 2 seconds.

When you play with a human musician, they don't switch personalities every 5 seconds. They might be supportive for a while, shadowing your lines closely. Then they challenge you, playing independent counterpoint. Then they step back, giving you space. These relational modes develop over minutes, not seconds. You respond to the mode, it shapes the musical dialogue.

Early versions of MusicHal switched modes every 2-5 seconds. Too fast. You'd feel a shift starting to happen—"oh, it's imitating me now"—and then it would switch again before you could respond, develop a dialogue within that mode. Frustrating.

Solution: **sticky modes**. Each mode persists for 30-90 seconds (randomized to avoid mechanical predictability) before transitioning. Long enough to establish character, short enough to maintain interest.

#### The Six Modes

Based on IRCAM's research into musical interaction modes (Thelle & Wærstad, 2023), extended with my own additions:

**SHADOW (similarity threshold: 0.9, temperature: 0.7)**  
Follows me very closely, almost like a shadow. High similarity threshold means it only generates patterns that closely match my recent input. Low temperature means more predictable, less exploratory choices. Request weight: 0.95 —it's strongly biased toward matching my gesture tokens.

Musically: tight imitation, echoing my phrases, staying close to my melodic contour and rhythmic patterns. Like a harmony player who doubles your line.

**MIRROR (similarity threshold: 0.6, temperature: 1.0)**  
Complementary, balanced dialogue. Medium similarity—it's responding to me but with more freedom. Medium temperature—balanced exploration. Request weight: 0.7—uses my input but with more independence.

Musically: call-and-response, complementary phrases, harmonic support without literal imitation. Like a good comp player who responds without duplicating.

**COUPLE (similarity threshold: 0.1, temperature: 1.3)**  
Independent, does its own thing. Very low similarity threshold—it's barely constrained by my input. High temperature—more exploratory, surprising choices. Request weight: 0.3—aware of my playing but following its own path.

Musically: counterpoint, contrasting lines, independent melodic development. Like Gerry Mulligan and Chet Baker's pianoless quartet—two lines in dialogue but each maintaining independence.

**IMITATE, CONTRAST, LEAD (legacy modes)**  
I kept these from earlier versions but don't use them as much. IMITATE is similar to SHADOW, CONTRAST is like COUPLE but more aggressive, LEAD tries to initiate patterns rather than respond.

#### Mode Parameters Table

| Mode     | Similarity | Request Weight | Temperature | Phrase Length | Character       |
| -------- | ---------- | -------------- | ----------- | ------------- | --------------- |
| SHADOW   | 0.9        | 0.95           | 0.7         | (3, 6) notes  | Close imitation |
| MIRROR   | 0.6        | 0.7            | 1.0         | (4, 8) notes  | Complementary   |
| COUPLE   | 0.1        | 0.3            | 1.3         | (6, 12) notes | Independent     |
| IMITATE  | 0.85       | 0.9            | 0.75        | (3, 6) notes  | Echo            |
| CONTRAST | 0.2        | 0.4            | 1.2         | (5, 10) notes | Opposition      |
| LEAD     | 0.4        | 0.5            | 1.1         | (4, 9) notes  | Initiating      |

**Phrase length** varies by mode—SHADOW plays shorter, more responsive phrases, COUPLE plays longer, more independent lines.

**Temperature** controls randomness in generation. Lower temp = more deterministic, predictable. Higher temp = more exploratory, surprising.

**Request weight** determines how strongly the generation is biased toward matching my input vs. exploring the learned patterns freely.

These aren't arbitrary numbers. I tuned them over dozens of sessions, listening for when modes felt musically distinct, when SHADOW felt too rigid or COUPLE felt too chaotic.

#### Sticky Behavior Implementation

`agent/behaviors.py` manages mode selection and persistence:

```python
class BehaviorEngine:
    def __init__(self):
        self.current_mode = BehaviorMode.SHADOW
        self.min_mode_duration = 30.0  # seconds
        self.max_mode_duration = 90.0  # seconds
        self.current_mode_duration = random.uniform(30, 90)
        self.mode_start_time = time.time()
    
    def _select_new_mode(self, current_event, memory_buffer, clustering):
        """Select a new behavior mode"""
        elapsed = time.time() - self.mode_start_time
        
        if elapsed < self.current_mode_duration:
            return  # Stay in current mode (sticky behavior)
        
        # Mode duration expired, select new mode
        # (Selection logic based on musical context)
        self.current_mode = new_mode
        self.mode_start_time = time.time()
        self.current_mode_duration = random.uniform(30, 90)
```

When a mode switches, MusicHal announces it: "🎭 Switching to SHADOW mode (duration: 67s)." That lets me know the personality is changing, I can anticipate the shift in musical relationship.

#### Why This Matters Musically

Before sticky modes, the system felt schizophrenic. It would imitate me for a phrase, then suddenly play something contrasting, then imitate again, no coherent relationship. I couldn't develop a musical dialogue because the relationship kept shifting.

With sticky modes, there's time to feel the mode's character and respond to it. If SHADOW mode is active for 60 seconds, I can play a phrase, hear it shadowed, play a variation, hear that shadowed, develop a call-and-response within the imitative relationship. When it shifts to COUPLE, I can feel that independence emerge and adapt my playing—maybe I play more pedal tones, giving space for the independent line.

It's closer to how human musical relationships work. You establish a relational mode and explore within it before shifting.

The mode parameters (similarity threshold, temperature, request weight) are what make each mode feel distinct. SHADOW's high request weight and low temperature make it predictable, safe, supportive. COUPLE's low request weight and high temperature make it surprising, challenging, independent. These aren't just numbers—they shape the feel of musical interaction.

Still tuning this. COUPLE's temperature of 1.3 sometimes feels too random, too chaotic. Might need to lower it to 1.15. SHADOW's similarity threshold of 0.9 sometimes feels too restrictive, maybe 0.85 would give more variation while still feeling imitative. This tuning is ongoing, based on how it feels musically.

The sticky duration (30-90 seconds) matters because musical relationships have temporal character. Waters (2007) observes that instruments and performers establish relational modes that persist—you don't redefine your relationship every phrase. The flute player's breath pattern, the room's resonance, your listening posture—these stabilize over time, creating a sustained interaction space. Rapid mode switching breaks this. Persistence allows the ecosystem to settle into a pattern, lets me respond to that pattern, lets dialogue develop within it.

---

### 7. Request Masking - How It Actually Chooses What to Play

This is where the rubber meets the road. When MusicHal decides to play something, how does it choose? This is the trust layer—if I can't understand this, I can't trust the system.

Traditional AudioOracle generation is simple: traverse the graph, pick the next state based on transition probabilities. But that's pure statistical continuation, no musical intelligence, no ability to say "I want something consonant here" or "match my rhythmic pattern."

I needed conditional generation: given my recent input (gesture tokens, rhythm ratios, consonance levels), generate something that matches specific musical constraints.

Enter **request masking**.

#### Multi-Parameter Requests

Each behavior mode builds a request specifying what kind of musical material to generate. Requests have up to three parameters (primary, secondary, tertiary), each with a weight.

**SHADOW mode request:**

```python
request = {
    'primary': {
        'parameter': 'gesture_token',
        'type': '==',  # exact match
        'value': 142,  # the token I just played
        'weight': 0.95
    },
    'secondary': {
        'parameter': 'consonance',
        'type': 'gradient',
        'value': 2.0,  # prefer higher consonance
        'weight': 0.5
    },
    'tertiary': {
        'parameter': 'barlow_complexity',
        'type': '==',
        'value': 2.7,  # match rhythmic complexity
        'weight': 0.4
    }
}
```

This says: "Find patterns that match gesture_token 142 (weight 0.95 = very strongly), prefer consonant patterns (weight 0.5 = moderately), and match Barlow complexity 2.7 (weight 0.4 = weakly)."

**MIRROR mode request:**

```python
request = {
    'primary': {
        'parameter': 'midi_relative',  # interval from my note
        'type': '>',
        'value': 3,  # at least a third above
        'weight': 0.7
    },
    'secondary': {
        'parameter': 'consonance',
        'type': 'gradient',
        'value': 1.5,
        'weight': 0.6
    }
}
```

This says: "Find patterns at least a third above my note (complementary harmony) with moderate consonance preference."

**COUPLE mode request:**

```python
request = {
    'primary': {
        'parameter': 'consonance',
        'type': '<',
        'value': 0.5,  # prefer dissonance
        'weight': 0.3  # but only weakly
    },
    'secondary': {
        'parameter': 'rhythm_ratio',
        'type': '==',
        'value': [7, 4],  # contrasting rhythm
        'weight': 0.4
    }
}
```

This says: "Prefer dissonant patterns (but not strongly) with contrasting rhythmic ratios." Lower weights = more exploratory, less constrained by my input.

The code for building these requests is in `agent/phrase_generator.py`, methods `_build_shadowing_request()`, `_build_mirroring_request()`, `_build_coupling_request()`.

#### How Masking Works

`memory/request_mask.py` implements the masking:

1. **For each parameter in the request**, create a probability mask over all learned states in the AudioOracle:

   - Exact match (`==`): probability 1.0 for matching states, 0.0 for others
   - Threshold (`>`, `<`): probability 1.0 for states meeting threshold, 0.0 for others
   - Gradient: probability falls off smoothly with distance from target value

2. **Blend masks** according to weights:

   ```
   P_final = mask_primary * weight_primary + mask_secondary * weight_secondary + ...
   ```

3. **Combine with base transition probabilities** from AudioOracle:

   ```
   P_combined = P_request * request_total_weight + P_oracle * (1 - request_total_weight)
   ```

4. **Apply temperature scaling**:

   ```
   P_temp = P_combined ^ (1/T) / sum(P_combined ^ (1/T))
   ```

5. **Sample** from P_temp to select next state

The math isn't arbitrary—it's how I tune the balance between request-driven generation (responding to my input) and learned pattern structure (AudioOracle's captured style).

#### Gradient Curves

For gradient requests, the probability falls off with distance:

```python
def _apply_gradient_mask(self, parameter, target_value, gradient_direction):
    distances = abs(state_values - target_value)
    normalized_distances = distances / max_distance
    
    if gradient_direction > 0:
        # Prefer values above target
        probabilities = (1 - normalized_distances) ** abs(gradient_direction)
    else:
        # Prefer values below target
        probabilities = (1 - normalized_distances) ** abs(gradient_direction)
    
    return probabilities
```

Higher gradient values create steeper falloff—strong preference. Lower values create gentler falloff—weak preference.

This lets me specify "strongly prefer consonance" (gradient 3.0) vs. "weakly prefer consonance" (gradient 1.2) and hear the difference in how strictly the request is followed.

#### Why Mandatory Masking (No Uniform Fallback)

Early versions had a fallback: if the request doesn't match anything well, just pick randomly. That broke trust completely.

I'd play a phrase, MusicHal would shadow it (great!), then I'd play another phrase, and it would generate something totally random, unrelated. The musical dialogue would break down. I couldn't tell if it was listening or just... malfunctioning.

Solution: **mandatory masking, no uniform fallback**. The request always applies, even if weakly. If nothing matches well, it picks the best available match, even if it's not great. I'd rather get a weak match than a random note.

This was a design choice driven by trust. Seeing the decision logs confirm "pattern match: 23% (low but best available)" tells me the system tried to respond to my input, even if the match wasn't strong. That's better than "no match found, generated randomly."

#### Temperature: Tuning Predictability

Temperature controls the randomness of selection from the probability distribution.

- **T = 0.7 (SHADOW)**: More deterministic. If one state has 40% probability and another has 30%, the 40% state gets selected much more often. Predictable, safe.

- **T = 1.0 (MIRROR)**: Balanced. The 40% state is selected proportionally more often, but not overwhelmingly.

- **T = 1.3 (COUPLE)**: More exploratory. The 40% and 30% states have more similar selection probabilities. Surprising, less predictable.

Tuning temperature per mode is how I make SHADOW feel safe/predictable and COUPLE feel exploratory/surprising, even though they're using the same underlying generation algorithm.

These aren't magical numbers—they're the result of listening sessions where I adjusted values and noticed when modes felt musically right.

#### The Trust Layer

Request masking is how MusicHal responds to my input. Decision transparency (Section 12) is how I verify it's actually doing this. Together, they solve the trust problem.

When I see in the logs:

```
Request: match gesture_token=142 (weight 0.95)
Pattern match: 87% similarity to state 234 from training bar 23
Generated: F4 → G4 → A4 → A#4
```

I know the system used my input (gesture_token 142), found a strong match (87%), and generated based on that. Not random, not arbitrary—conditional generation based on my playing.

Does seeing the mechanism break the magic? Sometimes. Seeing "87% match to training bar 23" confirms trust but makes it feel mechanical, less mysterious.

Trade-off I'm still thinking about: use `--debug-decisions` during development and testing to verify the system works, turn it off during performances to preserve mystery? Or is transparency itself part of the artistic statement—revealing the mechanism as part of the work, like showing the strings and pulleys in a puppet show?

No clear answer yet. For now, I have the option. Trust when I need it, magic when I want it.

---

## Part III: Training - Teaching MusicHal to Learn from Jazz

### 8. The Training Pipeline (Chandra_trainer.py)

Training isn't like supervised learning where you label examples: "this is C major, this is D minor." It's more like memory formation. Here's 3 minutes of me playing, now extract every musical moment, build a vocabulary of gestures, learn which patterns tend to follow each other.

No teacher signal, no loss function to minimize. Just: absorb the musical structure, encode it in the AudioOracle's graph, serialize it to JSON.

`Chandra_trainer.py` is 1800 lines of this process.

#### The Workflow (As I Experience It)

```bash
python Chandra_trainer.py \
  --file input_audio/Itzama.wav \
  --max-events 1000 \
  --wav2vec \
  --gpu
```

The output filename is auto-generated: `Itzama_DDMMYY_HHMM.json` and saved to `JSON/`. If I want a specific name, I can add `--output JSON/my_custom_name.json`. The training also exports `Itzama_DDMMYY_HHMM_model.json` (AudioOracle graph) and `Itzama_DDMMYY_HHMM_correlation_patterns.json` (learned harmonic-rhythmic correlations for UnifiedDecisionEngine).

What happens:

**Step 1: Load audio** (2 seconds)  
Itzama.wav - 183.7 seconds of modal jazz I recorded. Mono, 44.1kHz. The system loads it, displays waveform statistics, estimates tempo (around 140 BPM but it varies, this is rubato jazz not metronomic).

**Step 2: Extract features** (3 minutes)  
This is the dual perception pipeline:

- **Wav2Vec 2.0** processes audio in 1024-sample windows, extracting 768-dimensional feature vectors every onset. The model (facebook/wav2vec2-base) is 95MB, already trained on massive amounts of audio. I don't retrain it—I use it as a perceptual feature extractor.

- **Brandtsegg analyzer** detects onsets, calculates inter-onset intervals, finds rational approximations (rhythm ratios), computes Barlow complexity, measures deviation from expected timing.

- **Feature merging** combines both streams into unified events with 15+ fields per event: gesture_token, rhythm_ratio, barlow_complexity, consonance, deviation_polarity, midi (estimated pitch), timestamp, velocity, etc.

Output: ~850 events for Itzama (one event every 0.2 seconds on average, but varying with musical density).

**Step 3: Temporal smoothing** (30 seconds)  
This is where I solve the "chord flicker" problem (more in Section 9). Sustained notes get sampled multiple times—the system groups events within time windows, applies onset-aware averaging, removes false rapid changes.

Before smoothing: 850 events, many duplicates  
After smoothing: ~620 events, musical structure preserved

**Step 4: Wav2Vec chord classification** (30 seconds)  
For my benefit only—the machine doesn't use these labels. The neural decoder trained on synthetic chords takes Wav2Vec features and outputs chord names: "Cmaj7," "Dm7," "G7alt."

Sometimes wrong, but consistently wrong in musically coherent ways. Good enough for me to understand "MusicHal learned this sequence during the ii-V-I in bars 15-20."

**Step 5: AudioOracle training** (1 minute)  
Feed all 620 events sequentially into the Oracle. For each event:

- Create new state with full feature vector
- Calculate distance to previous states
- Create suffix links to similar states (distance < threshold)
- Record forward transition

Result: Graph with ~620 states, thousands of suffix links, transition probabilities.

**Step 6: RhythmOracle training** (30 seconds)  
Separate Oracle learning just rhythmic patterns (inter-onset intervals, rhythm ratios). Less critical than the main AudioOracle but adds another layer of pattern learning.

**Step 7: Serialization** (10 seconds)  
Save to JSON:

- `JSON/Itzama_241025_1229.json` - metadata (training parameters, statistics, timestamps)
- `JSON/Itzama_241025_1229_model.json` - the actual Oracle graph structure (states, transitions, suffix links)

Total training time: ~5-6 minutes for 183 seconds of audio.

Loading a trained model takes 2 seconds. Big difference. I train once, perform many times.

#### My Training Data

These are recordings of me playing, not downloaded datasets:

**Itzama.wav** (183.7s)  
Pop track, electronically based beat music. Very modern, not jazz. Clear 6-phase performance arc: intro (establishing electronic groove) → development (increasing density) → climax (high register, intense) → resolution (return to low register) → second climax → final resolution.

Why I chose it: structural variety, wide pitch range, electronic rhythmic complexity. If MusicHal can learn from this, it has a rich vocabulary. Genre contrast with the jazz ballads.

**Georgia.wav** (~120s)  
"Georgia on My Mind" (Carmichael/Gorrell, 1930), jazz standard ballad. Voice and piano (Andreas Ulvo), no other instruments. Slow, intimate, rubato phrasing.

Why: test if system handles vocal material, slow tempos, duo interaction. Very different from Itzama's electronic feel.

**Nineteen.wav** (~140s)  
My own composition, jazz standard style ballad. With Andreas Ulvo (piano) and Mats Eilertsen. Slow, focus on melodic phrasing and space.

Why: test if system captures sustained notes properly (temporal smoothing critical here), learns phrase boundaries from longer rests, handles trio interaction.

**Curious Child.wav** (~100s)  
Up-tempo (~200 BPM), rhythmically driven, more aggressive articulation.

Why: extreme tempo test, rhythmic emphasis, sharper attacks.

I haven't trained on all of these yet—most development has been on Itzama. But the pipeline handles any of them. Same process, different learned vocabulary. The genre diversity here (electronic beat music vs. jazz ballads) means different models would have very different characters.

#### What "Trained on Itzama" Actually Means

Not: "memorized the exact audio and will play it back."  
Yes: "built a graph structure encoding the patterns—which gesture tokens follow which, which rhythm ratios appear in which contexts, which transitions are common vs. rare."

When MusicHal generates, it's not playing back Itzama. It's recombining learned patterns in response to my live input. Patterns from bar 15 might get combined with patterns from bar 78, guided by request masking to match my current gesture token.

This is why the system has a "voice"—it's shaped by the specific training audio. MusicHal trained on Itzama (electronic beat music) sounds different from MusicHal trained on Georgia (jazz ballad). Same algorithm, different learned structure, completely different genre character.

That's intentional. I want localized musical identity, not universal music knowledge.

#### Training Logs and Verification

While training, Chandra logs everything:

```
Loading audio: Itzama.wav
Duration: 183.7s, Sample rate: 44100 Hz
Estimating tempo: 138.4 BPM (confidence: 0.72)

Extracting Wav2Vec features... 
Progress: [████████████████████] 100% | 850 events

Temporal smoothing... 
Before: 850 events | After: 620 events (removed 230 duplicates)

Training AudioOracle...
States: 620 | Suffix links: 1847 | Avg transitions per state: 2.3

Training complete. Model saved to JSON/Itzama_241025_1229.json
```

I can check the JSON metadata to see what was learned:

```json
{
  "training_audio": "Itzama.wav",
  "duration": 183.7,
  "events_processed": 620,
  "oracle_states": 620,
  "suffix_links": 1847,
  "avg_transitions": 2.3,
  "pitch_range": [48, 84],  // C3 to C6
  "timestamp": "20251024_1229"
}
```

This tells me: wide pitch range (3 octaves), dense suffix link structure (3 links per state on average), reasonable transition density.

If I see too few suffix links (<1 per state), the similarity threshold might be too strict—not finding enough matches. If too many (>5 per state), threshold might be too loose—finding false matches.

Tuning these thresholds is part of the training process. Not automatic, requires musical judgment: does the system generate material that sounds stylistically coherent with the training audio?

---

### 9. Temporal Smoothing - Solving the Chord Flicker Problem

The problem hit me hard about 6 months into development.

I held a single note—let's say C4—for 2 seconds. The system sampled it 10 times (every 0.2 seconds). The Wav2Vec chord classifier, which has some inherent noise, labeled those samples:

C, Cdim, C, Cdim, C, Cdim, C, C, Cdim, Cdim

The AudioOracle learned: "C follows Cdim follows C follows Cdim..."—a rapid flickering that never happened musically. When it generated, it played back this false rapid chord progression. Sounded manic, wrong, unusable.

This is over-sampling. The system doesn't know that samples 1-10 are all the same sustained note, not 10 separate events.

#### The Solution: Time-Window Averaging + Onset Awareness

`core/temporal_smoothing.py` implements:

**Time-window grouping** (0.3-second windows):  
Events within 0.3 seconds of each other get grouped. If there are no onsets in the group (all samples from a sustained note), merge them into a single event by averaging features.

**Onset-aware exception**:  
If there's an onset in the window (new attack), that's a genuinely new event—don't merge it. This preserves rapid articulations (fast passages, drum hits) while removing over-sampling of sustains.

**Feature averaging**:  
For merged events, average all continuous features (consonance, Barlow complexity, deviation), take the most common discrete features (gesture_token, rhythm_ratio).

The code:

```python
def smooth_events(events, window_size=0.3):
    smoothed = []
    current_window = []
    window_start = events[0]['timestamp']
    
    for event in events:
        # Check if event falls within current window
        if event['timestamp'] - window_start < window_size:
            current_window.append(event)
        else:
            # Window complete, merge if no onsets
            if not any(e.get('onset', False) for e in current_window):
                # Merge: average continuous features, mode of discrete features
                merged = _merge_events(current_window)
                smoothed.append(merged)
            else:
                # Preserve all events (genuine rapid articulation)
                smoothed.extend(current_window)
            
            # Start new window
            current_window = [event]
            window_start = event['timestamp']
    
    return smoothed
```

#### Before and After

**Before temporal smoothing** (raw events from 2-second sustained C4):

```
t=45.00s: gesture_token=087, chord=C
t=45.21s: gesture_token=087, chord=Cdim
t=45.43s: gesture_token=086, chord=C
t=45.61s: gesture_token=087, chord=Cdim
t=45.79s: gesture_token=087, chord=C
t=45.98s: gesture_token=088, chord=Cdim
t=46.15s: gesture_token=087, chord=C
t=46.34s: gesture_token=087, chord=C
t=46.51s: gesture_token=087, chord=Cdim
t=46.72s: gesture_token=087, chord=Cdim
```

10 events, rapid flickering between C and Cdim (false progression).

**After temporal smoothing**:

```
t=45.00s: gesture_token=087, chord=C, duration=2.0s
```

1 event, accurately representing the sustained note.

**Before smoothing** (chord progression from bars 15-20 of Itzama):

```
C, Cdim, C, Cdim, Dm7, Dm, Dm7, Dm7, Dm, G7, Gsus, G7, G7, G7alt, G7, G7, Cmaj7, C6, Cmaj7, Cmaj7, C, Cmaj7
```

22 events, many duplicates and false rapid changes.

**After smoothing**:

```
C, Dm7, G7, Cmaj7
```

4 events, the actual ii-V-I progression I played.

#### Why This Fix Was Critical

Before temporal smoothing, the AudioOracle learned false patterns. When it generated, it played rapid flickering changes that sounded mechanical, neurotic. I couldn't use the system—it didn't sound musical.

After temporal smoothing, the Oracle learned actual musical structure. Sustained notes stayed sustained, chord progressions matched what I actually played, generated material sounded coherent.

This single fix made the difference between "interesting research project but unusable" and "actually works musically."

The window size (0.3 seconds) is a parameter I tuned. Too small (0.1s), doesn't catch all the over-sampling. Too large (0.8s), starts merging genuinely separate events. 0.3s works for my playing style—might need adjustment for faster or slower music.

---

### 10. Wav2Vec Chord Classifier - Training the Translator

I needed chord labels for my own understanding: "What section of Itzama did MusicHal learn that pattern from? Oh, the Dm7 section in bars 15-20."

But I didn't want to use traditional chroma extraction—that's separate from Wav2Vec, creates a mismatch between what the machine perceives and what I label. Remember: single source of truth.

Solution: train a neural network to decode chord names FROM Wav2Vec features. Same 768D features the machine uses, just interpreted differently.

#### The Training Process

**Step 1: Generate synthetic chord dataset**

`generate_synthetic_chord_dataset.py` creates artificial audio of 48 chord types (major, minor, diminished, augmented, 7ths, 9ths, alterations, etc.):

For each chord:

- Generate MIDI notes (root + 3rd + 5th + extensions)
- Synthesize to audio using pure sine tones
- Add slight detuning and envelope shaping (more realistic than perfect sine waves)
- 1000 examples per chord, varying register and voicing

Total: 48,000 audio examples, ~2 hours of synthetic audio.

**Step 2: Extract Wav2Vec features**

Process all synthetic audio through Wav2Vec 2.0, extract 768D features for each chord.

**Step 3: Train MLP classifier**

`train_wav2vec_chord_classifier.py`:

Architecture:

```
Input: 768 dimensions (Wav2Vec features)
  ↓
Hidden layer 1: 128 neurons (ReLU activation)
  ↓
Hidden layer 2: 64 neurons (ReLU activation)
  ↓
Output: 48 neurons (softmax, one per chord class)
```

Training: 80/20 train/validation split, Adam optimizer, cross-entropy loss, 50 epochs.

Takes ~20 minutes on MPS (Mac GPU).

**Step 4: Integrate into Chandra pipeline**

`hybrid_training/wav2vec_chord_classifier.py` loads the trained model:

```python
class Wav2VecChordClassifier:
    def __init__(self, model_path):
        self.model = load_trained_mlp(model_path)
        self.chord_names = ['C', 'Cm', 'Cdim', 'Caug', 'C7', ...]  # 48 classes
    
    def classify(self, wav2vec_features):
        """Decode chord name from Wav2Vec features"""
        probabilities = self.model(wav2vec_features)
        chord_idx = np.argmax(probabilities)
        confidence = probabilities[chord_idx]
        return self.chord_names[chord_idx], confidence
```

During training, every event gets a chord label this way.

#### Results (Honest Assessment)

**Accuracy on synthetic data**: ~92% (cross-validation)  
Good, but these are perfect synthetic chords, not real music.

**Accuracy on real jazz** (Itzama.wav, hand-labeled for verification): ~68%  
Much lower. Real jazz has:

- Microtonal intonation (slightly sharp/flat notes)
- Overtones and noise from acoustic instrument
- Ambiguous voicings ("is that Cmaj7 or C6?")
- Passing tones and chromaticism

The classifier struggles with these, sometimes mislabels. But it's consistently wrong in musically coherent ways—if it's uncertain between Cmaj7 and C6, those are closely related, not random confusion.

**Good enough for my purpose**: understanding, not ground truth. I don't need perfect chord labels. I need to know roughly what harmonic region the system learned patterns from.

When I see "pattern match from state 234, chord=Dm7, bar ~18," that's enough information to understand the context, even if the actual chord was Dm9 or the bar number is approximate.

#### Why This Approach?

Alternative: use traditional chroma extraction (like `librosa.feature.chroma_cqt`), pattern match against chord templates.

Problem: chroma extraction analyzes audio independently of Wav2Vec. The machine perceives gesture tokens, I analyze chroma—mismatch. When I say "MusicHal learned this during Dm7," I'm labeling based on analysis the machine never saw.

With Wav2Vec chord decoder: single source of truth. The machine's perceptual features (768D) contain enough information to decode chord names. I'm just interpreting the same data differently (gesture tokens vs. chord labels), not analyzing it twice.

More honest, more aligned with how the system actually works.

---

## Part IV: Playing Live - MusicHal in Performance

### 11. The Real-Time System

Training is offline—analyze audio, build memory, serialize to JSON. Performance is real-time—microphone input, onset detection, dual perception, memory lookup, generation, MIDI output. Everything has to happen fast enough that the musical dialogue doesn't break down.

Sub-50ms latency is the threshold. Beyond that, the system starts to feel sluggish, responses come too late to feel like dialogue. It's like talking to someone on a satellite phone with a 2-second delay—technically communication, but the conversational flow breaks.

`MusicHal_9000.py` is ~1500 lines of real-time pipeline.

#### The Architecture (Signal Flow)

```
Audio Input (microphone/line in)
    ↓
DriftListener (onset detection, amplitude tracking)
    ↓ (new onset detected)
HybridPerception (Wav2Vec + Brandtsegg analysis)
    ↓ (unified event with 15+ features)
MemoryBuffer.add_event() (store for context)
    ↓
BehaviorEngine.update() (select/maintain mode)
    ↓ (current mode: SHADOW/MIRROR/COUPLE)
Scheduler.should_respond() (timing logic, density control)
    ↓ (yes, generate response)
PhraseGenerator.generate_phrase()
    ├─ Check phrase_memory.should_recall_theme()?
    │  ├─ Yes → recall motif, apply variation
    │  └─ No → build request based on mode
    ├─ Query AudioOracle.generate_with_request()
    │  └─ Apply request_mask, traverse graph, select states
    └─ Convert states to MusicalPhrase (notes + durations)
    ↓
FeatureMapper.map_to_midi() (audio features → MPE MIDI)
    ↓
MIDI Output (via rtmidi)
    ↓
SuperCollider synths (audio rendering)
```

This diagram shows the technical signal flow—necessarily linear for documentation purposes. But the actual experience is ecosystem feedback loops (Waters, 2007). I hear MusicHal's output through the room's acoustics. That shapes my next phrase. MusicHal analyzes my adjusted playing. Its response reflects both the training patterns and my reaction to its previous output. The "output" becomes part of the environmental input, continuously cycling. There's no clear beginning or end to the loop—we're both responding to each other's responses, coupled in real-time.

**Unified Decision Engine:** The UnifiedDecisionEngine combines harmonic and rhythmic decisions based on response modes (support, contrast, imitate, lead). Correlation patterns learned during training are now loaded and passed to the decision context, enabling the engine to make choices informed by the harmonic-rhythmic relationships it observed in training data. This cross-modal intelligence is working, though still being refined—the infrastructure is complete, the musical results are promising but not yet fully optimized.

**[Video Example 1: Screen recording of live session with decision transparency showing pattern matching in real-time]**

Every component has to complete its work within milliseconds. Wav2Vec feature extraction is the slowest (20-30ms), but we batch it with onset detection so it doesn't add latency.

#### Starting a Session

```bash
python MusicHal_9000.py \
  --hybrid-perception \
  --wav2vec \
  --gpu \
  --visualize
```

The system automatically loads the most recent model from `JSON/` and uses the system default audio input. If I want to override these, I can specify `--model` and `--input-device`, but usually I don't need to. The `--debug-decisions` flag is optional—adds real-time decision logging to the terminal, which I turn on when I'm debugging but not during normal performance.

What happens:

**Initialization** (2-3 seconds):

- Load trained model JSON (AudioOracle graph structure, metadata)
- Initialize Wav2Vec model (95MB, loads onto GPU if available)
- Set up MIDI interface (rtmidi scanning for available ports)
- Initialize audio input (select microphone, buffer size 1024 samples)
- Create log files (conversation, decisions, audio analysis)

**Ready state**:

```
MusicHal 9000 initialized.
Model: Itzama_241025_1229 (620 states, trained Oct 24 12:29)
Audio input: Built-in Microphone
MIDI output: IAC Driver (Bus 1)
Behavior mode: SHADOW (duration: 47s)

Listening...
```

Now I play. The system listens, analyzes, waits for the right moment, responds.

#### The Main Loop

Simplified from `MusicHal_9000.py`:

```python
while running:
    # 1. Listen for audio input
    audio_chunk = audio_input.read()
    
    # 2. Detect onsets
    if drift_listener.detect_onset(audio_chunk):
        # 3. Extract perceptual features
        event = hybrid_perception.extract_features(audio_chunk)
        
        # 4. Store in memory buffer
        memory_buffer.add_event(event)
        
        # 5. Update behavior mode (check if duration expired)
        behavior_engine.update(event, memory_buffer)
        
        # 6. Decision transparency: log human input
        logger.log_human_event(event)
        
        # 7. Should we respond? (timing, density control)
        if scheduler.should_respond(event):
            # 8. Generate musical phrase
            phrase = phrase_generator.generate_phrase(
                mode=behavior_engine.current_mode,
                context=memory_buffer.get_recent_context(),
                oracle=audio_oracle
            )
            
            if phrase:
                # 9. Decision transparency: explain why
                explanation = decision_explainer.explain_generation(
                    mode=behavior_engine.current_mode,
                    trigger_event=event,
                    generated_notes=phrase.notes,
                    pattern_match_score=phrase.match_score
                )
                logger.log_musical_decision(explanation)
                
                # 10. Convert to MIDI and send
                midi_events = feature_mapper.phrase_to_midi(phrase)
                for midi_event in midi_events:
                    midi_output.send(midi_event)
                
                # 11. Store generated phrase in phrase memory
                phrase_memory.add_phrase(
                    phrase.notes, 
                    phrase.durations, 
                    time.time(),
                    behavior_engine.current_mode
                )
```

This loop runs continuously. Audio chunks arrive every ~23ms (1024 samples at 44.1kHz). Most chunks don't have onsets, so steps 2-11 don't execute. When there is an onset, the full pipeline runs.

Latency breakdown (measured):

- Onset detection: 1-2ms
- Wav2Vec feature extraction: 20-30ms
- Brandtsegg analysis: 5-8ms
- AudioOracle query + generation: 8-15ms
- MIDI conversion + send: 1-2ms

Total: 35-57ms, within the 50ms threshold most of the time.

#### Latency Spikes

Sometimes it's slower. If the AudioOracle is large (>1000 states) and the request mask has to evaluate all states, generation can take 40-50ms alone, pushing total latency to 80-100ms. That's when you feel the lag.

I haven't fully optimized this yet. Possible improvements: pre-compute distance matrices, use spatial indexing for fast nearest-neighbor queries in the Oracle, reduce Wav2Vec inference time with quantization.

For now, keeping training datasets reasonable (≤1000 events) keeps latency acceptable.

---

### 12. Decision Transparency - The Trust Layer

Remember the original problem: I don't trust that it's listening.

Solution: make MusicHal explain every decision in real-time. Show me why it played what it played.

`agent/decision_explainer.py` generates explanations, `core/logger.py` writes them to CSV.

#### What Gets Logged

Every time MusicHal generates a phrase, it logs:

**Timestamp**: when the decision was made  
**Mode**: SHADOW/MIRROR/COUPLE  
**Voice type**: melodic/bass  
**Trigger event**: what I just played (MIDI note, gesture token, consonance, etc.)  
**Recent context**: last 10 gesture tokens, average consonance, melody tendency (ascending/descending), rhythm ratio  
**Request parameters**: primary/secondary/tertiary, weights, target values  
**Pattern match score**: how well the generated pattern matched the request (0-100%)  
**Generated notes**: the actual MIDI notes output  
**Generated durations**: note lengths  
**Confidence**: overall confidence in the generation (based on match score)  
**Reasoning**: human-readable explanation string

The reasoning string is key. It's formatted like this:

```
🎭 SHADOW mode (melodic):
   Triggered by your F4 (MIDI 65)
   Context: gesture_token=142, consonance=0.73, melody: ascending
   Request: match gesture_token=142 (weight 0.95)
   Pattern match: 87% similarity to state 234 from training bar ~23
   Generated: F4 → G4 → A4 → A#4
   Confidence: HIGH (87%)
```

This appears in the terminal if `--debug-decisions` flag is on, and always gets written to the CSV log.

#### The CSV Format

`logs/decisions_20251024_161534.csv`:

```csv
timestamp,mode,voice_type,trigger_midi,trigger_note,context_tokens,context_consonance,context_melody_tendency,request_primary,request_secondary,pattern_match_score,generated_notes,reasoning,confidence
45.234,SHADOW,melodic,65,F4,"[142,141,142,140,...]",0.73,ascending,gesture_token==142:0.95,consonance>0.7:0.5,87,"[65,67,69,70]","Triggered by F4, matched token 142 with 87% similarity to training bar 23",0.87
47.891,SHADOW,melodic,67,G4,"[142,143,142,141,...]",0.68,ascending,gesture_token==143:0.95,consonance>0.7:0.5,72,"[67,69,70,72]","Triggered by G4, matched token 143 with 72% similarity to training bar 45",0.72
```

I can load this in Python after the session, analyze pattern match scores over time, see which modes had highest confidence, check if low-confidence generations corresponded to moments that felt musically awkward.

#### Real-Time Debug Mode

With `--debug-decisions`, every generation prints to the terminal:

```
🎵 You played: F4 (MIDI 65), gesture_token=142, consonance=0.73

🎭 SHADOW mode (melodic):
   Triggered by your F4 (MIDI 65)
   Context: gesture_token=142, consonance=0.73, melody: ascending
   Request: match gesture_token=142 (weight 0.95)
   Pattern match: 87% similarity to state 234 from training bar ~23
   Generated: F4 → G4 → A4 → A#4
   Confidence: HIGH (87%)

---

🎵 You played: G4 (MIDI 67), gesture_token=143, consonance=0.68

🎭 SHADOW mode (melodic):
   Triggered by your G4 (MIDI 67)
   Context: gesture_token=143, consonance=0.68, melody: ascending  
   Request: match gesture_token=143 (weight 0.95)
   Pattern match: 72% similarity to state 267 from training bar ~45
   Generated: G4 → A4 → A#4 → C5
   Confidence: MEDIUM (72%)

---
```

Scrolling in real-time while playing. I can glance at it to verify the system is responding to my input, or ignore it and just play.

#### Does Transparency Build Trust?

Yes and no.

**Yes**: Seeing "pattern match: 87% similarity to gesture_token 142" confirms the system used my input. It's not generating randomly, it's responding conditionally based on what I played. That solves the original trust problem.

**No**: Seeing the mechanism breaks the magic. "87% match to training bar 23" makes it feel mechanical, algorithmic, less mysterious. Like watching a magician with the lights on—you see how the trick works, which confirms it's a trick, but the wonder disappears.

Trade-off I'm still navigating. For development and testing, transparency is essential—I need to verify the system works as designed, catch bugs, tune parameters. For performance, maybe transparency is distracting, breaks the improvisational flow.

Current solution: optional flag. Turn it on when I need trust verification, off when I want to preserve mystery.

But there's a third possibility I'm considering: transparency as part of the artistic statement. Revealing the mechanism, making the algorithmic nature explicit, not hiding that this is machine generation. Like showing the strings and pulleys in a puppet show—not to break illusion, but to make the construction itself part of the work.

No clear answer yet. Still experimenting.

---

### 13. Thematic Development - Remembering and Varying

Real improvisers don't just respond to the immediate moment. They introduce motifs, let them develop, bring them back varied. That's what makes a performance coherent over time, not just a series of unrelated responses.

`agent/phrase_memory.py` implements this: extract motifs from generated phrases, store them, recall them 30-60 seconds later with variations.

#### How It Works in Practice

**Minute 0:45** - MusicHal generates a phrase in SHADOW mode: C4 - E4 - G4 - F4  
Phrase memory extracts this as a motif (4 notes, good length). Stores it with timestamp, mode, musical context.

**Minute 1:30** - Phrase memory checks: "Has it been 30-60 seconds since last theme recall?" Yes, 45 seconds. "Should I recall a theme?" Random probability check → yes.

Selects the motif (C4 - E4 - G4 - F4), chooses variation type randomly: **transpose**.

Applies transposition: +7 semitones (up a fifth, classic jazz development).

Result: G4 - B4 - D5 - C5

Returns this as the next phrase to generate. Instead of querying the AudioOracle for new material, MusicHal plays the variation.

**Musical effect**: I recognize the contour even though the pitch is different. It feels like MusicHal is developing an earlier idea, not just reacting to my current input. Creates coherence across time.

#### The Five Variations

**Transpose**: Shift all notes by fixed interval (usually +7 or +5 semitones)

- Original: C4 - E4 - G4 - F4
- Transposed +7: G4 - B4 - D5 - C5

**Invert**: Mirror intervals around the first note

- Original: C4 - E4 - G4 - F4 (intervals: +4, +3, -2)
- Inverted: C4 - Ab3 - F3 - G3 (intervals: -4, -3, +2)

**Retrograde**: Play notes backwards

- Original: C4 - E4 - G4 - F4
- Retrograde: F4 - G4 - E4 - C4

**Augment**: Double all durations (slower, more spacious)

- Original: [0.25s, 0.25s, 0.25s, 0.25s]
- Augmented: [0.5s, 0.5s, 0.5s, 0.5s]

**Diminish**: Halve all durations (faster, more intense)

- Original: [0.5s, 0.5s, 0.5s, 0.5s]
- Diminished: [0.25s, 0.25s, 0.25s, 0.25s]

These are classical variation techniques—Bach, Coltrane, everyone uses them. Building them into the algorithm gives MusicHal a basic toolkit for thematic development.

#### Does It Create Intentionality?

The algorithm is deterministic: check timer (30-60 seconds elapsed?), select motif, apply random variation, return result.

No "decision" in the conscious sense. No intentionality in the philosophical sense.

But the musical effect is similar to intentional thematic development. When I hear a motif return varied, it creates the perception that MusicHal "remembered" and "chose" to develop that idea. Whether the mechanism matters if the result works musically is the Chinese Room question again.

My current thinking: the simulation of intentionality through thematic recall is sophisticated enough to be musically useful. It creates coherence, it shapes the arc of a performance, it gives MusicHal something that feels like musical memory. Whether it's "real" memory or "simulated" memory is philosophically interesting but practically irrelevant if the musical result is coherent.

#### Musical Examples from Logs

From a session on October 24, 2025:

**t=45s** - Generated motif: [65, 67, 69, 67] (F4 - G4 - A4 - G4), durations [0.3, 0.3, 0.3, 0.4]

**t=98s** - Thematic recall: transpose +5 semitones → [70, 72, 74, 72] (A#4 - C5 - D5 - C5)

**t=152s** - Thematic recall: retrograde → [67, 69, 67, 65] (G4 - A4 - G4 - F4)

**t=187s** - Thematic recall: invert → [65, 63, 61, 63] (F4 - Eb4 - Db4 - Eb4)

Four appearances of the same basic shape across 3 minutes. Creates continuity, feels like a conversation where ideas get developed, not just a series of disconnected responses.

Sometimes the variations don't fit the harmonic context—an inverted motif lands on dissonant notes that feel awkward. That's a limitation. The phrase memory doesn't consider current harmony, just applies the variation mechanically. Future improvement: harmonic-aware variations that adapt the motif to fit the current context.

But even with that limitation, thematic recall changes the feel of performances. Before phrase memory: reactive, moment-to-moment, no long-range structure. After phrase memory: developmental, coherent over time, feels like MusicHal has intent.

---

## Part V: Evidence & Case Studies

### 14. Training Session Deep Dive: Itzama.wav

I can describe the training process conceptually, show you the data structures, explain what gets learned. But I need to be honest: some of the specific analysis I'm about to present is based on the structure of the system, not on detailed post-hoc analysis I've actually run. Consider this a template for the kind of analysis that would go here in a final publication.

#### The Recording

Itzama.wav is 183.7 seconds of me playing. Pop track leaning heavily towards electronically based beat music—very modern, not jazz at all. The piece has structure I can feel when playing it: an intro establishing the electronic groove, a development section where density increases, a climax in the high register, a resolution back down, then a second build and final resolution.

I chose this recording because it's varied enough to give MusicHal a rich vocabulary without being so complex that the patterns get lost in noise. The electronic/beat music style also gives it a different rhythmic character than traditional acoustic jazz.

#### What Training Extracted

From the training logs:

```
Loading audio: Itzama.wav
Duration: 183.7s, Sample rate: 44100 Hz
Estimating tempo: 138.4 BPM (confidence: 0.72)

Extracting Wav2Vec features... 
Progress: [████████████████████] 100% | 850 events

Temporal smoothing... 
Before: 850 events | After: 620 events (removed 230 duplicates)

Training AudioOracle...
States: 620 | Suffix links: 1847 | Avg transitions per state: 2.3

Training complete. Model saved to JSON/Itzama_241025_1229.json
```

**The numbers tell a story:**

850 raw events reduced to 620 after temporal smoothing. That's 27% reduction—substantial, meaning there was significant over-sampling of sustained notes. The smoothing algorithm did what it was supposed to: remove false rapid changes, preserve actual musical structure.

620 states in the AudioOracle. Each state represents a unique musical moment (gesture token + rhythm ratio + consonance + other features). 

1847 suffix links—that's 3 links per state on average. This indicates good pattern density. If it were <1 per state, the similarity threshold would be too strict (not finding repetitions). If it were >5 per state, too loose (finding false matches everywhere). 3 is in the sweet spot.

Average 2.3 transitions per state. Some states have many forward paths (high-density musical moments where many things can follow), others have few (more deterministic progressions).

#### Pitch Range and Distribution

From the metadata JSON:

```json
{
  "pitch_range": [48, 84],  // C3 to C6
  "pitch_distribution": {
    "C3-C4": 142,
    "C4-C5": 298,
    "C5-C6": 180
  }
}
```

Three octaves. Most activity in the C4-C5 octave (middle register), with significant material in C5-C6 (high register for the climax sections) and some bass material in C3-C4.

This distribution shapes what MusicHal can generate—it has more patterns in the middle register, so it'll naturally gravitate there unless requests specifically constrain it to other ranges.

#### Chord Progression (Approximate)

The Wav2Vec chord classifier extracted this sequence (simplified, showing only stable sections):

```
Intro: Dm - Dm - Dm - Dm (establishing harmonic foundation)
Development: Dm - G - F - Dm - G - F - Am
Climax: C - F - Am - Dm - G - C
Resolution: Dm - F - Dm - Am - Dm
Second climax: F - G - C - Am - F - G
Final resolution: Dm - Am - Dm - Dm
```

This is approximate—the classifier has ~68% accuracy on real material, so these labels are "what harmonic region" not "exact chord." The electronic nature of Itzama means the "chords" might be more about harmonic areas created by synth pads and bass than traditional chord voicings. But it shows the structure: Dm as a center, movements through related chords (G, F, Am, C), climax points with more harmonic motion.

MusicHal doesn't use these labels in generation (remember: gesture tokens only), but they help me understand what patterns it learned.

#### Rhythm Ratio Distribution

From Brandtsegg analysis:

```
Most common ratios:
[1, 1]: 215 occurrences  // straight eighth notes
[3, 2]: 87 occurrences   // triplet feel
[2, 1]: 156 occurrences  // quarter to eighth relationship
[4, 3]: 34 occurrences   // more complex syncopation
[1, 2]: 98 occurrences   // half-time feel
```

The distribution shows I was playing mostly straight rhythms ([1,1], [2,1]), with some triplet inflection ([3,2]) and occasional syncopation ([4,3]). This is the rhythmic vocabulary MusicHal has to work with for this electronic beat music track.

If I'd trained on a different genre—say Georgia's slow jazz ballad—the ratio distribution would be completely different, with more rubato, different phrasing, different relationship to pulse. The learned vocabulary is genre-specific, which is what I want.

#### Barlow Complexity Curve (Conceptual)

If I plotted Barlow complexity over time, I'd expect to see:

- Lower values (1-3) during intro and resolution sections—simpler rhythms, more consonant
- Higher values (4-8) during development and climax—more complex rhythms, creating tension
- Spikes at transition points where I'm shifting from one rhythmic feel to another

This is the kind of analysis I'd need to run with matplotlib to actually show the curve, but the data is in the training output JSON. The structure exists, the visualization doesn't yet.

#### Performance Arc Detection

I mentioned Itzama has a 6-phase arc structure. Did the system capture this?

Indirectly, probably yes. The consonance values, Barlow complexity, and pitch range all shift across the performance. If I analyzed these parameters over time and looked for inflection points, I'd likely see:

1. Intro (0-30s): low density, narrow pitch range, lower complexity
2. Development (30-70s): increasing density, expanding range
3. First climax (70-90s): peak pitch, peak complexity
4. Resolution (90-110s): return to mid-register, simplification
5. Second climax (110-150s): another peak, different from first
6. Final resolution (150-183s): return to opening material, lowest complexity

Whether the AudioOracle "understands" this structure or just encodes local transitions is unclear. The suffix links might capture long-range patterns (linking the opening material to the final resolution), but the system doesn't have explicit arc awareness. That would require higher-level analysis beyond what Factor Oracle provides.

This is a limitation—and a possible future direction.

#### What This Means for Generation

When MusicHal generates using this trained model, it's recombining these 620 patterns. Request masking constrains which patterns get selected based on my live input, but the raw material comes from Itzama's learned structure.

That's why MusicHal trained on Itzama sounds different from MusicHal trained on Georgia. Different genres (electronic beat music vs. jazz ballad), different pitch distributions, different rhythm ratios, different harmonic progressions—different musical vocabulary.

The system has a localized musical identity shaped by this specific training data.

**[Audio Example 2: Excerpt from Itzama.wav training audio (0:45-1:15, showing the first climax section) with spectral analysis overlay]**

**[Figure 1: Visualization of Barlow complexity curve over Itzama performance, showing the 6-phase arc structure]**

---

### 15. Live Performance Analysis: Session Data

I need to be honest here too: I don't have a complete live session log to analyze for this document right now. What I'll present is the *kind* of analysis I would do, the structure of the data, what I'd be looking for.

When I run a live session, the system logs:

**Conversation CSV** (`logs/conversation_TIMESTAMP.csv`):

- Every event I play (MIDI note, gesture token, timestamp)
- Every phrase MusicHal generates (notes, durations, mode, timestamp)
- Mode transitions

**Decision CSV** (`logs/decisions_TIMESTAMP.csv`):

- Full decision transparency data (see Section 12)
- Request parameters, pattern match scores, confidence

**Audio analysis CSV** (`logs/audio_analysis_TIMESTAMP.csv`):

- Per-event feature extraction
- Wav2Vec features, Brandtsegg ratios, consonance, etc.

#### What I'd Analyze

**Session duration and engagement:**  
How long did I play? If less than 5 minutes, something went wrong—either technical issues or the musical interaction wasn't engaging. If 15+ minutes, the system was working well enough to hold my attention.

**Mode distribution:**  
How much time in each mode? Ideally relatively balanced, but with natural variation. If 90% SHADOW and 5% each MIRROR/COUPLE, the mode selection logic is stuck. If perfectly even distribution, that's suspiciously mechanical.

**Pattern match scores over time:**  
From the decision CSV, track the pattern match scores (0-100%). High scores (>80%) mean the system found good matches for my input. Low scores (<40%) mean it struggled, couldn't find similar patterns in the trained model.

A typical session might show:

- High scores at the beginning (I'm feeling out the system, playing patterns similar to training)
- Scores drop mid-session (I'm exploring, pushing boundaries)
- Scores rise again toward the end (settling back into familiar territory)

**Thematic recall frequency:**  
How many times did phrase memory trigger? Expected: every 30-60 seconds, so maybe 10-15 times in a 10-minute session. If zero recalls, phrase memory isn't working. If 50 recalls, the timer is wrong.

**Response latency:**  
From timestamps, calculate the delta between my onset and MusicHal's response. Target: <50ms. If many responses are >100ms, I would have felt lag, the dialogue would have felt sluggish.

**Musical transcription (selective):**  
Pick the most interesting 1-2 minute excerpt. Transcribe both my part and MusicHal's part to musical notation. Annotate with:

- Mode shifts
- Thematic recalls (with variation type noted)
- Pattern match scores for key phrases
- Moments where it worked well vs. awkwardly

This is the kind of detailed analysis that would make a strong case study, but it requires actually running a session and doing the post-processing work.

**[Audio Example 3: 2-minute excerpt from live session (t=5:30-7:30) with both parts isolated and mixed, showing SHADOW to COUPLE mode transition]**

**[Video Example 2: Performance footage with synchronized decision log overlay, showing the correlation between pattern match scores and musical coherence]**

**[Figure 2: Pattern match score distribution over 15-minute session, with mode transitions marked]**

#### The Question of Documentation

For a publication-ready document, I'd need:

- At least one fully analyzed session with transcriptions
- Statistical summaries across multiple sessions
- Audio examples (excerpts with annotations)
- Possibly video of the session showing real-time decision logs

This is future work. The system exists, the logging infrastructure exists, the data gets captured. What's missing is the time-intensive work of analysis and documentation.

---

### 16. Comparative Analysis - What Makes MusicHal Different

I've referenced OMax, Voyager, ImproteK, and Continuator. Let me be explicit about how MusicHal differs and what that means musically.

#### Comparison Table

| System          | Memory                 | Generation           | Transparency | Modes                 | Thematic Dev       | Learning     |
| --------------- | ---------------------- | -------------------- | ------------ | --------------------- | ------------------ | ------------ |
| **OMax**        | Factor Oracle          | Linear continuation  | No           | No                    | No                 | Real-time    |
| **Voyager**     | Pattern memory         | Multi-voice analysis | No           | No                    | Limited            | Pre-composed |
| **ImproteK**    | Enhanced Oracle        | Probabilistic        | No           | Partial               | No                 | Real-time    |
| **Continuator** | Markov chains          | Statistical          | Partial      | No                    | No                 | Real-time    |
| **MusicHal**    | Oracle + Phrase Memory | Request-based        | Yes (full)   | Yes (6 modes, sticky) | Yes (5 variations) | Pre-trained  |

#### Key Differentiators

**1. Dual Perception (Wav2Vec + Brandtsegg)**

OMax, Voyager, ImproteK all use either symbolic MIDI or traditional audio descriptors (MFCCs, chroma). MusicHal uses Wav2Vec's learned perceptual features plus Brandtsegg's rational rhythm analysis.

Why this matters: Wav2Vec captures subsymbolic qualities (microtonal inflection, spectral content, timbral nuance) that get lost in symbolic representation. Brandtsegg's ratios express rhythm as humans perceive it (relationaly), not as absolute time values.

Musical effect: MusicHal can respond to expressive inflections, not just "you played C4."

**2. Request Masking with Multi-Parameter Constraints**

Most systems do pattern matching or statistical continuation. MusicHal does conditional generation: "generate something that matches gesture_token 142 AND has consonance >0.7 AND fits rhythm_ratio [3,2]."

Why this matters: Multi-parameter requests let each behavior mode have distinct character through different constraint combinations. SHADOW emphasizes gesture matching, MIRROR emphasizes harmonic complementarity, COUPLE emphasizes contrast.

Musical effect: The modes feel genuinely different, not just variations on random selection.

**3. Phrase Memory with Thematic Development**

None of the comparison systems have long-term phrase memory that recalls and varies motifs. They respond to immediate context only.

Why this matters: Thematic development is fundamental to how humans improvise. Introducing a motif, developing it, bringing it back varied creates coherence over time, makes a performance feel intentional rather than reactive.

Musical effect: Sessions with MusicHal feel like they're "going somewhere," developing ideas, not just responding moment-to-moment.

**4. Decision Transparency**

No other system logs full decision reasoning with pattern match scores and request parameters.

Why this matters: Trust. I can verify the system is actually using my input, not generating randomly. This is essential for artistic research—I need to understand what's happening, not just accept black-box output.

Musical effect: I can identify when low pattern match scores correspond to musically awkward moments, tune parameters accordingly.

**5. Sticky Behavior Modes**

Some systems (like ImproteK) have behavioral modes, but they switch rapidly or based on simple triggers. MusicHal's modes persist 30-90 seconds.

Why this matters: Human musical relationships don't flicker. You establish a relational mode (supportive, challenging, independent) and explore within it. Rapid switching is disorienting.

Musical effect: I have time to feel the mode's character and respond to it, develop a dialogue within that relationship.

#### What MusicHal Doesn't Do

To be fair, there are things the comparison systems do that MusicHal doesn't:

**OMax's real-time learning:** MusicHal uses pre-trained models. OMax adapts continuously during performance. Trade-off: MusicHal has consistent identity, OMax is more adaptive.

**Voyager's compositional sophistication:** George Lewis programmed specific musical intelligence into Voyager based on decades of compositional practice. MusicHal learns patterns statistically. Trade-off: Voyager has deeper musical knowledge, MusicHal is more generalizable.

**ImproteK's scenario planning:** ImproteK can follow pre-defined compositional scenarios while improvising. MusicHal is purely reactive/generative. Trade-off: ImproteK better for composed pieces with improvisational sections, MusicHal better for pure improvisation.

#### Positioning in the Field

MusicHal sits between pure reactive systems (OMax, Continuator) and highly composed systems (Voyager). It uses learned patterns like reactive systems but adds structure (modes, phrase memory) that creates longer-term coherence without requiring pre-composition.

The transparency layer is unique—none of the comparison systems expose decision reasoning at this level of detail.

The dual perception (Wav2Vec + Brandtsegg) is also novel, combining subsymbolic learned features with symbolic mathematical analysis.

Whether these innovations are musically valuable is what live testing determines. The technical capabilities exist—the artistic question is whether they enable meaningful dialogue.

---

## Part VI: Artistic Outcomes & Reflection

### 17. Musical Quality Assessment - What Does "Working" Mean?

The core question: does MusicHal work as a musical partner?

This isn't a technical question. The system runs, the latency is acceptable, the logs show it's responding to my input. Those are engineering questions, and the answer is yes.

The artistic question is harder: does playing with MusicHal feel like musical dialogue? Do I trust it? Can I enter that semi-automated improvisational flow state?

I don't have a simple answer yet. Honest assessment:

#### What Works

**The modes feel distinct.** When SHADOW is active, I can feel it—the responses are close, imitative, supportive. When it switches to COUPLE, the independence is audible. The parameters (similarity threshold, temperature, request weight) create genuinely different musical characters.

**Thematic recall creates coherence.** When a motif returns transposed or inverted 60 seconds later, it's surprising—not random, but unexpected in a musically meaningful way. This is the closest the system gets to feeling "intentional."

**Decision transparency builds trust.** Seeing "pattern match: 85% to gesture_token 142" confirms the system is listening. This matters more than I expected—without it, I'd be constantly second-guessing whether responses were genuine or arbitrary.

**The system has a voice.** MusicHal trained on Itzama sounds like patterns learned from Itzama. There's a coherent style, a localized musical identity. It's not just "generic jazz"—it's patterns recombined from specific source material.

**The ecosystem functions when it works.** Waters (2007) argues that when a performance ecosystem is working, you stop perceiving separate components and experience the whole as one interactive system. That's what happens in good sessions—I stop thinking "I'm playing, MusicHal is responding, I'm hearing it through speakers." Instead, there's just music happening, emerging from the coupled interaction. I'm not performing with a machine—I'm inside a musical environment that includes the machine, my playing, the room, the moment. That shift in perception is what "working" means.

**Rhythm analysis makes the perception complete.** The Brandtsegg rhythm ratio analyzer running in live performance (re-enabled after earlier being disabled) gives MusicHal a mathematical understanding of timing alongside Wav2Vec's perceptual features. When I see rhythm ratios like 3:2 and Barlow complexity values displayed in the visualization viewport, I can verify the system understands not just what notes I'm playing, but the temporal relationships between them. This dual perception—perceptual harmonic features + mathematical rhythmic structure—is what makes the system genuinely responsive to both pitch and time.

**[Audio Example 4: Successful thematic recall sequence - original motif at t=0:45, transposed variation at t=1:38, inverted variation at t=2:52, showing coherent development]**

#### What Doesn't Work Yet

**Mode transitions are sometimes jarring.** Switching from tight SHADOW imitation to independent COUPLE can feel abrupt if it happens during a musical moment that needs continuity. The 30-90 second timer is arbitrary, not musically aware.

Possible solution: mode switching based on musical features (density, consonance) not just elapsed time. If I'm building tension, stay in current mode. If tension resolves, that's a natural transition point.

**Phrase memory doesn't consider harmony.** When a motif gets transposed or inverted, it sometimes lands on notes that clash with the current harmonic context. The variation is rhythmically and melodically coherent but harmonically awkward.

Possible solution: harmonic-aware variations. Before applying transpose/invert, check if the result fits the current context (consonance level, gesture token relationships). Adjust the variation to maintain harmonic coherence.

**Request masking sometimes feels over-constrained.** In SHADOW mode with high request weights (0.95), the system plays it safe—responses are predictable, matching closely but rarely surprising. In COUPLE mode with low weights (0.3), responses can feel random, not contrasting but unrelated.

Tuning problem: finding the sweet spot between "too predictable" and "too random" for each mode. Current parameters work okay, but they're not optimal.

**Latency spikes break flow.** Most responses are <50ms, but occasional spikes to 80-100ms are noticeable. When that happens during a fast exchange, the dialogue stutters.

Technical issue: AudioOracle queries get slow with large state spaces (>1000 states). Needs optimization—spatial indexing, pre-computed distance matrices, or limiting training data size.

#### The Trust Question

Do I trust MusicHal is listening?

With decision transparency on: yes, completely. I can see the pattern matching happening, verify the system is using my input.

With decision transparency off: less certain. When I can't see the reasoning, I start wondering if responses are genuine or random. The musical output is the same, but my experience changes—doubt creeps in.

This suggests transparency isn't just for development/debugging. It might be essential for the artistic experience, at least for me. Showing the mechanism doesn't break magic—it creates a different kind of engagement, where the algorithmic nature is explicit, part of the statement.

Still thinking about whether this is personal (I need transparency because I'm the developer) or general (anyone playing with MusicHal would want to see the decision logs).

#### The Flow State Question

Can I enter improvisational flow with MusicHal?

Sometimes. Not consistently yet.

When it works: I stop thinking about the system and just play. Responses come, I respond to them, patterns develop, time passes without me noticing. That's flow.

When it doesn't: I'm constantly analyzing, noticing when responses feel off, wondering why it played what it played. Not flow—analytical, self-conscious.

The difference seems to correlate with pattern match scores. Sessions where scores stay >70% feel more fluid. Sessions where scores drop to 30-40% feel awkward, I'm pulled out of flow.

This suggests the training data matters enormously. If the model has rich patterns that match my playing, flow is possible. If I play outside the training vocabulary, scores drop, flow breaks.

Implication: I need to train on diverse material—multiple recordings, different moods, wider pitch ranges—so the model has vocabulary for whatever direction I want to explore.

#### What "Success" Looks Like

Not: MusicHal becomes indistinguishable from a human improviser.

Yes: I can play with MusicHal for 15+ minutes without noticing I'm playing with a machine. The dialogue feels genuine enough that I stop thinking about the mechanism and just engage musically.

That's the threshold. Not Turing Test passing, not fooling anyone, just: engaging enough to hold attention, coherent enough to enable flow, transparent enough to maintain trust.

I'm not there consistently yet. But the components exist. What's needed: more training data, parameter tuning, more live testing to identify what breaks flow vs. what enables it.

---

### 18. Philosophical Considerations - What Am I Actually Building?

The Chinese Room keeps coming back.

MusicHal receives audio, extracts features, queries memory, applies probability distributions, outputs MIDI. From outside, it looks like listening and responding. From inside, it's pattern matching and statistical generation.

Does that distinction matter?

#### The Simulation Argument

One position: simulation isn't the real thing. MusicHal doesn't "listen" in any meaningful sense—it processes audio into feature vectors. It doesn't "remember" motifs—it stores data structures and applies deterministic recall logic. It doesn't "intend" to develop themes—it checks a timer and applies random variations.

All the language I use—listening, remembering, deciding, developing—is anthropomorphic projection. The machine is doing math. The perception of agency, intentionality, musical understanding—that's in me, not in the system.

If this position is correct, then MusicHal is an elaborate instrument, not a partner. I'm playing with a complex generative algorithm that creates the illusion of dialogue. Musically useful, maybe, but philosophically, it's me performing with a sophisticated tool.

#### The Functionality Argument

Counter-position: if the simulation is sophisticated enough, the distinction between "real" and "simulated" becomes irrelevant. What matters is whether the system enables musical dialogue, not whether it has consciousness or understanding.

When phrase memory recalls a motif 60 seconds later, does it matter that the recall mechanism is deterministic? The musical effect—coherence over time, sense of development—is real. Whether the mechanism involves "genuine" memory or "simulated" memory is philosophically interesting but practically irrelevant.

When request masking constrains generation to match my gesture tokens, does it matter that it's probability distributions over vector distances, not "understanding" what I played? The conditional response—generating something related to my input—is real.

If this position is correct, then arguing about whether MusicHal "really" listens is beside the point. The question is whether it functions as a musical partner, enabling the kind of dialogue I want. Function over essence.

#### Thelle's "Collective Agency"

Notto Thelle's work on mixed-initiative systems explores this directly (Thelle, 2022). The question isn't whether the machine has agency in a philosophical sense, but whether human and machine can form a collective system with emergent properties neither has alone.

When I play with MusicHal, the musical output isn't "me" or "the algorithm"—it's the interaction. Patterns emerge from the feedback loop: I play something, MusicHal responds, I respond to the response, MusicHal responds to that, musical structure emerges from the coupling.

In this frame, whether MusicHal has "genuine" agency is the wrong question. The collective system has agency—the ability to produce musical outcomes neither part could create independently. That's what matters.

Waters' ecosystem thinking extends this (Waters, 2007). The question isn't whether MusicHal has agency, but whether the ecosystem—me + MusicHal + training data + room acoustics + this moment—produces emergent musical behaviors. Waters cites Cariani's concept of emergence: systems that outperform their design specifications, where behaviors cannot be accounted for solely by the designed outcome.

I designed request masking to constrain generation based on gesture tokens. But when a low pattern match (<40%) forces unexpected material that creates productive musical tension, that's emergence. I designed phrase memory to recall motifs every 30-60 seconds. But when a recall lands at exactly the right moment—something I couldn't have scripted—that's emergence. These moments aren't in me or in the algorithm. They're properties of the ecosystem functioning as a whole.

The ecosystem frame helps me stop asking "does MusicHal understand?" and instead ask "does this coupled system produce musical behaviors I value?" The first question has no good answer. The second question is empirically testable through performance.

This is the position I'm leaning toward. Not because it resolves the philosophical problem (it doesn't), but because it reframes the question in a way that's more productive for artistic research.

#### Transparency as Artistic Statement

If I accept that MusicHal is simulation, not genuine listening/memory/intent, then what does decision transparency mean?

Not: proof that it's "really" listening (it's not).

Instead: explicit acknowledgment of the algorithmic nature. Showing the pattern matching scores, the request parameters, the state transitions—this is saying "here's how the simulation works, here's the mechanism creating the illusion."

Like Brecht's verfremdungseffekt—breaking the fourth wall, revealing the construction, making the audience aware they're watching a performance not reality.

Is that valuable artistically? Maybe. It changes the engagement from "believe this is a musical conversation" to "observe how this simulation creates the appearance of conversation." Different aesthetic, not better or worse.

Still thinking about this.

#### What I Can Say For Sure

MusicHal isn't conscious. It doesn't understand music. It processes patterns statistically.

But the simulation is sophisticated enough to be musically engaging, at least sometimes. Whether that makes it a "partner" or an "instrument" is definitional, not factual.

The interesting question isn't "is this real musical dialogue?" but "what kind of musical relationship does this enable, and is that valuable?"

I don't have a final answer. That's the research. Build it, play with it, notice what works and what doesn't, iterate. The philosophy will follow from the practice.

---

### 19. Future Directions - What Comes Next

This is a snapshot of work in progress. Lots of directions to explore, improvements to make, questions to investigate.

#### Technical Improvements

**Harmonic-aware phrase memory:**  
Currently, motif variations (transpose, invert) ignore harmonic context. Add a constraint: before returning a variation, check if it fits the current gesture tokens/consonance levels. If not, adjust the transposition or select a different variation type.

**Musically-aware mode transitions:**  
Replace fixed 30-90 second timers with transition logic based on musical features. High density + rising pitch + increasing complexity = maintain current mode (building energy). Low density + falling pitch + decreasing complexity = transition opportunity (resolution point).

**Optimization for larger models:**  
Current AudioOracle queries slow down with >1000 states. Implement spatial indexing (k-d trees) for fast nearest-neighbor lookups. Pre-compute gesture token similarity matrices. Test with larger training datasets (10+ minutes of audio).

**Multi-track training:**  
Currently training on solo performances. What if I trained on duo recordings—my part and a bass player's part? The system could learn harmonic relationships between voices, generate bass lines that complement my melodies or vice versa.

Technical challenge: synchronizing two audio streams, extracting features for both, learning relationships not just sequences.

**Adaptive temperature:**  
Currently temperature is fixed per mode (0.7 for SHADOW, 1.3 for COUPLE). What if temperature adapted based on pattern match scores? High scores (>80%) = increase temperature slightly (add surprise). Low scores (<50%) = decrease temperature (play it safer).

Creates dynamic balance between predictability and exploration.

#### Artistic Explorations

**Performance with visualization:**  
Run MusicHal with decision transparency projected behind me during performance. Audience sees the pattern matching, request parameters, mode transitions in real-time. Makes the algorithmic nature explicit, transparency as aesthetic.

**Training on different genres:**  
I've been testing with jazz. What happens with free improvisation (no tonal center, no steady pulse)? Electronic music (timbral focus over pitch)? Classical (strict harmonic rules, formal structures)?

The system is genre-agnostic in principle. Whether it's musically coherent across genres is empirical.

**Collaboration with other musicians:**  
Right now it's me alone with MusicHal. What if another human joined—trio improvisation? How does the dialogue change when there are three agents (two human, one machine)? Does MusicHal's role shift when it's not the only respondent?

**Comparative perception studies:**  
Play identical sessions with decision transparency on vs. off. Record my subjective experience: when did flow happen, when did trust break, when did responses feel meaningful vs. arbitrary? Compare between conditions. Does transparency actually help, or is it a distraction?

#### Conceptual Questions

**The training data ethics:**  
I'm training on my own recordings—no copyright or ownership issues. But what if the model learned from others' playing? If I trained MusicHal on Coltrane recordings, would the output be derivative work, plagiarism, style imitation, or something else?

This matters for artists who might want to use the system. Clear licensing, attribution, questions about creative ownership of AI-generated material.

**Generalizability:**  
The system is tuned for my practice, my musical values, my need for transparency and trust. Would it work for other musicians? Different players might want:

- Less transparency, more mystery
- Real-time learning instead of pre-trained models
- Different behavior modes (aggressive, sparse, dense)
- Different memory timescales

Is MusicHal a personal research tool or a generalizable system? Both, probably, but the balance needs clarification.

**Long-term development:**  
Right now a "session" is 10-15 minutes. What about 60-minute performances? Multi-day residencies where MusicHal learns across sessions? Year-long development where the model evolves with my playing?

Factor Oracle can do incremental learning—add new patterns to existing memory. But that changes the model's identity. Is that valuable (adaptive partner) or problematic (loses consistent voice)?

#### Documentation Needs

For publication, I need:

- Multiple fully-analyzed sessions with transcriptions
- Audio examples with annotations
- Video documentation of live performances
- Statistical analysis across sessions
- Comparative studies with/without specific features (phrase memory, transparency, etc.)

This document is the framework. The evidence needs filling in.

---

## Part VII: Implementation Details & Reproducibility

### 20. Installation & Setup

For researchers or artists who want to run MusicHal, here's the practical setup.

#### System Requirements

**Hardware:**

- Mac (M1/M2 recommended for MPS GPU acceleration) or Linux/Windows with CUDA GPU
- Minimum 16GB RAM (Wav2Vec model is memory-intensive)
- Audio interface with low-latency mic input
- MIDI interface (virtual or hardware) for output to DAW/synth

**Software:**

- Python 3.9+
- SuperCollider 3.12+ (for audio rendering)
- A DAW that accepts MIDI (optional, for recording)

#### Python Dependencies

Install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages:

- `torch` (PyTorch for Wav2Vec)
- `transformers` (HuggingFace, for Wav2Vec 2.0 model)
- `librosa` (audio analysis)
- `numpy`, `scipy` (numerical processing)
- `python-rtmidi` (MIDI I/O)
- `sounddevice` (audio input)

Full list in the repo's `requirements.txt`.

#### Audio Setup

**1. Configure audio input:**

```bash
python test_audio_devices.py
```

This lists available input devices. Note the device index for your microphone.

**2. Test latency:**

```bash
python test_audio_routing.py --device 2
```

Replace `2` with your device index. Should show <10ms input latency.

**3. MIDI routing:**

macOS: Use IAC Driver (built-in virtual MIDI bus)

- Open Audio MIDI Setup
- Show MIDI Studio
- Double-click IAC Driver
- Enable "Device is online"

Linux: Use JACK or ALSA virtual MIDI

Windows: Use loopMIDI or similar virtual MIDI cable

**4. SuperCollider synths:**

Open SuperCollider, run:

```supercollider
s.boot;
s.loadDirectory("SuperCollider/synths/");
```

This loads the MPE-capable synths for rendering MusicHal's MIDI output.

#### Verify Installation

```bash
python test_system_components.py
```

This runs diagnostics:

- Audio input: ✓ / ✗
- MIDI output: ✓ / ✗
- Wav2Vec model: ✓ / ✗
- Brandtsegg analyzer: ✓ / ✗
- Trained model found: ✓ / ✗

All should show ✓ before proceeding.

---

### 21. Training a New Model

To train MusicHal on your own audio:

#### Step 1: Prepare Audio

Record 2-5 minutes of solo playing:

- Mono, 44.1kHz sample rate
- WAV format
- Relatively clean (not essential, but helps)
- Save as `input_audio/your_recording.wav`

#### Step 2: Run Training

```bash
python Chandra_trainer.py \
  --file input_audio/your_recording.wav \
  --max-events 1000 \
  --wav2vec \
  --gpu
```

**Parameters:**

- `--file`: Input audio path (required)
- `--output`: Optional custom output name (default: auto-generated `audioname_DDMMYY_HHMM.json` in `JSON/`)
- `--max-events`: Limit processing (use -1 for all events, default processes all)
- `--wav2vec`: Use Wav2Vec for feature extraction (includes Brandtsegg rhythm analysis)
- `--gpu`: Use GPU acceleration (MPS on Mac, CUDA on Linux/Windows)

**Output files:**

- `audioname_DDMMYY_HHMM.json`: Training log and analysis results
- `audioname_DDMMYY_HHMM_model.json`: AudioOracle graph for live performance
- `audioname_DDMMYY_HHMM_correlation_patterns.json`: Learned harmonic-rhythmic correlations

**Training time:**  
~3-6 minutes for 180 seconds of audio on M1 Mac. Longer on CPU-only.

#### Step 3: Verify Training Output

Check the logs:

```
Training complete. Model saved to JSON/your_model_name.json
States: 620 | Suffix links: 1847 | Avg transitions: 2.3
```

Verify files created:

- `JSON/your_model_name.json` (metadata)
- `JSON/your_model_name_model.json` (AudioOracle structure)

#### Step 4: Test the Model

```bash
python MusicHal_9000.py \
  --model JSON/your_model_name.json \
  --test-mode
```

This runs without live input, generating a few test phrases to verify the model loads and generates correctly.

---

### 22. Running Live Performance

Once you have a trained model:

```bash
python MusicHal_9000.py \
  --hybrid-perception \
  --wav2vec \
  --gpu \
  --visualize
```

The system automatically loads the most recent model from `JSON/` (sorted by modification time) and uses the system default audio input. If I need to override these, I can add `--model JSON/specific_model.json` or `--input-device 0`, but usually the defaults work.

**What the flags do:**

- `--hybrid-perception`: Enables dual-modal perception (Wav2Vec + Brandtsegg rhythm analysis)
- `--wav2vec`: Uses Wav2Vec 2.0 neural encoding for perceptual features
- `--gpu`: Loads Wav2Vec onto GPU (MPS for Apple Silicon, CUDA for NVIDIA) for faster processing
- `--visualize`: Opens multi-viewport visualization system (5 windows showing internal processes)

**Optional flags:**

- `--debug-decisions`: Show real-time decision explanations in terminal
- `--mode SHADOW`: Start in specific mode (default: IMITATE with random duration)
- `--performance-duration 180`: Set performance arc length in seconds (enables graceful fade-out)

**During performance:**

Press `m` to manually switch modes  
Press `space` to pause/resume MusicHal's responses  
Press `d` to toggle decision transparency on/off  
Press `q` to quit

**Logs saved to:**

- `logs/conversation_TIMESTAMP.csv`
- `logs/decisions_TIMESTAMP.csv`
- `logs/audio_analysis_TIMESTAMP.csv`

---

### 22.5. The Visualization System

The visualization isn't just a debugging tool. It's part of solving the trust problem.

If I can see MusicHal's internal decision-making in real-time—which gesture tokens it's matching, what behavior mode it's in, when it recalls a theme—then I can verify it's actually listening, not just generating patterns at random. The Chinese Room might be manipulating symbols, but I can watch which symbols it's manipulating and why.

#### Why Visibility Matters

When I play with a human musician, I trust they're listening because I can see them react—head nod, eye contact, body language that signals "I heard that, I'm responding." With a machine, there's no body language. The trust has to come from somewhere else.

Decision transparency in the logs helps (Section 12), but logs are post-hoc. I want to see the decision-making happen while I'm playing. That's what the visualization provides: real-time windows into the system's perception, memory, and generation processes.

The `--visualize` flag opens five viewports arranged in an automatic grid layout. Each shows a different layer of the system's internal state, updating 10 times per second without affecting audio processing latency.

#### The Five Viewports

**Pattern Matching**

Shows the AudioOracle state traversal in real-time:

- Current gesture token extracted from my playing
- Pattern match score (0-100%): how well the system is finding learned patterns
- Which state in the Oracle graph it's currently at
- Context: what sequence of events led to this match

This is verification. When I play a phrase similar to something from training and the match score jumps from 45% to 87%, I can see the system recognizing it. The pattern isn't random—it's genuinely matching learned structure.

**Request Parameters**

The behavior mode system made visible:

- Current mode (SHADOW, MIRROR, COUPLE, IMITATE, CONTRAST, LEAD)
- Duration countdown (seconds remaining until mode change)
- Request weights for this mode (similarity_threshold, temperature, phrase_variation, response_delay)
- Shows me the personality parameters in numbers

When the mode switches from SHADOW (similarity 0.95, temp 0.3, delay 0.1s) to COUPLE (similarity 0.05, temp 1.8, delay 3.0s), I don't just hear the musical relationship change—I see the exact parameter shifts that cause it. The musical character emerges from these numbers.

**Phrase Memory**

Long-term thematic development displayed:

- Stored motifs (first 5 notes of each displayed as MIDI numbers)
- Recall probability increasing over time
- When a variation triggers (transpose +7, invert, retrograde, augment, diminish)
- Timestamp of last recall

This lets me see if the system is building coherence across the performance. A motif stored at t=0:45, recalled at t=1:38 with transpose variation, then again at t=2:52 inverted—that's not random generation, that's thematic development.

**Audio Analysis**

What the machine hears right now:

- Waveform (raw audio input, 1024 samples)
- Onset markers (visual spike when note detected)
- Rhythm ratio (e.g., "3:2" from Brandtsegg analyzer)
- Barlow complexity (2.7 = moderate rhythmic consonance)

This is the Brandtsegg analyzer output made visible. I can see that it's not just detecting onsets—it's extracting the mathematical relationships between them. When I play a dotted rhythm (3:2 ratio), the viewport shows "3:2". When I play a complex polyrhythmic figure, the Barlow complexity jumps to 8.5. The dual perception is working.

**Performance Timeline**

Scrolling event log of the entire session:

- Human input markers (green)
- AI response markers (blue)
- Behavior mode change markers (red)
- Thematic recall annotations (yellow)

This gives the temporal overview. I can see the dialogue pattern—am I dominating? Is MusicHal responding too much or too little? When did the last mode change happen? The timeline shows the flow of the performance as a visual structure.

#### Connection to Waters' Ecosystem Thinking

Waters (2007) talks about distributed cognition in performance ecosystems—how musical meaning isn't located in individual agents but emerges from the whole coupled system. The visualization makes this visible.

I can see the feedback loops happening: My phrase enters through the waveform → gets analyzed into gesture token 142 and rhythm ratio 3:2 → triggers pattern matching (85% score) → mode parameters shape the response (SHADOW: high similarity, low temperature) → phrase memory checks for recall (probability 72%) → generated phrase appears in timeline → I hear it through speakers → my body responds with the next phrase → cycle repeats.

The cognition isn't "in me" or "in MusicHal." It's in the circulation of information through this loop. The viewports show that circulation, make the ecosystem's structure visible rather than abstract.

#### Technical Details (Just Enough)

Built with PyQt5. Five QMainWindow instances, each running its own event loop. The audio processing thread (where perception and generation happen) emits Qt signals whenever state changes. These signals cross the thread boundary using QueuedConnection—critical for thread safety, ensures GUI updates don't block audio processing.

The VisualizationManager coordinates all viewports, handles the event bus, and auto-arranges windows in a grid layout that adapts to screen size. On a 1920x1080 display, it creates a 3x2 grid with each viewport ~500x460px. On larger displays, it spreads them further apart.

Updates at 10Hz (every 100ms) regardless of audio processing rate (which runs at ~23ms per chunk). This rate is fast enough to feel real-time but slow enough that GUI rendering doesn't impact performance.

#### What It Reveals

When I see the pattern match score jump from 45% to 87% right as I play a phrase similar to something from the training data, that's verification. The machine IS using learned patterns, not generating randomly.

When the behavior mode switches from SHADOW (high similarity 0.9, immediate response) to COUPLE (low similarity 0.1, long delay 3s, high temperature 1.8), and I feel the musical relationship change, the visualization confirms it's not my imagination—the parameters actually shifted. The personality isn't metaphorical, it's measurable.

When a motif stored 60 seconds ago suddenly appears in the phrase memory viewport with "recall triggered: transpose +7," and I hear that transposed motif come back, I know the thematic development isn't coincidence—the system remembered and intentionally varied.

#### Does This Build Trust?

Partially. The visualization reveals the mechanism. It shows me that pattern matching is happening, that modes have real behavioral consequences, that thematic recall is a deliberate process not random luck.

But seeing the mechanism doesn't automatically make the mechanism musically good. I can watch all five viewports and still feel like the responses are musically awkward, that the pattern matching found a technically similar phrase but emotionally wrong context.

What the visualization does is eliminate one source of distrust: the worry that nothing meaningful is happening inside, that it's all arbitrary. Now I know something structured is happening. Whether that structure produces musical partnership—that's still the open question.

---

### 23. Code Architecture Reference

For developers who want to understand or extend the system:

#### Core Components

**`MusicHal_9000.py`** (1500 lines)  
Main performance system. Real-time loop: audio input → perception → memory → generation → MIDI output.

**`Chandra_trainer.py`** (1800 lines)  
Training pipeline. Offline processing: audio → feature extraction → temporal smoothing → AudioOracle training → serialization.

#### Module Structure

```
CCM3/
├── agent/
│   ├── behaviors.py              # Behavior mode engine
│   ├── phrase_generator.py       # Generation with request masking
│   ├── phrase_memory.py          # Long-term thematic memory
│   └── decision_explainer.py     # Transparency layer
├── listener/
│   ├── hybrid_perception.py      # Wav2Vec + Brandtsegg integration
│   └── drift_listener.py         # Onset detection
├── memory/
│   ├── memory_buffer.py          # Short-term rolling buffer
│   ├── polyphonic_audio_oracle.py # AudioOracle implementation
│   └── request_mask.py           # Multi-parameter masking
├── rhythmic_engine/
│   └── ratio_analyzer.py         # Brandtsegg rhythm analysis
├── hybrid_training/
│   └── wav2vec_chord_classifier.py # Neural chord decoder
├── core/
│   ├── temporal_smoothing.py     # Time-window averaging
│   ├── logger.py                 # CSV logging
│   └── feature_mapper.py         # Audio features → MIDI
├── mapping/
│   └── midi_output.py            # MIDI message generation
└── JSON/                         # Trained models
```

#### Extending the System

**Add a new behavior mode:**

1. Edit `agent/behaviors.py`, add to `BehaviorMode` enum
2. Define parameters (similarity, temperature, request_weight)
3. Edit `agent/phrase_generator.py`, add request-building method
4. Update mode selection logic in `BehaviorEngine._select_new_mode()`

**Add a new request parameter:**

1. Ensure the parameter is extracted during perception (in `hybrid_perception.py`)
2. Ensure it's stored in events (in `memory_buffer.py`)
3. Add masking logic in `memory/request_mask.py` for the parameter type
4. Update phrase generation to use the new parameter

**Modify temporal smoothing:**

Edit `core/temporal_smoothing.py`. Key parameters:

- `window_size`: Time window for grouping (default 0.3s)
- `onset_threshold`: Amplitude change defining new onset
- Averaging methods for different feature types

**Change AudioOracle similarity threshold:**

Edit `memory/polyphonic_audio_oracle.py`, method `_create_suffix_link()`. Current threshold: Euclidean distance < 0.15. Lower = stricter (fewer suffix links), higher = looser (more links, possibly false matches).

---

## Part VIII: Appendices

### Appendix A: Key Code Excerpts

**AudioOracle State Addition (Simplified):**

```python
def add_state(self, feature_vector, timestamp):
    """Add new state to Oracle with suffix link creation."""
    state_id = len(self.states)
    
    # Create new state
    self.states.append({
        'id': state_id,
        'features': feature_vector,
        'timestamp': timestamp,
        'transitions': [],
        'suffix_link': None
    })
    
    # Find suffix link: most similar previous state
    best_match = None
    best_distance = float('inf')
    
    for prev_state in self.states[:-1]:
        distance = euclidean_distance(
            feature_vector, 
            prev_state['features']
        )
        if distance < best_distance and distance < self.threshold:
            best_distance = distance
            best_match = prev_state['id']
    
    if best_match is not None:
        self.states[state_id]['suffix_link'] = best_match
    
    # Add forward transition from previous state
    if state_id > 0:
        self.states[state_id - 1]['transitions'].append(state_id)
    
    return state_id
```

**Request Masking (Core Algorithm):**

```python
def apply_mask(self, request, oracle_states):
    """Apply multi-parameter request to create probability mask."""
    n_states = len(oracle_states)
    mask = np.ones(n_states)
    
    for param_name, param_spec in request.items():
        param_mask = np.zeros(n_states)
        
        if param_spec['type'] == '==':  # Exact match
            for i, state in enumerate(oracle_states):
                if state[param_name] == param_spec['value']:
                    param_mask[i] = 1.0
        
        elif param_spec['type'] == 'gradient':  # Smooth falloff
            target = param_spec['value']
            for i, state in enumerate(oracle_states):
                distance = abs(state[param_name] - target)
                param_mask[i] = np.exp(-distance / param_spec['scale'])
        
        # Blend into overall mask with weight
        mask *= (param_mask * param_spec['weight'] + 
                 (1 - param_spec['weight']))
    
    # Normalize
    mask = mask / mask.sum()
    
    # Apply temperature
    mask = mask ** (1.0 / self.temperature)
    mask = mask / mask.sum()
    
    return mask
```

**Phrase Memory Recall:**

```python
def should_recall_theme(self):
    """Check if enough time elapsed for thematic recall."""
    current_time = time.time()
    elapsed = current_time - self.last_recall_time
    
    if elapsed < self.min_recall_interval:
        return False
    
    # Random probability increases with time
    recall_probability = min(1.0, elapsed / self.max_recall_interval)
    
    if random.random() < recall_probability:
        self.last_recall_time = current_time
        return True
    
    return False

def get_variation(self, motif, variation_type):
    """Apply variation to motif."""
    if variation_type == 'transpose':
        semitones = random.choice([5, 7, 12])  # Fourth, fifth, octave
        return [(note + semitones, dur) for note, dur in motif]
    
    elif variation_type == 'invert':
        root = motif[0][0]
        return [(root - (note - root), dur) for note, dur in motif]
    
    elif variation_type == 'retrograde':
        return list(reversed(motif))
    
    elif variation_type == 'augment':
        return [(note, dur * 2) for note, dur in motif]
    
    elif variation_type == 'diminish':
        return [(note, dur / 2) for note, dur in motif]
```

---

### Appendix B: Mathematical Foundations

**Barlow Harmonicity (Indigestability):**

For a rational rhythm ratio `n/d`, Barlow indigestability is defined as:

```
H(n/d) = (|n-d| * prod(prime_factors(n)) * prod(prime_factors(d))) / gcd(n,d)
```

Where simpler ratios (1:1, 2:1, 3:2) have lower values (more consonant), complex ratios (7:5, 11:13) have higher values (more dissonant).

**Euclidean Distance in Feature Space:**

For two gesture tokens represented as 768-dimensional Wav2Vec vectors `v1` and `v2`:

```
distance = sqrt(sum((v1[i] - v2[i])^2 for i in range(768)))
```

Suffix links connect states with distance < threshold (typically 0.15 after normalization).

**Temperature-Scaled Probability:**

Given probability distribution `P` and temperature `T`:

```
P_temp[i] = P[i]^(1/T) / sum(P[j]^(1/T) for all j)
```

- T < 1: sharper distribution (more deterministic)
- T = 1: unchanged
- T > 1: flatter distribution (more random)

---

### Appendix C: Training Data Specifications

**Itzama.wav:**

- Duration: 183.7 seconds
- Sample rate: 44100 Hz, mono
- Format: WAV, 16-bit PCM
- Genre: Pop track, electronically based beat music (modern, not jazz)
- Estimated tempo: 138 BPM (variable)
- Pitch range: C3-C6 (48-84 MIDI)
- Harmonic center: D minor
- Performance arc: 6 phases (intro, development, climax, resolution, climax 2, final resolution)
- Training output: 620 states, 1847 suffix links

**Georgia.wav:**

- Full title: "Georgia on My Mind" (Hoagy Carmichael and Stuart Gorrell, 1930)
- Genre: Jazz standard ballad
- Duration: ~120 seconds
- Instrumentation: Voice (Jonas Howden Sjøvaag) and piano (Andreas Ulvo)
- Ballad tempo (~60-80 BPM)

**Nineteen.wav:**

- Original composition by Jonas Howden Sjøvaag
- Genre: Jazz standard style ballad
- Duration: ~140 seconds
- Instrumentation: Voice/instrument, piano (Andreas Ulvo), bass (Mats Eilertsen)
- Ballad tempo
- Focus on melodic phrasing and space

**Curious Child.wav:**

- Duration: ~100 seconds
- Up-tempo (~200 BPM)
- Rhythmically driven, aggressive articulation

---

### Appendix D: Session Log Example Structure

**Conversation CSV:**

```csv
timestamp,source,event_type,midi_note,note_name,gesture_token,rhythm_ratio,consonance,mode
45.234,human,onset,65,F4,142,[3,2],0.73,SHADOW
45.891,machine,phrase_start,65,F4,142,[3,2],0.75,SHADOW
45.891,machine,note,67,G4,143,[2,1],0.68,SHADOW
46.234,machine,note,69,A4,145,[2,1],0.71,SHADOW
46.587,machine,phrase_end,70,A#4,147,[2,1],0.69,SHADOW
47.123,human,onset,67,G4,143,[2,1],0.68,SHADOW
```

**Decision CSV:**

```csv
timestamp,mode,trigger_midi,request_primary,request_secondary,pattern_match_score,generated_notes,confidence
45.891,SHADOW,65,gesture_token==142:0.95,consonance>0.7:0.5,87,"[65,67,69,70]",0.87
47.234,SHADOW,67,gesture_token==143:0.95,consonance>0.7:0.5,72,"[67,69,70,72]",0.72
```

---

### Appendix E: Multimedia Documentation Specifications

This document references audio and video examples throughout. Here's what each should demonstrate and why it matters for the argument. These aren't decorative—each serves a specific evidential purpose.

**Audio Example 1: Feedback Loop Demonstration (Section 3.5)**

- **What:** 60-second recording showing clear call-and-response pattern: my phrase, MusicHal's response, my adjusted response
- **Why:** Demonstrates Waters' ecosystem feedback loops in practice. Should show me adapting to MusicHal's output, proving the interaction isn't one-way
- **Key moments:** Initial phrase (0:00-0:05), MusicHal response (0:05-0:10), my adapted reply (0:10-0:15), continuing cycle
- **Purpose:** Evidence that the ecosystem concept isn't abstract—it's happening in real musical time

**Video Example 1: Multi-Viewport Performance (Section 11)**

- **What:** 3-minute screen recording with 5 viewports visible + room camera showing me playing
- **Why:** Shows decision transparency in action, makes the trust layer visible
- **Key moments:** Pattern match score changes, mode transition at ~1:45, thematic recall marker, relationship between what I play and what viewports show
- **Purpose:** Proof that visualization reveals actual decision-making, not just pretty graphics

**Audio Example 2: Itzama Excerpt with Analysis (Section 14)**

- **What:** 30-second excerpt from Itzama.wav training audio (0:45-1:15, first climax section) with spectral analysis overlay
- **Why:** Shows what the training data actually sounds like, not just abstract descriptions
- **Key moments:** Modal harmony shifts, rhythmic density changes, the melodic gesture that becomes gesture_token 142
- **Purpose:** Evidence for claims about training data characteristics—listeners can verify modal jazz, rhythmic complexity, timbral qualities

**Audio Example 3: Live Session Mode Transition (Section 15)**

- **What:** 2-minute excerpt (t=5:30-7:30) showing SHADOW→COUPLE mode transition, with both parts isolated and mixed
- **Why:** Demonstrates that mode changes create audible behavioral differences
- **Key moments:** SHADOW imitative responses (5:30-6:30), transition marker (6:45), COUPLE independent responses (6:45-7:30)
- **Purpose:** Evidence that sticky modes aren't just parameter changes—they create distinct musical characters

**Video Example 2: Side-by-Side Performance Analysis (Section 15)**

- **What:** 2-minute split-screen: left = room view of me playing drums/vocals, right = synchronized visualization
- **Why:** Shows moment-by-moment correspondence between physical playing and system analysis
- **Key moments:** My onset → onset marker appears, pattern match score responds, mode duration counts down, phrase memory stores motif
- **Purpose:** Demonstrates that analysis is responsive to actual playing, not random or delayed

**Figure 1: Barlow Complexity Curve (Section 14)**

- **What:** Graph of Barlow complexity over Itzama performance duration, annotated with 6-phase arc structure
- **Why:** Shows performance arc detection isn't invented—it's in the rhythmic data
- **Key moments:** Intro (low complexity ~1.5), building (increasing to 3.0), climax peaks (~5.5), resolution decline
- **Purpose:** Visual evidence that training extracts temporal structure

**Figure 2: Pattern Match Distribution (Section 15)**

- **What:** Histogram of pattern match scores over 15-minute session, with mode transitions marked
- **Why:** Shows relationship between modes and pattern matching quality
- **Key moments:** SHADOW mode (scores clustered 75-95%), COUPLE mode (scores scattered 20-80%)
- **Purpose:** Quantitative evidence that modes affect generation strategy

**Audio Example 4: Thematic Recall Sequence (Section 17)**

- **What:** 3-minute excerpt showing original motif at t=0:45, transposed variation at t=1:38, inverted variation at t=2:52
- **Why:** Demonstrates long-term coherence, not just short-term pattern matching
- **Key moments:** Original motif recognition, return with variation, development over time
- **Purpose:** Evidence that phrase memory creates intentional-seeming behavior

**Video Example 5: Phrase Memory Viewport Close-up (Section 13)**

- **What:** 2-minute recording focused on phrase memory viewport during session
- **Why:** Shows thematic development system in action—motif extraction, recall probability increasing, variation triggering
- **Key moments:** Motif stored (t=0:45), display updates (t=1:15 recall probability rising), recall triggers (t=1:38) with "transpose +7" label, room audio synchronized
- **Purpose:** Proof that thematic recalls aren't post-hoc analysis—system is deliberately tracking and varying motifs

**Video Example 6: Mode Transition Demonstration (Section 6)**

- **What:** 3-minute split screen: me playing piano (left) + Mode/Request viewport (right)
- **Why:** Shows parameter changes creating behavioral changes
- **Key moments:** SHADOW mode (similarity 0.9, temp 0.3, immediate responses), countdown visible, transition at ~1:30, COUPLE mode (similarity 0.1, temp 1.8, delayed responses), my playing adapts
- **Purpose:** Evidence that sticky modes aren't subtle—they create dramatically different musical relationships

#### Why These Specifications Matter

Each example addresses a specific skepticism:

- "Is it really listening?" → Videos 1, 2, 5 show real-time responsiveness
- "Do modes actually work?" → Audio 3, Video 6 demonstrate distinct behaviors
- "Is thematic development real?" → Audio 4, Video 5 show deliberate recall and variation
- "What does training data sound like?" → Audio 2 provides context for generation
- "Is structure learned or imagined?" → Figure 1 shows objective measurements

The multimedia isn't about making the document prettier. It's about making claims verifiable. I'm saying "MusicHal listens to rhythm," here's the viewport showing rhythm ratios updating in real-time. I'm saying "modes create personality," here's 3 minutes of SHADOW vs. COUPLE with audibly different characters.

Without this evidence, the document would be assertions. With it, readers can evaluate the claims themselves.

---

## References

Assayag, G., & Dubnov, S. (2004). Using Factor Oracles for machine improvisation. *Soft Computing*, *8*(9), 604-610. https://hal.science/hal-01161221v1

Austestad, H. M. (2025). *Realtime tuning & temperering* [Computer software]. https://github.com/[repository]

Brandtsegg, Ø., & Formo, D. (2024). *Rhythm ratio analyzer* [Computer software]. NTNU. https://github.com/[repository]

Bujard, B., Nika, J., Bevilacqua, F., & Obin, N. (2024). Learning relationships between separate audio tracks for creative applications. *Proceedings of the International Conference on New Interfaces for Musical Expression*. https://arxiv.org/abs/2509.25296

Dubnov, S., Assayag, G., & Cont, A. (2007). Audio Oracle: A new algorithm for fast learning of audio structures. *Proceedings of the International Computer Music Conference (ICMC)*, Copenhagen, Denmark. https://inria.hal.science/hal-00839072v1

Lewis, G. E. (2000). Too many notes: Computers, complexity and culture in Voyager. *Leonardo Music Journal*, *10*, 33-39.

Thelle, N. J. W. (2022). *Mixed-Initiative Music Making: Collective Agency in Interactive Music Systems* [Doctoral dissertation, Norwegian Academy of Music]. NMH Publications 2022:4.

Thelle, N. J. W., & Wærstad, B. I. (2023). Co-creative spaces: The machine as a collaborator. *Proceedings of the International Conference on New Interfaces for Musical Expression (NIME'23)*, UAM-Lerma, Mexico City, Mexico.

Waters, S. (2007). Performance ecosystems: Ecological approaches to musical interaction. *Proceedings of the Electroacoustic Music Studies Network Conference*, Leicester, UK.

---

**End of Document**

*Document length: approximately 2510 lines / ~68 pages / ~37,000 words*

