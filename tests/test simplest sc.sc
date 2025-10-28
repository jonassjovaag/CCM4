// Boot the server
s.boot;

// Server configuration
s.options.numBuffers = 1024;
s.options.memSize = 8192;
s.waitForBoot({
    thisProcess.openUDPPort(10111);
    "Ready for OSC input on port 10111".postln;
});

// Create sine wave synth definition with ADSR
SynthDef(\voiceSynth, {
    arg freq=440, amp=0.5, gate=1, pan=0,
    attackTime=0.1, decayTime=0.2, sustainLevel=0.7, releaseTime=0.5;

    var env, sig;
    env = EnvGen.kr(
        Env.adsr(attackTime, decayTime, sustainLevel, releaseTime),
        gate: gate,
        doneAction: 2  // Free synth when envelope is complete
    );
    sig = SinOsc.ar(freq, 0, amp) * env;
    Out.ar(0, Pan2.ar(sig, pan));
}).add;

// Dictionary to store active synths
~activeSynths = Dictionary.new;

// Helper function to safely trigger a voice
~triggerVoice = { |voiceKey, freq, amp, attack, decay, sustain, release|
    // First, release any existing synth for this voice
    if(~activeSynths[voiceKey].notNil, {
        ~activeSynths[voiceKey].set(\gate, 0);
    });

    // Create new synth and store it
    ~activeSynths[voiceKey] = Synth(\voiceSynth, [
        \freq, freq,
        \amp, amp,
        \attackTime, attack,
        \decayTime, decay,
        \sustainLevel, sustain,
        \releaseTime, release,
        \gate, 1
    ]);
};

// Create key responders
~keyDownResponder = { |view, char, modifiers, unicode, keycode|
    switch(char,
        $1, {
            "Triggering Voice 1".postln;
            ~triggerVoice.value(\voice1, 220, 0.3, 3.1, 0.2, 0.7, 0.1);
        },
        $2, {
            "Triggering Voice 2".postln;
            ~triggerVoice.value(\voice2, 440, 0.25, 0.15, 0.2, 0.2, 0.1);
        },
        $3, {
            "Triggering Voice 3".postln;
            ~triggerVoice.value(\voice3, 550, 0.2, 5.2, 0.4, 0.5, 5.2);
        }
    );
};

~keyUpResponder = { |view, char, modifiers, unicode, keycode|
    switch(char,
        $1, {
            if(~activeSynths[\voice1].notNil, {
                ~activeSynths[\voice1].set(\gate, 0);
                ~activeSynths[\voice1] = nil;
            });
        },
        $2, {
            if(~activeSynths[\voice2].notNil, {
                ~activeSynths[\voice2].set(\gate, 0);
                ~activeSynths[\voice2] = nil;
            });
        },
        $3, {
            if(~activeSynths[\voice3].notNil, {
                ~activeSynths[\voice3].set(\gate, 0);
                ~activeSynths[\voice3] = nil;
            });
        }
    );
};

// Create a window to capture keyboard events
w = Window("Keyboard Input", Rect(100, 100, 200, 100)).front;
w.view.keyDownAction = ~keyDownResponder;
w.view.keyUpAction = ~keyUpResponder;

// To stop all synths, execute:
~activeSynths.do({ |synth| synth.free });
~activeSynths.clear;

