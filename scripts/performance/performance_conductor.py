#!/usr/bin/env python3
# performance_conductor.py
# Orchestrates multiple MusicHal_9000 sessions with Hydrasynth patch changes
# Creates a "concert" experience with timed pieces and pauses

import sys
import os
import time
import signal
import yaml
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from midi_io.hydrasynth_controller import HydrasynthController, HydrasynthPatch


@dataclass
class Piece:
    """Represents a single piece in the setlist"""
    name: str
    duration_minutes: float
    patch_bank: int = 0
    patch_program: int = 0
    patch_name: Optional[str] = None  # Use named patch from library
    pause_after_minutes: float = 0.0

    # Optional MusicHal parameters (override defaults)
    density: Optional[float] = None
    give_space: Optional[float] = None
    initiative: Optional[float] = None
    enable_meld: bool = False
    enable_somax: bool = False

    def __post_init__(self):
        # Ensure numeric types
        self.duration_minutes = float(self.duration_minutes)
        self.pause_after_minutes = float(self.pause_after_minutes)


@dataclass
class Setlist:
    """A complete setlist/concert program"""
    name: str
    pieces: List[Piece]
    description: str = ""

    # Global settings for all pieces
    hydrasynth_port: Optional[str] = None  # MIDI port for Hydrasynth
    melodic_port: str = "IAC Meld"
    bass_port: str = "IAC Meld Bass"

    # Patch library for named patches
    patch_library: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def total_music_time(self) -> float:
        """Total music duration in minutes"""
        return sum(p.duration_minutes for p in self.pieces)

    def total_pause_time(self) -> float:
        """Total pause duration in minutes"""
        return sum(p.pause_after_minutes for p in self.pieces)

    def total_time(self) -> float:
        """Total concert duration in minutes"""
        return self.total_music_time() + self.total_pause_time()


class PerformanceConductor:
    """
    Orchestrates a concert of multiple MusicHal_9000 sessions

    Handles:
    - Loading setlists from YAML
    - Changing Hydrasynth patches between pieces
    - Starting/stopping MusicHal_9000 with timed durations
    - Managing pauses between pieces
    - Progress display and logging
    """

    def __init__(self,
                 setlist: Optional[Setlist] = None,
                 dry_run: bool = False,
                 verbose: bool = True):
        """
        Initialize the conductor

        Args:
            setlist: Setlist to perform (can be loaded later)
            dry_run: If True, simulate without starting MusicHal
            verbose: Print detailed progress
        """
        self.setlist = setlist
        self.dry_run = dry_run
        self.verbose = verbose

        # State
        self.running = False
        self.current_piece_index = 0
        self.concert_start_time: Optional[datetime] = None

        # Controllers
        self.hydrasynth: Optional[HydrasynthController] = None
        self.musichal: Optional[Any] = None  # EnhancedDriftEngineAI instance

        # Signal handling
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on Ctrl+C"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\n\n🛑 Concert interrupted - stopping gracefully...")
        self.stop()
        sys.exit(0)

    def load_setlist_from_yaml(self, filepath: str) -> bool:
        """
        Load setlist from YAML file

        Args:
            filepath: Path to YAML setlist file

        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            # Parse pieces
            pieces = []
            for piece_data in data.get('pieces', []):
                piece = Piece(
                    name=piece_data.get('name', f"Piece {len(pieces) + 1}"),
                    duration_minutes=piece_data.get('duration_minutes', 5),
                    patch_bank=piece_data.get('patch_bank', 0),
                    patch_program=piece_data.get('patch_program', 0),
                    patch_name=piece_data.get('patch_name'),
                    pause_after_minutes=piece_data.get('pause_after_minutes', 0),
                    density=piece_data.get('density'),
                    give_space=piece_data.get('give_space'),
                    initiative=piece_data.get('initiative'),
                    enable_meld=piece_data.get('enable_meld', False),
                    enable_somax=piece_data.get('enable_somax', False)
                )
                pieces.append(piece)

            # Create setlist
            self.setlist = Setlist(
                name=data.get('name', 'Untitled Concert'),
                pieces=pieces,
                description=data.get('description', ''),
                hydrasynth_port=data.get('hydrasynth_port'),
                melodic_port=data.get('melodic_port', 'IAC Meld'),
                bass_port=data.get('bass_port', 'IAC Meld Bass'),
                patch_library=data.get('patch_library', {})
            )

            print(f"📋 Loaded setlist: {self.setlist.name}")
            print(f"   {len(pieces)} pieces, {self.setlist.total_time():.1f} min total")
            return True

        except Exception as e:
            print(f"Failed to load setlist: {e}")
            return False

    def _connect_hydrasynth(self) -> bool:
        """Connect to Hydrasynth for patch changes"""
        try:
            port_name = self.setlist.hydrasynth_port if self.setlist else None

            # Use bass port as default for Hydrasynth Explorer
            if not port_name and self.setlist:
                port_name = self.setlist.bass_port

            self.hydrasynth = HydrasynthController(
                port_name=port_name,
                auto_connect=True
            )

            # Load patch library if defined
            if self.setlist and self.setlist.patch_library:
                self.hydrasynth.load_patch_library(self.setlist.patch_library)

            return self.hydrasynth.connected

        except Exception as e:
            print(f"Failed to connect Hydrasynth: {e}")
            return False

    def _change_patch_for_piece(self, piece: Piece) -> bool:
        """Change Hydrasynth patch for a piece"""
        if not self.hydrasynth or not self.hydrasynth.connected:
            print("   (Hydrasynth not connected - skipping patch change)")
            return False

        if piece.patch_name:
            return self.hydrasynth.change_patch_by_name(piece.patch_name)
        else:
            return self.hydrasynth.change_patch(piece.patch_bank, piece.patch_program)

    def _start_musichal(self, piece: Piece) -> bool:
        """Start MusicHal_9000 for a piece"""
        if self.dry_run:
            print(f"   [DRY RUN] Would start MusicHal for {piece.duration_minutes} minutes")
            return True

        try:
            # Import here to avoid circular imports and allow dry runs
            from scripts.performance.MusicHal_9000 import EnhancedDriftEngineAI

            # Convert duration to integer minutes for timeline
            duration_int = int(piece.duration_minutes)

            # Create MusicHal instance
            self.musichal = EnhancedDriftEngineAI(
                midi_port=self.setlist.melodic_port if self.setlist else "IAC Meld",
                enable_mpe=True,
                enable_meld=piece.enable_meld,
                enable_somax=piece.enable_somax,
                performance_duration=duration_int,
                enable_visualization=False,  # No GUI in conductor mode
                enable_gpt_reflection=True,
                debug_decisions=False
            )

            # Apply piece-specific parameters
            if piece.density is not None:
                self.musichal.set_density_level(piece.density)
            if piece.give_space is not None:
                self.musichal.set_give_space_factor(piece.give_space)
            if piece.initiative is not None:
                self.musichal.set_initiative_budget(piece.initiative)

            # Start the performance
            success = self.musichal.start()
            if success:
                print(f"   🎵 MusicHal started ({duration_int} min)")
            return success

        except Exception as e:
            print(f"   Failed to start MusicHal: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _wait_for_musichal(self, piece: Piece):
        """Wait for MusicHal to complete its performance"""
        if self.dry_run:
            # Simulate time passing (accelerated for dry run)
            print(f"   [DRY RUN] Simulating {piece.duration_minutes} min performance...")
            time.sleep(2)  # Brief simulation
            return

        if not self.musichal:
            return

        # Wait for performance to complete
        # MusicHal's timeline manager will set running=False when done
        duration_seconds = piece.duration_minutes * 60
        start = time.time()

        while self.running and self.musichal.running:
            elapsed = time.time() - start
            remaining = duration_seconds - elapsed

            if remaining <= 0:
                break

            # Progress update every 30 seconds
            if int(elapsed) % 30 == 0:
                mins_remaining = remaining / 60
                print(f"   ⏱️  {mins_remaining:.1f} min remaining", end='\r')

            time.sleep(1.0)

        # Ensure cleanup
        if self.musichal:
            self.musichal.stop()
            self.musichal = None

    def _wait_pause(self, minutes: float):
        """Wait for pause between pieces"""
        if minutes <= 0:
            return

        if self.dry_run:
            print(f"   [DRY RUN] Would pause for {minutes} minutes")
            time.sleep(1)
            return

        print(f"\n   ⏸️  Pause: {minutes:.1f} minutes")

        pause_seconds = minutes * 60
        start = time.time()

        while self.running:
            elapsed = time.time() - start
            remaining = pause_seconds - elapsed

            if remaining <= 0:
                break

            # Update every 10 seconds
            if int(elapsed) % 10 == 0:
                mins_remaining = remaining / 60
                print(f"   ⏸️  {mins_remaining:.1f} min remaining in pause", end='\r')

            time.sleep(1.0)

        print()  # Clear the line

    def _print_concert_header(self):
        """Print concert program"""
        if not self.setlist:
            return

        print("\n" + "=" * 60)
        print(f"🎭 CONCERT: {self.setlist.name}")
        if self.setlist.description:
            print(f"   {self.setlist.description}")
        print("=" * 60)
        print(f"\n📋 Program ({len(self.setlist.pieces)} pieces):\n")

        cumulative = 0.0
        for i, piece in enumerate(self.setlist.pieces, 1):
            patch_info = piece.patch_name or f"Bank {piece.patch_bank}, Prog {piece.patch_program}"
            print(f"   {i}. {piece.name}")
            print(f"      Duration: {piece.duration_minutes} min | Patch: {patch_info}")
            if piece.pause_after_minutes > 0:
                print(f"      Followed by: {piece.pause_after_minutes} min pause")
            cumulative += piece.duration_minutes + piece.pause_after_minutes
            print()

        print(f"📊 Total: {self.setlist.total_music_time():.1f} min music + "
              f"{self.setlist.total_pause_time():.1f} min pauses = "
              f"{self.setlist.total_time():.1f} min")
        print("=" * 60 + "\n")

    def start(self) -> bool:
        """
        Start the concert

        Returns:
            True if concert completed successfully
        """
        if not self.setlist or not self.setlist.pieces:
            print("No setlist loaded")
            return False

        self.running = True
        self.concert_start_time = datetime.now()
        self.current_piece_index = 0

        # Print program
        self._print_concert_header()

        # Connect to Hydrasynth
        print("🔌 Connecting to Hydrasynth...")
        if not self._connect_hydrasynth():
            print("   Warning: Could not connect to Hydrasynth (continuing without patch changes)")

        print(f"\n🎬 Concert starting at {self.concert_start_time.strftime('%H:%M:%S')}")
        expected_end = self.concert_start_time + timedelta(minutes=self.setlist.total_time())
        print(f"   Expected end: {expected_end.strftime('%H:%M:%S')}\n")

        # Perform each piece
        for i, piece in enumerate(self.setlist.pieces):
            if not self.running:
                break

            self.current_piece_index = i

            print("-" * 40)
            print(f"🎼 PIECE {i + 1}/{len(self.setlist.pieces)}: {piece.name}")
            print("-" * 40)

            # Change patch
            print("   Changing Hydrasynth patch...")
            self._change_patch_for_piece(piece)

            # Small delay for patch to load
            time.sleep(0.5)

            # Start MusicHal
            print(f"   Starting performance ({piece.duration_minutes} min)...")
            if self._start_musichal(piece):
                self._wait_for_musichal(piece)
                print(f"   ✅ {piece.name} complete")
            else:
                print(f"   ❌ Failed to start piece")

            # Pause after piece
            if piece.pause_after_minutes > 0 and self.running:
                self._wait_pause(piece.pause_after_minutes)

        # Concert complete
        if self.running:
            end_time = datetime.now()
            duration = end_time - self.concert_start_time
            print("\n" + "=" * 60)
            print("🎭 CONCERT COMPLETE")
            print(f"   Started: {self.concert_start_time.strftime('%H:%M:%S')}")
            print(f"   Ended: {end_time.strftime('%H:%M:%S')}")
            print(f"   Duration: {duration}")
            print("=" * 60 + "\n")

        self.stop()
        return True

    def stop(self):
        """Stop the concert"""
        self.running = False

        # Stop MusicHal if running
        if self.musichal:
            print("   Stopping MusicHal...")
            self.musichal.stop()
            self.musichal = None

        # Disconnect Hydrasynth
        if self.hydrasynth:
            self.hydrasynth.disconnect()
            self.hydrasynth = None

    def get_status(self) -> Dict[str, Any]:
        """Get current conductor status"""
        return {
            'running': self.running,
            'setlist_name': self.setlist.name if self.setlist else None,
            'current_piece': self.current_piece_index + 1 if self.setlist else 0,
            'total_pieces': len(self.setlist.pieces) if self.setlist else 0,
            'concert_start': self.concert_start_time.isoformat() if self.concert_start_time else None,
            'hydrasynth_connected': self.hydrasynth.connected if self.hydrasynth else False
        }


def create_example_setlist() -> Setlist:
    """Create the example setlist from the user's specification"""
    return Setlist(
        name="Four Movements",
        description="A four-part generative concert for Hydrasynth Explorer",
        pieces=[
            Piece(
                name="Movement I: Opening",
                duration_minutes=10,
                patch_bank=0,
                patch_program=0,  # Replace with actual patch number
                pause_after_minutes=5
            ),
            Piece(
                name="Movement II: Development",
                duration_minutes=7,
                patch_bank=0,
                patch_program=1,  # Replace with actual patch number
                pause_after_minutes=15
            ),
            Piece(
                name="Movement III: Exploration",
                duration_minutes=13,
                patch_bank=0,
                patch_program=2,  # Replace with actual patch number
                pause_after_minutes=5
            ),
            Piece(
                name="Movement IV: Finale",
                duration_minutes=6,
                patch_bank=0,
                patch_program=3,  # Replace with actual patch number
                pause_after_minutes=0
            ),
        ],
        bass_port="IAC Meld Bass",
        melodic_port="IAC Meld"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Performance Conductor - Orchestrate MusicHal_9000 concerts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a setlist YAML file
  python performance_conductor.py --setlist my_concert.yaml

  # Dry run to preview without starting MusicHal
  python performance_conductor.py --setlist my_concert.yaml --dry-run

  # Run the built-in example setlist
  python performance_conductor.py --example

  # Generate an example YAML setlist file
  python performance_conductor.py --generate-example
        """
    )

    parser.add_argument('--setlist', type=str, help='Path to setlist YAML file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate concert without starting MusicHal')
    parser.add_argument('--example', action='store_true',
                        help='Run built-in example setlist')
    parser.add_argument('--generate-example', type=str, nargs='?',
                        const='example_setlist.yaml',
                        help='Generate example YAML setlist file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Generate example YAML
    if args.generate_example:
        example_yaml = """# Performance Conductor Setlist
# Define your concert program here

name: "Four Movements"
description: "A four-part generative concert for Hydrasynth Explorer"

# MIDI port configuration
hydrasynth_port: "IAC Meld Bass"  # Port for Hydrasynth patch changes
melodic_port: "IAC Meld"           # Port for melodic voice
bass_port: "IAC Meld Bass"         # Port for bass voice

# Optional: Define named patches for easier reference
patch_library:
  warm_pad:
    bank: 0
    program: 12
    display_name: "Warm Evolving Pad"
  dark_bass:
    bank: 0
    program: 45
    display_name: "Dark Resonant Bass"
  ethereal:
    bank: 1
    program: 3
    display_name: "Ethereal Texture"
  finale_lead:
    bank: 0
    program: 88
    display_name: "Finale Lead"

# The concert program
pieces:
  - name: "Movement I: Opening"
    duration_minutes: 10
    patch_name: "warm_pad"       # Use named patch
    pause_after_minutes: 5
    # Optional MusicHal parameters:
    # density: 0.4
    # give_space: 0.5
    # initiative: 0.6

  - name: "Movement II: Development"
    duration_minutes: 7
    patch_name: "dark_bass"
    pause_after_minutes: 15

  - name: "Movement III: Exploration"
    duration_minutes: 13
    patch_name: "ethereal"
    pause_after_minutes: 5
    enable_somax: true           # Enable SomaxBridge for this piece

  - name: "Movement IV: Finale"
    duration_minutes: 6
    patch_bank: 0                # Or use bank/program directly
    patch_program: 88
    pause_after_minutes: 0
    density: 0.7                 # Higher density for finale
    initiative: 0.8
"""
        output_path = args.generate_example
        with open(output_path, 'w') as f:
            f.write(example_yaml)
        print(f"📝 Generated example setlist: {output_path}")
        return

    # Create conductor
    conductor = PerformanceConductor(
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    # Load setlist
    if args.setlist:
        if not conductor.load_setlist_from_yaml(args.setlist):
            sys.exit(1)
    elif args.example:
        conductor.setlist = create_example_setlist()
        print("📋 Using built-in example setlist")
    else:
        parser.print_help()
        print("\n❌ Please specify --setlist or --example")
        sys.exit(1)

    # Start the concert
    try:
        conductor.start()
    except KeyboardInterrupt:
        print("\n🛑 Concert interrupted")
        conductor.stop()


if __name__ == "__main__":
    main()
