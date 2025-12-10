# hydrasynth_controller.py
# Hydrasynth-specific MIDI control for patch changes and parameter control
# Separate module to keep synth-specific functionality isolated

import mido
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class HydrasynthPatch:
    """Represents a Hydrasynth patch/preset"""
    bank: int  # Bank number (0-127 for single byte, or combined MSB/LSB)
    program: int  # Program number within bank (0-127)
    name: Optional[str] = None  # Human-readable name


class HydrasynthController:
    """
    Controller for ASM Hydrasynth synthesizers

    Handles patch changes via standard MIDI Bank Select + Program Change.
    Designed to work alongside MPE note output on separate channels.

    MIDI Implementation:
    - Bank Select MSB: CC 0
    - Bank Select LSB: CC 32
    - Program Change: Standard MIDI Program Change

    Channel Strategy:
    - Patch changes go to master channel (channel 1, mido index 0)
    - This is compatible with MPE mode where notes use channels 2-15
    """

    def __init__(self,
                 port_name: Optional[str] = None,
                 master_channel: int = 0,  # mido uses 0-indexed channels
                 auto_connect: bool = True):
        """
        Initialize Hydrasynth controller

        Args:
            port_name: MIDI port name (None = use default or first available)
            master_channel: MIDI channel for patch changes (0-indexed, default 0 = channel 1)
            auto_connect: Automatically connect on init
        """
        self.port_name = port_name
        self.master_channel = master_channel
        self.port: Optional[mido.ports.BaseOutput] = None

        # Current state tracking
        self.current_bank: int = 0
        self.current_program: int = 0
        self.connected: bool = False

        # Patch library (optional, for named presets)
        self.patch_library: Dict[str, HydrasynthPatch] = {}

        if auto_connect:
            self.connect()

    def connect(self, port_name: Optional[str] = None) -> bool:
        """
        Connect to MIDI output port

        Args:
            port_name: Override port name (uses init value if None)

        Returns:
            True if connected successfully
        """
        if port_name:
            self.port_name = port_name

        try:
            if self.port_name:
                self.port = mido.open_output(self.port_name)
            else:
                # Find first available port
                outputs = mido.get_output_names()
                if outputs:
                    # Prefer ports with "Hydra" or "Bass" in name
                    hydra_ports = [p for p in outputs if 'hydra' in p.lower() or 'bass' in p.lower()]
                    if hydra_ports:
                        self.port = mido.open_output(hydra_ports[0])
                    else:
                        self.port = mido.open_output(outputs[0])
                else:
                    print("No MIDI output ports available")
                    return False

            self.connected = True
            print(f"🎹 Hydrasynth controller connected to: {self.port.name}")
            return True

        except Exception as e:
            print(f"Failed to connect Hydrasynth controller: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from MIDI port"""
        if self.port:
            self.port.close()
            self.port = None
        self.connected = False

    def change_patch(self, bank: int = 0, program: int = 0,
                     delay_ms: float = 10.0) -> bool:
        """
        Change Hydrasynth patch using Bank Select + Program Change

        The Hydrasynth expects messages in this order:
        1. Bank Select MSB (CC 0)
        2. Bank Select LSB (CC 32)
        3. Program Change

        Args:
            bank: Bank number (0-16383, split into MSB/LSB)
            program: Program number within bank (0-127)
            delay_ms: Delay between messages in milliseconds

        Returns:
            True if messages sent successfully
        """
        if not self.port or not self.connected:
            print("Hydrasynth controller not connected")
            return False

        try:
            # Split bank into MSB and LSB
            bank_msb = (bank >> 7) & 0x7F  # High 7 bits
            bank_lsb = bank & 0x7F          # Low 7 bits

            # Clamp program to valid range
            program = max(0, min(127, program))

            # Send Bank Select MSB (CC 0)
            self.port.send(mido.Message('control_change',
                                       channel=self.master_channel,
                                       control=0,
                                       value=bank_msb))

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # Send Bank Select LSB (CC 32)
            self.port.send(mido.Message('control_change',
                                       channel=self.master_channel,
                                       control=32,
                                       value=bank_lsb))

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # Send Program Change
            self.port.send(mido.Message('program_change',
                                       channel=self.master_channel,
                                       program=program))

            # Update state
            self.current_bank = bank
            self.current_program = program

            print(f"🎛️ Hydrasynth patch changed: Bank {bank} (MSB:{bank_msb}, LSB:{bank_lsb}), Program {program}")
            return True

        except Exception as e:
            print(f"Failed to change Hydrasynth patch: {e}")
            return False

    def change_patch_by_name(self, name: str) -> bool:
        """
        Change patch using a named preset from the library

        Args:
            name: Patch name (case-insensitive)

        Returns:
            True if patch found and changed successfully
        """
        name_lower = name.lower()

        for patch_name, patch in self.patch_library.items():
            if patch_name.lower() == name_lower:
                success = self.change_patch(patch.bank, patch.program)
                if success and patch.name:
                    print(f"   Patch: {patch.name}")
                return success

        print(f"Patch '{name}' not found in library")
        return False

    def register_patch(self, name: str, bank: int, program: int,
                       display_name: Optional[str] = None):
        """
        Register a named patch in the library

        Args:
            name: Key name for lookup (used in setlists)
            bank: Bank number
            program: Program number
            display_name: Human-readable name (optional)
        """
        self.patch_library[name] = HydrasynthPatch(
            bank=bank,
            program=program,
            name=display_name or name
        )

    def load_patch_library(self, patches: Dict[str, Dict[str, Any]]):
        """
        Load multiple patches into the library

        Args:
            patches: Dict of patch definitions
                     {"name": {"bank": 0, "program": 12, "display_name": "Warm Pad"}}
        """
        for name, config in patches.items():
            self.register_patch(
                name=name,
                bank=config.get('bank', 0),
                program=config.get('program', 0),
                display_name=config.get('display_name')
            )
        print(f"📚 Loaded {len(patches)} patches into Hydrasynth library")

    def get_current_patch(self) -> Dict[str, int]:
        """Get current patch state"""
        return {
            'bank': self.current_bank,
            'program': self.current_program
        }

    def test_connection(self) -> bool:
        """
        Test the MIDI connection by sending a patch change

        Returns:
            True if test succeeded
        """
        if not self.connected:
            return False

        print("Testing Hydrasynth MIDI connection...")

        # Send current patch (effectively a no-op if same patch)
        result = self.change_patch(self.current_bank, self.current_program)

        if result:
            print("✅ Hydrasynth connection test passed")
        else:
            print("❌ Hydrasynth connection test failed")

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            'connected': self.connected,
            'port_name': self.port.name if self.port else None,
            'master_channel': self.master_channel + 1,  # Human-readable (1-16)
            'current_bank': self.current_bank,
            'current_program': self.current_program,
            'library_size': len(self.patch_library)
        }

    def __enter__(self):
        """Context manager entry"""
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Convenience function for quick patch changes
def quick_patch_change(bank: int, program: int,
                       port_name: Optional[str] = None) -> bool:
    """
    Quick utility to change Hydrasynth patch without persistent connection

    Args:
        bank: Bank number
        program: Program number
        port_name: MIDI port name (optional)

    Returns:
        True if successful
    """
    with HydrasynthController(port_name=port_name) as ctrl:
        return ctrl.change_patch(bank, program)


# Test/demo when run directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hydrasynth MIDI Patch Controller')
    parser.add_argument('--port', type=str, help='MIDI output port name')
    parser.add_argument('--bank', type=int, default=0, help='Bank number (default: 0)')
    parser.add_argument('--program', type=int, default=0, help='Program number (default: 0)')
    parser.add_argument('--list-ports', action='store_true', help='List available MIDI ports')
    parser.add_argument('--test', action='store_true', help='Test connection')

    args = parser.parse_args()

    if args.list_ports:
        print("Available MIDI output ports:")
        for port in mido.get_output_names():
            print(f"  - {port}")
    else:
        ctrl = HydrasynthController(port_name=args.port)

        if args.test:
            ctrl.test_connection()
        else:
            ctrl.change_patch(args.bank, args.program)

        ctrl.disconnect()
