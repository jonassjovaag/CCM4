# audio_file_learning/__init__.py
# Audio File Learning Module for Drift Engine AI
# Standalone module for learning from pre-recorded audio files

from .file_processor import AudioFileProcessor
from .batch_trainer import BatchTrainer
from .learn_from_files import main as learn_from_files_main

__all__ = ['AudioFileProcessor', 'BatchTrainer', 'learn_from_files_main']
