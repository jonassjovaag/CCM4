"""
Lightweight Music Transformer for Enhanced Training
Provides deep musical analysis for training the AudioOracle system
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import librosa
from dataclasses import dataclass


@dataclass
class MusicalInsights:
    """Container for transformer musical analysis results"""
    chord_progression: List[str]
    scale_analysis: Dict[str, float]
    musical_form: Dict[str, Any]
    harmonic_rhythm: List[float]
    melodic_contour: List[float]
    rhythmic_patterns: List[str]
    key_signature: str
    tempo_analysis: Dict[str, float]
    confidence_scores: Dict[str, float]


class LightweightMusicTransformer(nn.Module):
    """
    Lightweight transformer for musical analysis
    Designed for training mode (not real-time)
    """
    
    def __init__(self, 
                 feature_dim: int = 15,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 max_seq_length: int = 50000):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Musical analysis heads
        self.chord_head = nn.Linear(hidden_dim, 12)  # 12 pitch classes
        self.scale_head = nn.Linear(hidden_dim, 24)  # Major/minor scales
        self.form_head = nn.Linear(hidden_dim, 8)    # Musical form types
        self.harmony_head = nn.Linear(hidden_dim, 1)  # Harmonic complexity
        self.melody_head = nn.Linear(hidden_dim, 1)   # Melodic contour
        self.rhythm_head = nn.Linear(hidden_dim, 1)   # Rhythmic complexity
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer
        
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            
        Returns:
            Dictionary of musical analysis outputs
        """
        batch_size, seq_len, _ = features.shape
        
        # Project input features
        x = self.input_projection(features)
        
        # Add positional encoding
        if seq_len <= self.max_seq_length:
            if seq_len > self.pos_encoding.shape[1]:
                # Dynamically extend positional encoding if needed
                self.pos_encoding = self._create_positional_encoding(seq_len, self.hidden_dim)
            x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        else:
            # For very long sequences, use sliding window approach
            x = x + self.pos_encoding[:, :self.max_seq_length, :].to(x.device)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Musical analysis heads
        chord_logits = self.chord_head(encoded)
        scale_logits = self.scale_head(encoded)
        form_logits = self.form_head(encoded)
        harmony_scores = self.harmony_head(encoded)
        melody_scores = self.melody_head(encoded)
        rhythm_scores = self.rhythm_head(encoded)
        confidence_scores = torch.sigmoid(self.confidence_head(encoded))
        
        return {
            'chord_logits': chord_logits,
            'scale_logits': scale_logits,
            'form_logits': form_logits,
            'harmony_scores': harmony_scores,
            'melody_scores': melody_scores,
            'rhythm_scores': rhythm_scores,
            'confidence_scores': confidence_scores,
            'encoded_features': encoded
        }
    
    def analyze_musical_features(self, features: torch.Tensor) -> MusicalInsights:
        """
        Analyze musical features and return structured insights
        
        Args:
            features: Input features [seq_len, feature_dim]
            
        Returns:
            MusicalInsights object with analysis results
        """
        self.eval()
        with torch.no_grad():
            # Move features to same device as model
            features = features.to(next(self.parameters()).device)
            
            # Process in chunks if sequence is too long
            seq_len = features.shape[0]
            if seq_len > 1000:  # Process in chunks for very long sequences
                chunk_size = 1000
                chunk_insights = []
                
                for i in range(0, seq_len, chunk_size):
                    chunk = features[i:i+chunk_size]
                    chunk_batch = chunk.unsqueeze(0)
                    chunk_outputs = self.forward(chunk_batch)
                    chunk_insights.append(chunk_outputs)
                
                # Combine insights from all chunks
                insights = self._combine_chunk_insights(chunk_insights)
            else:
                # Process normally for shorter sequences
                features_batch = features.unsqueeze(0)
                outputs = self.forward(features_batch)
                insights = self._extract_insights_from_outputs(outputs)
            
            return insights
    
    def _extract_insights_from_outputs(self, outputs: Dict[str, torch.Tensor]) -> MusicalInsights:
        """Extract insights from transformer outputs"""
        chord_progression = self._extract_chord_progression(outputs['chord_logits'][0])
        scale_analysis = self._extract_scale_analysis(outputs['scale_logits'][0])
        musical_form = self._extract_musical_form(outputs['form_logits'][0])
        harmonic_rhythm = outputs['harmony_scores'][0].squeeze().tolist()
        melodic_contour = outputs['melody_scores'][0].squeeze().tolist()
        rhythmic_patterns = self._extract_rhythmic_patterns(outputs['rhythm_scores'][0])
        key_signature = self._extract_key_signature(scale_analysis)
        tempo_analysis = self._extract_tempo_analysis(outputs['encoded_features'][0])
        confidence_scores = {
            'chord': outputs['confidence_scores'][0].mean().item(),
            'scale': outputs['confidence_scores'][0].mean().item(),
            'form': outputs['confidence_scores'][0].mean().item(),
            'harmony': outputs['confidence_scores'][0].mean().item(),
            'melody': outputs['confidence_scores'][0].mean().item(),
            'rhythm': outputs['confidence_scores'][0].mean().item()
        }
        
        return MusicalInsights(
            chord_progression=chord_progression,
            scale_analysis=scale_analysis,
            musical_form=musical_form,
            harmonic_rhythm=harmonic_rhythm,
            melodic_contour=melodic_contour,
            rhythmic_patterns=rhythmic_patterns,
            key_signature=key_signature,
            tempo_analysis=tempo_analysis,
            confidence_scores=confidence_scores
        )
    
    def _combine_chunk_insights(self, chunk_insights: List[Dict[str, torch.Tensor]]) -> MusicalInsights:
        """Combine insights from multiple chunks"""
        # Combine chord progressions
        all_chord_progressions = []
        all_scale_analyses = []
        all_musical_forms = []
        all_harmonic_rhythms = []
        all_melodic_contours = []
        all_rhythmic_patterns = []
        all_confidence_scores = []
        
        for chunk_output in chunk_insights:
            chord_prog = self._extract_chord_progression(chunk_output['chord_logits'][0])
            scale_analysis = self._extract_scale_analysis(chunk_output['scale_logits'][0])
            musical_form = self._extract_musical_form(chunk_output['form_logits'][0])
            harmonic_rhythm = chunk_output['harmony_scores'][0].squeeze().tolist()
            melodic_contour = chunk_output['melody_scores'][0].squeeze().tolist()
            rhythmic_patterns = self._extract_rhythmic_patterns(chunk_output['rhythm_scores'][0])
            confidence_scores = {
                'chord': chunk_output['confidence_scores'][0].mean().item(),
                'scale': chunk_output['confidence_scores'][0].mean().item(),
                'form': chunk_output['confidence_scores'][0].mean().item(),
                'harmony': chunk_output['confidence_scores'][0].mean().item(),
                'melody': chunk_output['confidence_scores'][0].mean().item(),
                'rhythm': chunk_output['confidence_scores'][0].mean().item()
            }
            
            all_chord_progressions.extend(chord_prog)
            all_scale_analyses.append(scale_analysis)
            all_musical_forms.append(musical_form)
            all_harmonic_rhythms.extend(harmonic_rhythm)
            all_melodic_contours.extend(melodic_contour)
            all_rhythmic_patterns.extend(rhythmic_patterns)
            all_confidence_scores.append(confidence_scores)
        
        # Average scale analysis across chunks
        avg_scale_analysis = {}
        for scale_name in all_scale_analyses[0].keys():
            avg_scale_analysis[scale_name] = sum(sa[scale_name] for sa in all_scale_analyses) / len(all_scale_analyses)
        
        # Average musical form across chunks
        avg_musical_form = {}
        for form_name in all_musical_forms[0].keys():
            avg_musical_form[form_name] = sum(mf[form_name] for mf in all_musical_forms) / len(all_musical_forms)
        
        # Average confidence scores
        avg_confidence_scores = {}
        for score_name in all_confidence_scores[0].keys():
            avg_confidence_scores[score_name] = sum(cs[score_name] for cs in all_confidence_scores) / len(all_confidence_scores)
        
        key_signature = self._extract_key_signature(avg_scale_analysis)
        tempo_analysis = {'estimated_tempo': 120.0, 'tempo_variation': 0.1, 'tempo_stability': 0.9}  # Default values
        
        return MusicalInsights(
            chord_progression=all_chord_progressions,
            scale_analysis=avg_scale_analysis,
            musical_form=avg_musical_form,
            harmonic_rhythm=all_harmonic_rhythms,
            melodic_contour=all_melodic_contours,
            rhythmic_patterns=all_rhythmic_patterns,
            key_signature=key_signature,
            tempo_analysis=tempo_analysis,
            confidence_scores=avg_confidence_scores
        )
    
    def _extract_chord_progression(self, chord_logits: torch.Tensor) -> List[str]:
        """Extract chord progression from logits"""
        chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_progression = []
        
        for i in range(chord_logits.shape[0]):
            chord_idx = torch.argmax(chord_logits[i]).item()
            chord_progression.append(chord_names[chord_idx])
        
        return chord_progression
    
    def _extract_scale_analysis(self, scale_logits: torch.Tensor) -> Dict[str, float]:
        """Extract scale analysis from logits"""
        scale_names = [
            'C_major', 'C_minor', 'C#_major', 'C#_minor',
            'D_major', 'D_minor', 'D#_major', 'D#_minor',
            'E_major', 'E_minor', 'F_major', 'F_minor',
            'F#_major', 'F#_minor', 'G_major', 'G_minor',
            'G#_major', 'G#_minor', 'A_major', 'A_minor',
            'A#_major', 'A#_minor', 'B_major', 'B_minor'
        ]
        
        scale_probs = torch.softmax(scale_logits.mean(dim=0), dim=0)
        scale_analysis = {}
        
        for i, scale_name in enumerate(scale_names):
            scale_analysis[scale_name] = scale_probs[i].item()
        
        return scale_analysis
    
    def _extract_musical_form(self, form_logits: torch.Tensor) -> Dict[str, Any]:
        """Extract musical form analysis"""
        form_types = ['verse', 'chorus', 'bridge', 'intro', 'outro', 'solo', 'break', 'coda']
        form_probs = torch.softmax(form_logits.mean(dim=0), dim=0)
        
        form_analysis = {}
        for i, form_type in enumerate(form_types):
            form_analysis[form_type] = form_probs[i].item()
        
        return form_analysis
    
    def _extract_rhythmic_patterns(self, rhythm_scores: torch.Tensor) -> List[str]:
        """Extract rhythmic patterns from scores"""
        patterns = []
        for score in rhythm_scores:
            if score.item() > 0.7:
                patterns.append("complex")
            elif score.item() > 0.4:
                patterns.append("moderate")
            else:
                patterns.append("simple")
        return patterns
    
    def _extract_key_signature(self, scale_analysis: Dict[str, float]) -> str:
        """Extract most likely key signature"""
        return max(scale_analysis.items(), key=lambda x: x[1])[0]
    
    def _extract_tempo_analysis(self, features: torch.Tensor) -> Dict[str, float]:
        """Extract tempo analysis from features"""
        # Simple tempo estimation based on feature patterns
        tempo_scores = features[:, 6] if features.shape[1] > 6 else torch.zeros(features.shape[0])
        
        return {
            'estimated_tempo': float(tempo_scores.mean() * 120 + 60),  # Rough estimate
            'tempo_variation': float(tempo_scores.std()),
            'tempo_stability': float(1.0 - tempo_scores.std())
        }


class TransformerMusicalAnalyzer:
    """
    High-level interface for transformer-based musical analysis
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = LightweightMusicTransformer()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with random weights (in real implementation, load pre-trained)
            self.model = self.model.to(self.device)
            print("⚠️ Using randomly initialized transformer (for demo)")
    
    def load_model(self, model_path: str):
        """Load pre-trained transformer model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            print(f"✅ Loaded transformer model from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("⚠️ Using randomly initialized transformer")
    
    def analyze_audio_features(self, features: List[Dict]) -> MusicalInsights:
        """
        Analyze audio features using transformer
        
        Args:
            features: List of feature dictionaries from AudioOracle
            
        Returns:
            MusicalInsights object
        """
        # Convert features to tensor
        feature_tensor = self._features_to_tensor(features)
        
        # Analyze with transformer
        insights = self.model.analyze_musical_features(feature_tensor)
        
        return insights
    
    def _features_to_tensor(self, features: List[Dict]) -> torch.Tensor:
        """Convert feature dictionaries to tensor"""
        feature_matrix = []
        
        for feature_dict in features:
            # Extract numerical features
            feature_vector = [
                feature_dict.get('rms_db', 0.0),
                feature_dict.get('f0', 440.0),
                feature_dict.get('centroid', 1000.0),
                feature_dict.get('rolloff', 2000.0),
                feature_dict.get('bandwidth', 1000.0),
                feature_dict.get('contrast', 0.5),
                feature_dict.get('flatness', 0.1),
                feature_dict.get('mfcc_1', 0.0),
                feature_dict.get('duration', 0.5),
                feature_dict.get('attack_time', 0.1),
                feature_dict.get('release_time', 0.3),
                feature_dict.get('tempo', 120.0),
                feature_dict.get('beat_position', 0.0),
                feature_dict.get('midi', 60),
                feature_dict.get('cents', 0.0)
            ]
            feature_matrix.append(feature_vector)
        
        return torch.tensor(feature_matrix, dtype=torch.float32)
    
    def save_model(self, model_path: str):
        """Save transformer model"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'feature_dim': self.model.feature_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'max_seq_length': self.model.max_seq_length
                }
            }
            torch.save(checkpoint, model_path)
            print(f"✅ Saved transformer model to {model_path}")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
