#!/usr/bin/env python3
"""
Wav2Vec → Chord Classifier
===========================

Neural network classifier that maps Wav2Vec features (768D) to chord labels.
This ensures human-readable chord labels are derived from the same features
the machine uses, maintaining conceptual consistency.

Architecture:
    Audio → Wav2Vec (768D) ┬→ Quantizer → Gesture Tokens [MACHINE]
                            └→ Chord Classifier → Chord Labels [HUMAN]

Usage:
    # Train classifier
    classifier = Wav2VecChordClassifier()
    classifier.train(wav2vec_features, chord_labels)
    classifier.save('models/wav2vec_chord_classifier.pt')
    
    # Use in pipeline
    classifier = Wav2VecChordClassifier.load('models/wav2vec_chord_classifier.pt')
    chord = classifier.predict(wav2vec_feature)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import os


class Wav2VecChordClassifier(nn.Module):
    """
    Neural network classifier: Wav2Vec features → Chord labels
    
    Maps 768D Wav2Vec embeddings to chord types (major, minor, 7th, etc.)
    """
    
    def __init__(self, input_dim: int = 768, num_chord_types: int = 204):
        """
        Initialize classifier
        
        Args:
            input_dim: Dimension of Wav2Vec features (768)
            num_chord_types: Number of chord classes (17 types × 12 roots = 204)
        """
        super(Wav2VecChordClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_chord_types = num_chord_types
        
        # Classifier network: 768 → 512 → 256 → num_chord_types
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_chord_types)
        )
        
        # Store label mappings
        self.label_to_idx = {}
        self.idx_to_label = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Wav2Vec features [batch_size, 768]
            
        Returns:
            Logits [batch_size, num_chord_types]
        """
        return self.classifier(x)
    
    def predict(self, wav2vec_features: np.ndarray, device: str = 'cpu') -> str:
        """
        Predict chord label from Wav2Vec features
        
        Args:
            wav2vec_features: 768D Wav2Vec feature vector
            device: Device to run prediction on ('cpu', 'cuda', 'mps')
            
        Returns:
            Chord label (e.g., "Cmaj7", "Dm", "G7")
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            if isinstance(wav2vec_features, list):
                wav2vec_features = np.array(wav2vec_features)
            
            x = torch.FloatTensor(wav2vec_features).unsqueeze(0).to(device)
            
            # Get prediction
            logits = self.forward(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            
            # Map to chord label
            if pred_idx in self.idx_to_label:
                return self.idx_to_label[pred_idx]
            else:
                return "C"  # Default fallback
    
    def predict_with_confidence(self, wav2vec_features: np.ndarray, device: str = 'cpu') -> Tuple[str, float]:
        """
        Predict chord label with confidence score
        
        Args:
            wav2vec_features: 768D Wav2Vec feature vector
            device: Device to run prediction on ('cpu', 'cuda', 'mps')
            
        Returns:
            Tuple of (chord_label, confidence_score)
            where confidence is the softmax probability of the predicted class
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            if isinstance(wav2vec_features, list):
                wav2vec_features = np.array(wav2vec_features)
            
            x = torch.FloatTensor(wav2vec_features).unsqueeze(0).to(device)
            
            # Get prediction with confidence
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
            
            # Map to chord label
            if pred_idx in self.idx_to_label:
                chord_label = self.idx_to_label[pred_idx]
            else:
                chord_label = "C"  # Default fallback
                confidence = 0.0  # No confidence in fallback
            
            return chord_label, confidence
    
    def predict_batch(self, wav2vec_features_batch: np.ndarray, device: str = 'cpu') -> List[str]:
        """
        Predict chord labels for a batch of Wav2Vec features
        
        Args:
            wav2vec_features_batch: [batch_size, 768] array
            device: Device to run prediction on ('cpu', 'cuda', 'mps')
            
        Returns:
            List of chord labels
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(wav2vec_features_batch).to(device)
            logits = self.forward(x)
            pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
            
            return [self.idx_to_label.get(idx, "C") for idx in pred_indices]
    
    def train_classifier(self,
                        features: np.ndarray,
                        labels: List[str],
                        epochs: int = 50,
                        batch_size: int = 32,
                        learning_rate: float = 0.001,
                        val_split: float = 0.2,
                        verbose: bool = True,
                        device: str = 'cpu') -> Dict:
        """
        Train the classifier on ground truth data
        
        Args:
            features: [N, 768] array of Wav2Vec features
            labels: List of N chord labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            verbose: Print training progress
            
        Returns:
            Training history dict
        """
        # Build label mappings
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_chord_types = len(unique_labels)
        
        # Update final layer if needed
        if self.classifier[-1].out_features != self.num_chord_types:
            self.classifier[-1] = nn.Linear(256, self.num_chord_types)
        
        # Convert labels to indices
        label_indices = np.array([self.label_to_idx[label] for label in labels])
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            features, label_indices, test_size=val_split, random_state=42, stratify=label_indices
        )
        
        # Convert to tensors (keep on CPU for DataLoader)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        if verbose:
            print(f"\n🎓 Training Wav2Vec → Chord Classifier")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")
            print(f"   Chord classes: {self.num_chord_types}")
            print(f"   Epochs: {epochs}\n")
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_features, batch_labels in train_loader:
                # Move batch to device
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.forward(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / len(X_train)
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_val_tensor).sum().item() / len(X_val)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if verbose:
            print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
        
        # Final evaluation
        self.eval()
        with torch.no_grad():
            val_outputs = self.forward(X_val_tensor)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_predicted_np = val_predicted.cpu().numpy()
            
            if verbose:
                print("\n📊 Final Validation Accuracy: {:.2%}".format(best_val_acc))
                print(f"   Correct predictions: {int(best_val_acc * len(X_val))}/{len(X_val)}")
        
        return history
    
    def save(self, filepath: str):
        """Save model and label mappings"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'num_chord_types': self.num_chord_types,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }, filepath)
        
        print(f"✅ Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Wav2VecChordClassifier':
        """Load model and label mappings"""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            num_chord_types=checkpoint['num_chord_types']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.label_to_idx = checkpoint['label_to_idx']
        model.idx_to_label = checkpoint['idx_to_label']
        model.eval()
        
        return model


def generate_harmonic_summary(chord_predictions: List[str]) -> Dict:
    """
    Generate player-friendly harmonic summary from chord predictions
    
    Args:
        chord_predictions: List of chord labels
        
    Returns:
        Dict with percentages of chord types
    """
    chord_types = {
        'major': 0,
        'minor': 0,
        'dominant': 0,
        'diminished': 0,
        'augmented': 0,
        'suspended': 0,
        'other': 0
    }
    
    for chord in chord_predictions:
        chord_lower = chord.lower()
        
        # Major chords
        if 'maj' in chord_lower or (chord[0].isupper() and 'm' not in chord_lower and 'sus' not in chord_lower):
            chord_types['major'] += 1
        # Minor chords
        elif 'm' in chord_lower and 'maj' not in chord_lower and 'dim' not in chord_lower:
            chord_types['minor'] += 1
        # Dominant (7th without maj)
        elif '7' in chord and 'maj' not in chord_lower:
            chord_types['dominant'] += 1
        # Diminished
        elif 'dim' in chord_lower or 'ø' in chord:
            chord_types['diminished'] += 1
        # Augmented
        elif 'aug' in chord_lower or '+' in chord:
            chord_types['augmented'] += 1
        # Suspended
        elif 'sus' in chord_lower:
            chord_types['suspended'] += 1
        else:
            chord_types['other'] += 1
    
    total = len(chord_predictions)
    if total == 0:
        return {k: 0.0 for k in chord_types}
    
    return {k: (v/total)*100 for k, v in chord_types.items()}


if __name__ == "__main__":
    # Quick test
    print("Wav2Vec Chord Classifier Module")
    print("Use train_wav2vec_chord_classifier.py to train the model")

