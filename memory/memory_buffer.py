# memory_buffer.py
# Rolling buffer and memory management for AI listener system

import time
import numpy as np
import json
import pickle
import os
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

@dataclass
class MusicalMoment:
    """Represents a musical moment in the memory buffer"""
    timestamp: float
    features: np.ndarray  # [centroid_z, rms_norm, cents_smooth, ioi_norm]
    event_data: Dict
    cluster_id: Optional[int] = None

class MemoryBuffer:
    """
    Rolling buffer for musical moments with efficient storage and retrieval
    Maintains 120-300 seconds of musical history
    """
    
    def __init__(self, max_duration_seconds: float = 180.0, 
                 feature_dimensions: int = 4):
        self.max_duration_seconds = max_duration_seconds
        self.feature_dimensions = feature_dimensions
        self.buffer: deque[MusicalMoment] = deque()
        
        # Feature normalization parameters
        self.feature_stats = {
            'centroid': {'mean': 0.0, 'std': 1.0},
            'rms': {'mean': 0.0, 'std': 1.0},
            'cents': {'mean': 0.0, 'std': 1.0},
            'ioi': {'mean': 0.0, 'std': 1.0}
        }
        
        # Update statistics as we collect data
        self._update_stats = True
        self._stats_window = 1000  # Update stats from last 1000 moments
        
    def add_moment(self, event_data: Dict) -> MusicalMoment:
        """Add a new musical moment to the buffer"""
        current_time = time.time()
        
        # Extract features from event
        features = self._extract_features(event_data)
        
        # Create musical moment
        moment = MusicalMoment(
            timestamp=current_time,
            features=features,
            event_data=event_data
        )
        
        # Add to buffer
        self.buffer.append(moment)
        
        # Remove old moments
        self._cleanup_old_moments(current_time)
        
        # Update feature statistics
        if self._update_stats:
            self._update_feature_stats()
        
        return moment
    
    def _extract_features(self, event_data: Dict) -> np.ndarray:
        """Extract normalized feature vector from event data"""
        # Raw features
        centroid = event_data.get('centroid', 0.0)
        rms_db = event_data.get('rms_db', -80.0)
        cents = event_data.get('cents', 0.0)
        ioi = event_data.get('ioi', 0.0)
        
        # Normalize features using current statistics
        centroid_norm = (centroid - self.feature_stats['centroid']['mean']) / \
                       max(self.feature_stats['centroid']['std'], 1e-6)
        rms_norm = (rms_db - self.feature_stats['rms']['mean']) / \
                   max(self.feature_stats['rms']['std'], 1e-6)
        cents_norm = (cents - self.feature_stats['cents']['mean']) / \
                     max(self.feature_stats['cents']['std'], 1e-6)
        ioi_norm = (ioi - self.feature_stats['ioi']['mean']) / \
                   max(self.feature_stats['ioi']['std'], 1e-6)
        
        return np.array([centroid_norm, rms_norm, cents_norm, ioi_norm])
    
    def _cleanup_old_moments(self, current_time: float):
        """Remove moments older than max_duration_seconds"""
        cutoff_time = current_time - self.max_duration_seconds
        while self.buffer and self.buffer[0].timestamp < cutoff_time:
            self.buffer.popleft()
    
    def _update_feature_stats(self):
        """Update feature normalization statistics"""
        if len(self.buffer) < 10:
            return
        
        # Use recent moments for statistics
        recent_moments = list(self.buffer)[-self._stats_window:]
        
        # Extract raw features
        centroids = []
        rms_values = []
        cents_values = []
        ioi_values = []
        
        for moment in recent_moments:
            event = moment.event_data
            centroids.append(event.get('centroid', 0.0))
            rms_values.append(event.get('rms_db', -80.0))
            cents_values.append(event.get('cents', 0.0))
            ioi_values.append(event.get('ioi', 0.0))
        
        # Update statistics
        self.feature_stats['centroid']['mean'] = np.mean(centroids)
        self.feature_stats['centroid']['std'] = np.std(centroids)
        self.feature_stats['rms']['mean'] = np.mean(rms_values)
        self.feature_stats['rms']['std'] = np.std(rms_values)
        self.feature_stats['cents']['mean'] = np.mean(cents_values)
        self.feature_stats['cents']['std'] = np.std(cents_values)
        self.feature_stats['ioi']['mean'] = np.mean(ioi_values)
        self.feature_stats['ioi']['std'] = np.std(ioi_values)
    
    def get_recent_moments(self, duration_seconds: float = 30.0) -> List[MusicalMoment]:
        """Get moments from the last duration_seconds"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        recent_moments = []
        for moment in reversed(self.buffer):
            if moment.timestamp >= cutoff_time:
                recent_moments.append(moment)
            else:
                break
        
        return list(reversed(recent_moments))
    
    def get_all_moments(self) -> List[MusicalMoment]:
        """Get all moments in the buffer"""
        return list(self.buffer)
    
    def get_feature_matrix(self, moments: Optional[List[MusicalMoment]] = None) -> np.ndarray:
        """Get feature matrix for clustering"""
        if moments is None:
            moments = self.get_all_moments()
        
        if not moments:
            return np.array([]).reshape(0, self.feature_dimensions)
        
        return np.array([moment.features for moment in moments])
    
    def find_neighbors(self, target_features: np.ndarray, 
                      radius: float = 1.0, 
                      max_results: int = 10) -> List[MusicalMoment]:
        """Find moments similar to target features within radius"""
        if not self.buffer:
            return []
        
        distances = []
        for moment in self.buffer:
            dist = np.linalg.norm(moment.features - target_features)
            if dist <= radius:
                distances.append((dist, moment))
        
        # Sort by distance and return top results
        distances.sort(key=lambda x: x[0])
        return [moment for _, moment in distances[:max_results]]
    
    def find_distant_moments(self, target_features: np.ndarray,
                           min_distance: float = 2.0,
                           max_results: int = 10) -> List[MusicalMoment]:
        """Find moments that contrast with target features"""
        if not self.buffer:
            return []
        
        distances = []
        for moment in self.buffer:
            dist = np.linalg.norm(moment.features - target_features)
            if dist >= min_distance:
                distances.append((dist, moment))
        
        # Sort by distance (descending) and return top results
        distances.sort(key=lambda x: x[0], reverse=True)
        return [moment for _, moment in distances[:max_results]]
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the current buffer"""
        if not self.buffer:
            return {
                'count': 0,
                'duration_seconds': 0.0,
                'oldest_timestamp': 0.0,
                'newest_timestamp': 0.0
            }
        
        timestamps = [moment.timestamp for moment in self.buffer]
        return {
            'count': len(self.buffer),
            'duration_seconds': max(timestamps) - min(timestamps),
            'oldest_timestamp': min(timestamps),
            'newest_timestamp': max(timestamps),
            'feature_stats': self.feature_stats
        }
    
    def save_to_file(self, filepath: str) -> bool:
        """Save memory buffer and feature stats to file"""
        try:
            # Create data structure for saving
            save_data = {
                'moments': [],
                'feature_stats': self.feature_stats,
                'max_duration_seconds': self.max_duration_seconds,
                'feature_dimensions': self.feature_dimensions,
                'save_timestamp': time.time()
            }
            
            # Convert moments to serializable format
            for moment in self.buffer:
                moment_data = {
                    'timestamp': moment.timestamp,
                    'features': moment.features.tolist(),  # Convert numpy array to list
                    'event_data': moment.event_data,
                    'cluster_id': moment.cluster_id
                }
                save_data['moments'].append(moment_data)
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"💾 Saved {len(self.buffer)} musical moments to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save memory buffer: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load memory buffer and feature stats from file"""
        try:
            if not os.path.exists(filepath):
                print(f"📁 No existing memory file found at {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Restore configuration
            self.max_duration_seconds = save_data.get('max_duration_seconds', 180.0)
            self.feature_dimensions = save_data.get('feature_dimensions', 4)
            self.feature_stats = save_data.get('feature_stats', {
                'centroid': {'mean': 0.0, 'std': 1.0},
                'rms': {'mean': 0.0, 'std': 1.0},
                'cents': {'mean': 0.0, 'std': 1.0},
                'ioi': {'mean': 0.0, 'std': 1.0}
            })
            
            # Restore moments
            self.buffer.clear()
            for moment_data in save_data.get('moments', []):
                moment = MusicalMoment(
                    timestamp=moment_data['timestamp'],
                    features=np.array(moment_data['features']),  # Convert list back to numpy array
                    event_data=moment_data['event_data'],
                    cluster_id=moment_data.get('cluster_id')
                )
                self.buffer.append(moment)
            
            print(f"📂 Loaded {len(self.buffer)} musical moments from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load memory buffer: {e}")
            return False
    
    def merge_from_file(self, filepath: str, max_age_days: float = 30.0) -> bool:
        """Merge moments from file, keeping only recent ones"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            
            # Load moments that are not too old
            loaded_count = 0
            for moment_data in save_data.get('moments', []):
                if current_time - moment_data['timestamp'] <= max_age_seconds:
                    moment = MusicalMoment(
                        timestamp=moment_data['timestamp'],
                        features=np.array(moment_data['features']),
                        event_data=moment_data['event_data'],
                        cluster_id=moment_data.get('cluster_id')
                    )
                    self.buffer.append(moment)
                    loaded_count += 1
            
            # Sort by timestamp to maintain chronological order
            self.buffer = deque(sorted(self.buffer, key=lambda m: m.timestamp))
            
            # Clean up old moments
            self._cleanup_old_moments(current_time)
            
            print(f"🔄 Merged {loaded_count} recent musical moments from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to merge memory buffer: {e}")
            return False
