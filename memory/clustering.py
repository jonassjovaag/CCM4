# clustering.py
# Clustering system for musical moment classification

import numpy as np
import json
import pickle
import os
import time
from sklearn.cluster import KMeans
from typing import List, Dict, Optional, Tuple
from .memory_buffer import MusicalMoment, MemoryBuffer, NumpyEncoder

class MusicalClustering:
    """
    Clustering system for musical moments using k-means
    Provides similarity-based queries for AI agent
    """
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.is_trained = False
        
        # Cluster metadata
        self.cluster_stats = {}
        self.cluster_examples = {}
        
    def train(self, memory_buffer: MemoryBuffer, min_samples: int = 50) -> bool:
        """Train clustering model on memory buffer data"""
        moments = memory_buffer.get_all_moments()
        
        if len(moments) < min_samples:
            return False
        
        # Get feature matrix
        feature_matrix = memory_buffer.get_feature_matrix(moments)
        
        if feature_matrix.shape[0] < min_samples:
            return False
        
        # Train k-means
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(moments)),
            random_state=self.random_state,
            n_init=10
        )
        
        self.kmeans.fit(feature_matrix)
        self.cluster_centers = self.kmeans.cluster_centers_
        self.is_trained = True
        
        # Assign cluster IDs to moments
        cluster_labels = self.kmeans.labels_
        for i, moment in enumerate(moments):
            moment.cluster_id = cluster_labels[i]
        
        # Update cluster statistics
        self._update_cluster_stats(moments)
        
        return True
    
    def _update_cluster_stats(self, moments: List[MusicalMoment]):
        """Update statistics for each cluster"""
        self.cluster_stats = {}
        self.cluster_examples = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_moments = [m for m in moments if m.cluster_id == cluster_id]
            
            if cluster_moments:
                # Calculate cluster statistics
                features = np.array([m.features for m in cluster_moments])
                
                self.cluster_stats[cluster_id] = {
                    'count': len(cluster_moments),
                    'mean_features': np.mean(features, axis=0),
                    'std_features': np.std(features, axis=0),
                    'time_span': max(m.timestamp for m in cluster_moments) - 
                               min(m.timestamp for m in cluster_moments)
                }
                
                # Store representative examples
                self.cluster_examples[cluster_id] = cluster_moments[:5]
    
    def predict_cluster(self, features: np.ndarray) -> int:
        """Predict cluster for given features"""
        if not self.is_trained:
            return 0
        
        return self.kmeans.predict(features.reshape(1, -1))[0]
    
    def get_cluster_center(self, cluster_id: int) -> Optional[np.ndarray]:
        """Get cluster center features"""
        if not self.is_trained or cluster_id >= self.n_clusters:
            return None
        
        return self.cluster_centers[cluster_id]
    
    def find_similar_clusters(self, target_features: np.ndarray, 
                            radius: float = 1.0) -> List[int]:
        """Find clusters similar to target features"""
        if not self.is_trained:
            return []
        
        similar_clusters = []
        for cluster_id in range(self.n_clusters):
            center = self.cluster_centers[cluster_id]
            distance = np.linalg.norm(center - target_features)
            if distance <= radius:
                similar_clusters.append(cluster_id)
        
        return similar_clusters
    
    def find_contrasting_clusters(self, target_features: np.ndarray,
                                min_distance: float = 2.0) -> List[int]:
        """Find clusters that contrast with target features"""
        if not self.is_trained:
            return []
        
        contrasting_clusters = []
        for cluster_id in range(self.n_clusters):
            center = self.cluster_centers[cluster_id]
            distance = np.linalg.norm(center - target_features)
            if distance >= min_distance:
                contrasting_clusters.append(cluster_id)
        
        return contrasting_clusters
    
    def get_cluster_moments(self, cluster_id: int, 
                          memory_buffer: MemoryBuffer) -> List[MusicalMoment]:
        """Get all moments belonging to a specific cluster"""
        all_moments = memory_buffer.get_all_moments()
        return [m for m in all_moments if m.cluster_id == cluster_id]
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        """Get information about a specific cluster"""
        if cluster_id not in self.cluster_stats:
            return None
        
        return {
            'id': cluster_id,
            'stats': self.cluster_stats[cluster_id],
            'examples': self.cluster_examples.get(cluster_id, []),
            'center': self.cluster_centers[cluster_id] if self.is_trained else None
        }
    
    def get_all_cluster_info(self) -> Dict[int, Dict]:
        """Get information about all clusters"""
        return {cluster_id: self.get_cluster_info(cluster_id) 
                for cluster_id in range(self.n_clusters)}
    
    def retrain_if_needed(self, memory_buffer: MemoryBuffer, 
                         retrain_threshold: int = 100) -> bool:
        """Retrain clustering if buffer has grown significantly"""
        moments = memory_buffer.get_all_moments()
        
        if len(moments) >= retrain_threshold and not self.is_trained:
            return self.train(memory_buffer)
        
        return False
    
    def save_to_file(self, filepath: str) -> bool:
        """Save clustering model and metadata to file"""
        try:
            if not self.is_trained:
                print("⚠️ No trained clustering model to save")
                return False
            
            # Create save data structure
            save_data = {
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'is_trained': self.is_trained,
                'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
                'cluster_stats': self.cluster_stats,
                'cluster_examples': self._serialize_cluster_examples(),
                'save_timestamp': time.time()
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"💾 Saved clustering model with {self.n_clusters} clusters to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save clustering model: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load clustering model and metadata from file"""
        try:
            if not os.path.exists(filepath):
                print(f"📁 No existing clustering file found at {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Restore basic parameters
            self.n_clusters = save_data.get('n_clusters', 8)
            self.random_state = save_data.get('random_state', 42)
            self.is_trained = save_data.get('is_trained', False)
            
            # Restore cluster centers
            cluster_centers_data = save_data.get('cluster_centers')
            if cluster_centers_data is not None:
                self.cluster_centers = np.array(cluster_centers_data)
            else:
                self.cluster_centers = None
            
            # Restore cluster statistics and examples
            self.cluster_stats = save_data.get('cluster_stats', {})
            self.cluster_examples = self._deserialize_cluster_examples(
                save_data.get('cluster_examples', {})
            )
            
            # Recreate the KMeans model (we can't pickle sklearn models in JSON)
            if self.is_trained and self.cluster_centers is not None:
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=10
                )
                # Set the cluster centers manually
                self.kmeans.cluster_centers_ = self.cluster_centers
                self.kmeans.labels_ = np.zeros(len(self.cluster_centers))  # Dummy labels
            
            print(f"📂 Loaded clustering model with {self.n_clusters} clusters from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load clustering model: {e}")
            return False
    
    def _serialize_cluster_examples(self) -> Dict:
        """Convert cluster examples to serializable format"""
        serialized = {}
        for cluster_id, examples in self.cluster_examples.items():
            serialized[str(cluster_id)] = []
            for moment in examples:
                serialized[str(cluster_id)].append({
                    'timestamp': moment.timestamp,
                    'features': moment.features.tolist(),
                    'event_data': moment.event_data,
                    'cluster_id': moment.cluster_id
                })
        return serialized
    
    def _deserialize_cluster_examples(self, serialized_data: Dict) -> Dict:
        """Convert serialized cluster examples back to MusicalMoment objects"""
        deserialized = {}
        for cluster_id_str, examples_data in serialized_data.items():
            cluster_id = int(cluster_id_str)
            deserialized[cluster_id] = []
            for moment_data in examples_data:
                moment = MusicalMoment(
                    timestamp=moment_data['timestamp'],
                    features=np.array(moment_data['features']),
                    event_data=moment_data['event_data'],
                    cluster_id=moment_data.get('cluster_id')
                )
                deserialized[cluster_id].append(moment)
        return deserialized
