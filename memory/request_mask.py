#!/usr/bin/env python3
"""
Request Mask for Conditional Generation
Ported from Brandtsegg/Formo probabilistic_logic.py

This module enables goal-directed generation by creating probability masks
that bias generation toward desired qualities (e.g., high consonance, specific tokens).

@author: Øyvind Brandtsegg, Daniel Formo (original)
@adapted_by: Jonas Sjøvaag (CCM3 integration)
@license: GPL
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

class RequestMask:
    """
    Creates probability masks for conditional generation
    
    Supports various request types:
    - Exact match ('==')
    - Thresholds ('>', '<', 'abs >', 'abs <')
    - Gradients ('gradient', 'gr_abs') - power curves favoring high/low values
    """
    
    def __init__(self):
        """Initialize request mask generator"""
        pass
    
    def create_mask(self, 
                   corpus: np.ndarray,
                   parameter_values: np.ndarray,
                   request: Dict,
                   corpus_size: int) -> np.ndarray:
        """
        Create probability mask based on request
        
        Args:
            corpus: Full corpus array (for context, not always used)
            parameter_values: Values of the requested parameter across corpus
            request: Request specification dict with:
                - parameter: Parameter name (e.g., 'consonance', 'gesture_token')
                - type: Request type ('==', '>', '<', 'gradient', 'gr_abs', 'abs >', 'abs <')
                - value: Target value or gradient shape
                - weight: Blend weight (0.0-1.0), 1.0 = hard constraint
            corpus_size: Number of valid events in corpus
            
        Returns:
            Mask array (0.0-1.0 per event)
        """
        request_type = request.get('type', '==')
        value = request.get('value', 0.0)
        tolerance = request.get('tolerance', 0.01)  # Get tolerance from request
        
        mask = np.zeros(corpus_size)
        
        # Exact match (with tolerance for range-based matching)
        if request_type == '==':
            mask = self._create_exact_match_mask(parameter_values, value, corpus_size, tolerance)
        
        # Threshold: greater than
        elif request_type == '>':
            mask = self._create_threshold_mask(parameter_values, value, 'greater', corpus_size)
        
        # Threshold: less than
        elif request_type == '<':
            mask = self._create_threshold_mask(parameter_values, value, 'less', corpus_size)
        
        # Threshold: absolute value greater than
        elif request_type == 'abs >':
            mask = self._create_threshold_mask(np.abs(parameter_values), value, 'greater', corpus_size)
        
        # Threshold: absolute value less than
        elif request_type == 'abs <':
            mask = self._create_threshold_mask(np.abs(parameter_values), value, 'less', corpus_size)
        
        # Gradient: power curve favoring high or low values
        elif request_type == 'gradient':
            mask = self._create_gradient_mask(parameter_values, value, corpus_size, use_abs=False)
        
        # Absolute gradient: power curve on absolute values
        elif request_type == 'gr_abs':
            mask = self._create_gradient_mask(parameter_values, value, corpus_size, use_abs=True)
        
        else:
            raise ValueError(f"Unknown request type: {request_type}")
        
        # If all masks are zero, enable all (no valid options)
        if np.max(mask) == 0:
            mask = np.ones(corpus_size)
        
        return mask
    
    def _create_exact_match_mask(self, 
                                 parameter_values: np.ndarray,
                                 target_value: float,
                                 corpus_size: int,
                                 tolerance: float = 0.01) -> np.ndarray:
        """
        Create mask for exact value matching
        
        Args:
            parameter_values: Parameter values across corpus
            target_value: Target value to match
            corpus_size: Number of valid events
            tolerance: Matching tolerance for floating point values
            
        Returns:
            Binary mask (1.0 for matches, 0.0 otherwise)
        """
        mask = np.zeros(corpus_size)
        
        # For integer-like values (e.g., gesture tokens), use exact match
        if isinstance(target_value, int) or (isinstance(target_value, float) and target_value.is_integer()):
            mask = (parameter_values[:corpus_size] == target_value).astype(float)
        else:
            # For continuous values, use tolerance (inclusive)
            mask = (np.abs(parameter_values[:corpus_size] - target_value) <= tolerance).astype(float)
        
        # If no exact matches, find closest
        if np.sum(mask) == 0:
            closest_idx = np.argmin(np.abs(parameter_values[:corpus_size] - target_value))
            mask[closest_idx] = 1.0
        
        return mask
    
    def _create_threshold_mask(self,
                               parameter_values: np.ndarray,
                               threshold: float,
                               direction: str,
                               corpus_size: int) -> np.ndarray:
        """
        Create binary threshold mask
        
        Args:
            parameter_values: Parameter values across corpus
            threshold: Threshold value
            direction: 'greater' or 'less'
            corpus_size: Number of valid events
            
        Returns:
            Binary mask
        """
        mask = np.zeros(corpus_size)
        
        if direction == 'greater':
            mask = (parameter_values[:corpus_size] > threshold).astype(float)
        elif direction == 'less':
            mask = (parameter_values[:corpus_size] < threshold).astype(float)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        return mask
    
    def _create_gradient_mask(self,
                             parameter_values: np.ndarray,
                             gradient_shape: float,
                             corpus_size: int,
                             use_abs: bool = False) -> np.ndarray:
        """
        Create gradient mask using power curve
        
        Args:
            parameter_values: Parameter values across corpus
            gradient_shape: Shape parameter:
                - Positive: favor high values
                - Negative: favor low values
                - Magnitude controls steepness (higher = steeper)
            corpus_size: Number of valid events
            use_abs: Whether to use absolute values
            
        Returns:
            Continuous mask (0.0-1.0)
        """
        values = parameter_values[:corpus_size].copy()
        
        if use_abs:
            values = np.abs(values)
        
        # Normalize to [0, 1]
        val_min = np.min(values)
        val_max = np.max(values)
        
        if val_max - val_min < 1e-10:
            # All values are the same
            return np.ones(corpus_size)
        
        normalized = (values - val_min) / (val_max - val_min)
        
        # Invert if negative gradient (favor low values)
        if gradient_shape < 0:
            normalized = 1.0 - normalized
        
        # Apply power curve
        mask = np.power(normalized, abs(gradient_shape))
        
        return mask
    
    def blend_with_probability(self,
                               base_probability: np.ndarray,
                               mask: np.ndarray,
                               weight: float) -> np.ndarray:
        """
        Blend request mask with base probability distribution
        
        Args:
            base_probability: Base probability from oracle
            mask: Request mask
            weight: Blend weight (0.0 = ignore mask, 1.0 = hard constraint)
            
        Returns:
            Blended probability distribution
        """
        if weight == 1.0:
            # Hard constraint: multiply (zero out non-matching)
            result = base_probability * mask
        else:
            # Soft constraint: blend
            result = mask * weight + base_probability * (1.0 - weight)
        
        # Renormalize
        total = np.sum(result)
        if total > 0:
            result = result / total
        else:
            # Fallback to uniform
            result = np.ones_like(result) / len(result)
        
        return result


def test_request_mask():
    """Test the request mask functionality"""
    print("🎭 Testing RequestMask...")
    
    rm = RequestMask()
    
    # Create test corpus
    corpus_size = 10
    test_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    base_prob = np.ones(corpus_size) / corpus_size  # Uniform
    
    # Test 1: Exact match
    print("\n📍 Test 1: Exact match (value=0.5)")
    request = {'parameter': 'test', 'type': '==', 'value': 0.5, 'weight': 1.0}
    mask = rm.create_mask(None, test_values, request, corpus_size)
    print(f"   Mask: {mask}")
    print(f"   Selected values: {test_values[mask > 0]}")
    
    # Test 2: Greater than threshold
    print("\n📍 Test 2: Greater than (value=0.6)")
    request = {'parameter': 'test', 'type': '>', 'value': 0.6, 'weight': 1.0}
    mask = rm.create_mask(None, test_values, request, corpus_size)
    print(f"   Mask: {mask}")
    print(f"   Selected values: {test_values[mask > 0]}")
    
    # Test 3: Gradient (favor high values)
    print("\n📍 Test 3: Gradient (shape=2, favor high)")
    request = {'parameter': 'test', 'type': 'gradient', 'value': 2.0, 'weight': 1.0}
    mask = rm.create_mask(None, test_values, request, corpus_size)
    print(f"   Mask: {mask}")
    print(f"   Probability weights: {mask / np.sum(mask)}")
    
    # Test 4: Gradient (favor low values)
    print("\n📍 Test 4: Gradient (shape=-2, favor low)")
    request = {'parameter': 'test', 'type': 'gradient', 'value': -2.0, 'weight': 1.0}
    mask = rm.create_mask(None, test_values, request, corpus_size)
    print(f"   Mask: {mask}")
    print(f"   Probability weights: {mask / np.sum(mask)}")
    
    # Test 5: Blending with base probability
    print("\n📍 Test 5: Soft constraint (weight=0.5)")
    request = {'parameter': 'test', 'type': '>', 'value': 0.7, 'weight': 0.5}
    mask = rm.create_mask(None, test_values, request, corpus_size)
    blended = rm.blend_with_probability(base_prob, mask, request['weight'])
    print(f"   Base probability: {base_prob}")
    print(f"   Mask: {mask}")
    print(f"   Blended result: {blended}")
    
    print("\n✅ RequestMask tests complete!")


if __name__ == "__main__":
    test_request_mask()

