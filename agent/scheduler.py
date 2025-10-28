# scheduler.py
# Behavior scheduling and timing control for AI agent

import time
import random
from typing import Dict
from .behaviors import MusicalDecision

class BehaviorScheduler:
    """
    Schedules and manages AI agent behavior timing
    Controls initiative budget, density, and musical flow
    """
    
    def __init__(self):
        self.initiative_budget = 0.8  # 0.0 to 1.0 (balanced responsiveness)
        self.density_level = 0.4      # 0.0 to 1.0 (moderate activity)
        self.give_space_factor = 0.3  # How much to give space
        
        # Timing controls - TUNED for less frantic response
        self.min_decision_interval = 0.5  # seconds (increased for more space)
        self.max_decision_interval = 3.0  # seconds
        self.last_decision_time = time.time()
        
        # Initiative tracking
        self.initiative_used = 0.0
        self.initiative_recharge_rate = 0.2  # per second (increased from 0.1)
        self.max_initiative = 1.0
        
        # Density tracking
        self.recent_decisions = []
        self.density_window = 10.0  # seconds
        
    def should_make_decision(self) -> bool:
        """Determine if the agent should make a musical decision now"""
        current_time = time.time()
        
        # Check minimum interval
        if current_time - self.last_decision_time < self.min_decision_interval:
            return False
        
        # Check initiative budget
        if self.initiative_used >= self.initiative_budget:
            return False
        
        # Check density (don't overcrowd)
        if self._is_too_dense(current_time):
            return False
        
        # Random chance based on density
        decision_probability = self._calculate_decision_probability()
        return random.random() < decision_probability
    
    def _is_too_dense(self, current_time: float) -> bool:
        """Check if recent decisions are too dense"""
        # Remove old decisions
        cutoff_time = current_time - self.density_window
        self.recent_decisions = [d for d in self.recent_decisions if d['timestamp'] > cutoff_time]
        
        # Check density
        decision_count = len(self.recent_decisions)
        max_decisions = int(self.density_window * (0.5 + self.density_level * 0.5))
        
        return decision_count >= max_decisions
    
    def _calculate_decision_probability(self) -> float:
        """Calculate probability of making a decision"""
        base_probability = 0.25  # Increased from 0.1 for better responsiveness
        
        # Adjust based on density level
        density_factor = 0.4 + self.density_level * 0.4  # Increased range
        
        # Adjust based on initiative
        initiative_factor = 0.3 + (self.initiative_budget - self.initiative_used) * 0.4  # Reduced range
        
        # Adjust based on give space factor (less aggressive)
        # Clamp give_space_factor to valid range [0, 1]
        clamped_give_space = max(0.0, min(1.0, self.give_space_factor))
        space_factor = 1.0 - clamped_give_space * 0.5  # Reduced from 0.8 to 0.5
        
        # Ensure result is non-negative
        result = base_probability * density_factor * initiative_factor * space_factor
        return max(0.0, result)
    
    def record_decision(self, decision: MusicalDecision):
        """Record a decision for density tracking"""
        current_time = time.time()
        
        # Update timing
        self.last_decision_time = current_time
        
        # Record decision
        self.recent_decisions.append({
            'timestamp': current_time,
            'mode': decision.mode,
            'confidence': decision.confidence
        })
        
        # Use initiative
        initiative_cost = self._calculate_initiative_cost(decision)
        self.initiative_used += initiative_cost
    
    def _calculate_initiative_cost(self, decision: MusicalDecision) -> float:
        """Calculate initiative cost for a decision"""
        base_cost = 0.1
        
        # Higher confidence decisions cost more
        confidence_factor = 0.5 + decision.confidence * 0.5
        
        # Lead decisions cost more than imitate
        mode_costs = {
            'imitate': 0.8,
            'contrast': 1.0,
            'lead': 1.2
        }
        
        mode_factor = mode_costs.get(decision.mode.value, 1.0)
        
        return base_cost * confidence_factor * mode_factor
    
    def update_initiative(self):
        """Update initiative budget over time"""
        current_time = time.time()
        time_delta = current_time - getattr(self, '_last_update_time', current_time)
        self._last_update_time = current_time
        
        # Recharge initiative
        self.initiative_used = max(0.0, self.initiative_used - 
                                 self.initiative_recharge_rate * time_delta)
    
    def set_density_level(self, level: float):
        """Set density level (0.0 = sparse, 1.0 = dense)"""
        self.density_level = max(0.0, min(1.0, level))
    
    def set_give_space_factor(self, factor: float):
        """Set give space factor (0.0 = active, 1.0 = giving space)"""
        self.give_space_factor = max(0.0, min(1.0, factor))
    
    def set_initiative_budget(self, budget: float):
        """Set initiative budget (0.0 to 1.0)"""
        self.initiative_budget = max(0.0, min(1.0, budget))
    
    def get_status(self) -> Dict:
        """Get current scheduler status"""
        return {
            'initiative_budget': self.initiative_budget,
            'initiative_used': self.initiative_used,
            'density_level': self.density_level,
            'give_space_factor': self.give_space_factor,
            'recent_decisions_count': len(self.recent_decisions),
            'last_decision_time': self.last_decision_time,
            'time_since_last_decision': time.time() - self.last_decision_time
        }
