# ai_agent.py
# Main AI Agent coordinator for Drift Engine

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .behaviors import BehaviorEngine, MusicalDecision
from .scheduler import BehaviorScheduler
from .density import DensityController

@dataclass
class AgentState:
    """Current state of the AI agent"""
    is_active: bool
    current_mode: str
    confidence: float
    last_decision_time: float
    decisions_made: int
    activity_level: float

class AIAgent:
    """
    Main AI Agent coordinator
    Integrates behavior engine, scheduler, and density control
    """
    
    def __init__(self, rhythm_oracle=None, visualization_manager=None, config=None):
        self.behavior_engine = BehaviorEngine(rhythm_oracle, visualization_manager=visualization_manager, config=config)
        self.scheduler = BehaviorScheduler()
        self.density_controller = DensityController()
        self.visualization_manager = visualization_manager
        self.config = config
        
        # Agent state
        self.is_active = True
        self.decisions_made = 0
        self.last_decision_time = 0.0
        
        # Callbacks
        self.on_decision_made: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'imitate_decisions': 0,
            'contrast_decisions': 0,
            'lead_decisions': 0,
            'avg_confidence': 0.0,
            'uptime': 0.0
        }
        
        self.start_time = time.time()
    
    def process_event(self, event_data: Dict, memory_buffer, clustering, 
                     activity_multiplier: float = 1.0, arc_context: Optional[Dict] = None) -> List[MusicalDecision]:
        """Process an audio event and make musical decisions
        
        Args:
            activity_multiplier: Performance arc activity level (0.0-1.0) from timeline manager
            arc_context: Performance arc context (phase, engagement_level, etc.)
        """
        if not self.is_active:
            return []
        
        current_time = time.time()
        
        # Update density controller
        self.density_controller.update_activity(event_data)
        
        # Update scheduler
        self.scheduler.update_initiative()
        
        # AUTONOMOUS MODE: Skip scheduler check if autonomous generation enabled
        # In autonomous mode, PhraseGenerator handles its own timing via should_respond()
        if not (hasattr(self.behavior_engine, 'phrase_generator') and 
                self.behavior_engine.phrase_generator.autonomous_mode):
            # Check if we should make a decision (human-reactive mode)
            if not self.scheduler.should_make_decision():
                return []
        
        # Make decisions (melodic and bass) with activity multiplier and arc context
        decisions = self.behavior_engine.decide_behavior(
            event_data, memory_buffer, clustering, activity_multiplier, arc_context
        )
        
        # Record decisions
        for decision in decisions:
            self.scheduler.record_decision(decision)
            self._update_stats(decision)
        
        # Update agent state
        self.last_decision_time = current_time
        self.decisions_made += len(decisions)
        
        # Call callback if set
        if self.on_decision_made:
            for decision in decisions:
                self.on_decision_made(decision)
        
        return decisions
    
    def _update_stats(self, decision: MusicalDecision):
        """Update agent statistics"""
        self.stats['total_decisions'] += 1
        
        if decision.mode.value == 'imitate':
            self.stats['imitate_decisions'] += 1
        elif decision.mode.value == 'contrast':
            self.stats['contrast_decisions'] += 1
        elif decision.mode.value == 'lead':
            self.stats['lead_decisions'] += 1
        
        # Update average confidence
        total = self.stats['total_decisions']
        current_avg = self.stats['avg_confidence']
        self.stats['avg_confidence'] = (current_avg * (total - 1) + decision.confidence) / total
        
        # Update uptime
        self.stats['uptime'] = time.time() - self.start_time
    
    def set_active(self, active: bool):
        """Set agent active/inactive"""
        self.is_active = active
    
    def set_density_level(self, level: float):
        """Set density level (0.0 = sparse, 1.0 = dense)"""
        self.density_controller.set_target_density(level)
        self.scheduler.set_density_level(level)
    
    def set_give_space_factor(self, factor: float):
        """Set give space factor (0.0 = active, 1.0 = giving space)"""
        self.scheduler.set_give_space_factor(factor)
    
    def set_initiative_budget(self, budget: float):
        """Set initiative budget (0.0 to 1.0)"""
        self.scheduler.set_initiative_budget(budget)
    
    def get_agent_state(self) -> AgentState:
        """Get current agent state"""
        return AgentState(
            is_active=self.is_active,
            current_mode=self.behavior_engine.get_current_mode().value,
            confidence=self.stats['avg_confidence'],
            last_decision_time=self.last_decision_time,
            decisions_made=self.decisions_made,
            activity_level=self.density_controller.current_density
        )
    
    def get_density_recommendation(self) -> Dict:
        """Get density recommendation"""
        return self.density_controller.get_density_recommendation()
    
    def get_scheduler_status(self) -> Dict:
        """Get scheduler status"""
        return self.scheduler.get_status()
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return self.stats.copy()
    
    def get_activity_summary(self, duration_seconds: float = 30.0) -> Dict:
        """Get activity summary"""
        return self.density_controller.get_activity_summary(duration_seconds)
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.stats = {
            'total_decisions': 0,
            'imitate_decisions': 0,
            'contrast_decisions': 0,
            'lead_decisions': 0,
            'avg_confidence': 0.0,
            'uptime': 0.0
        }
        self.start_time = time.time()
        self.decisions_made = 0
