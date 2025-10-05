"""
AutoML Optimizer - Optuna-based hyperparameter optimization
Automated optimization for CivicMindAI response quality and performance
"""

import optuna
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AutoMLOptimizer:
    def __init__(self):
        """Initialize AutoML Optimizer with Optuna"""
        self.study = optuna.create_study(
            direction='maximize',
            study_name='CivicMindAI_Optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimization parameters
        self.current_params = self._initialize_default_params()
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
        # Response quality tracking
        self.response_evaluations = []
        self.optimization_frequency = 50  # Optimize every 50 responses
        self.response_count = 0
        
        # Best parameters tracking
        self.best_params = self.current_params.copy()
        self.best_score = 0.0
        
    def _initialize_default_params(self) -> Dict[str, Any]:
        """Initialize default optimization parameters"""
        return {
            # Response generation parameters
            "rag_weight": 0.5,           # Weight for RAG responses
            "kag_weight": 0.3,           # Weight for KAG responses  
            "cag_weight": 0.2,           # Weight for CAG responses
            
            # Quality thresholds
            "min_confidence_threshold": 0.7,    # Minimum confidence to show response
            "relevance_threshold": 0.6,         # Minimum relevance score
            "completeness_threshold": 0.8,      # Completeness requirement
            
            # Response optimization
            "max_response_length": 500,          # Maximum response length
            "min_response_length": 50,           # Minimum response length
            "include_contact_probability": 0.8,  # Probability of including contact info
            
            # Caching parameters
            "cache_hit_boost": 0.15,            # Boost for cache hits
            "fresh_data_penalty": -0.05,        # Penalty for very fresh data
            
            # Personalization parameters
            "location_relevance_boost": 0.2,    # Boost for location-relevant responses
            "specialization_boost": 0.15,       # Boost for specialized responses
            
            # Emergency handling
            "emergency_priority_multiplier": 2.0,   # Priority multiplier for emergency queries
            "emergency_response_time_target": 2.0,  # Target response time for emergencies (seconds)
            
            # Learning parameters
            "learning_rate": 0.01,               # Learning rate for parameter updates
            "momentum": 0.9,                     # Momentum for parameter updates
            "regularization": 0.001              # L2 regularization strength
        }
    
    def optimize_response(self, response_data: Dict[str, Any], query: str) -> str:
        """Optimize response based on current parameters and query context"""
        
        self.response_count += 1
        if response_data is None:
            response_data = {"response": ""}
        else:
            try:
                original_response = response_data.get("response", "")
            except AttributeError:
                response_data = {"response": str(response_data)}
        
        # Extract response components
        original_response = response_data.get("response", "")
        
        # Apply current optimization parameters
        optimized_response = self._apply_optimization_rules(original_response, query, response_data)
        
        # Evaluate response quality
        quality_score = self._evaluate_response_quality(optimized_response, query, response_data)
        
        # Store evaluation for optimization
        self.response_evaluations.append({
            "query": query,
            "response": optimized_response,
            "quality_score": quality_score,
            "parameters_used": self.current_params.copy(),
            "timestamp": datetime.now()
        })
        
        # Trigger optimization if needed
        if self.response_count % self.optimization_frequency == 0:
            self._trigger_hyperparameter_optimization()
        
        return optimized_response
    
    def _apply_optimization_rules(self, response: str, query: str, response_data: Dict[str, Any]) -> str:
        """Apply optimization rules to improve response quality"""
        
        optimized_response = response
        query_lower = query.lower()
        
        # Rule 1: Response length optimization
        if len(response.split()) > self.current_params["max_response_length"]:
            # Truncate and add continuation
            words = response.split()[:self.current_params["max_response_length"]]
            optimized_response = " ".join(words) + "..."
        elif len(response.split()) < self.current_params["min_response_length"]:
            # Add helpful information
            optimized_response += self._generate_helpful_addition(query, response_data)
        
        # Rule 2: Contact information inclusion
        if (np.random.random() < self.current_params["include_contact_probability"] and 
            "contact" not in response.lower()):
            contact_info = self._get_relevant_contact(query)
            if contact_info:
                optimized_response += f"\n\n📞 **Contact**: {contact_info}"
        
        # Rule 3: Emergency query handling
        if any(word in query_lower for word in ["emergency", "urgent", "fire", "ambulance"]):
            emergency_multiplier = self.current_params["emergency_priority_multiplier"]
            optimized_response = self._prioritize_emergency_response(optimized_response, emergency_multiplier)
        
        # Rule 4: Location relevance boost
        if response_data.get("location_detected"):
            location_boost = self.current_params["location_relevance_boost"]
            optimized_response = self._add_location_context(optimized_response, response_data, location_boost)
        
        # Rule 5: Specialization enhancement
        if response_data.get("specialization_match"):
            spec_boost = self.current_params["specialization_boost"] 
            optimized_response = self._enhance_specialized_response(optimized_response, spec_boost)
        
        # Rule 6: Cache optimization
        if response_data.get("from_cache"):
            cache_boost = self.current_params["cache_hit_boost"]
            optimized_response = self._optimize_cached_response(optimized_response, cache_boost)
        
        return optimized_response
    
    def _generate_helpful_addition(self, query: str, response_data: Dict[str, Any]) -> str:
        """Generate helpful additional information for short responses"""
        
        query_lower = query.lower()
        additions = []
        
        if "property tax" in query_lower:
            additions.append("💡 **Tip**: Pay before due date to avoid penalties.")
        elif "water" in query_lower:
            additions.append("💡 **Tip**: Report multiple issues in one complaint for faster resolution.")
        elif "electricity" in query_lower:
            additions.append("💡 **Tip**: Keep your consumer number handy when calling.")
        elif "garbage" in query_lower:
            additions.append("💡 **Tip**: Proper waste segregation helps faster collection.")
        else:
            additions.append("💡 **Tip**: Keep relevant documents ready when visiting offices.")
        
        return " " + np.random.choice(additions) if additions else ""
    
    def _get_relevant_contact(self, query: str) -> Optional[str]:
        """Get relevant contact information based on query"""
        
        query_lower = query.lower()
        
        contact_mapping = {
            "water": "Metro Water: 044-45671200",
            "electricity": "TNEB: 94987-94987", 
            "property": "Revenue Dept: 044-25619515",
            "garbage": "Corporation: 1913",
            "emergency": "Emergency: Fire-101, Police-100, Ambulance-108",
            "general": "Chennai Corporation: 1913"
        }
        
        for keyword, contact in contact_mapping.items():
            if keyword in query_lower:
                return contact
        
        return contact_mapping["general"]
    
    def _prioritize_emergency_response(self, response: str, multiplier: float) -> str:
        """Prioritize and enhance emergency responses"""
        
        if not response.startswith("🚨"):
            response = f"🚨 **URGENT** - {response}"
        
        # Add immediate action items for emergencies
        emergency_actions = [
            "📞 **Call immediately for urgent assistance**",
            "🏥 **Seek immediate help if needed**", 
            "⚡ **Priority handling for emergency situations**"
        ]
        
        action = np.random.choice(emergency_actions)
        return f"{response}\n\n{action}"
    
    def _add_location_context(self, response: str, response_data: Dict[str, Any], boost: float) -> str:
        """Add location-specific context to responses"""
        
        location = response_data.get("user_area", "")
        if location and "location" not in response.lower():
            return f"{response}\n\n📍 **Location-specific**: This information applies to {location} area."
        
        return response
    
    def _enhance_specialized_response(self, response: str, boost: float) -> str:
        """Enhance responses when specialization match is found"""
        
        if "specialized" not in response.lower():
            return f"{response}\n\n🎯 **Specialized Service**: Enhanced support available for this area."
        
        return response
    
    def _optimize_cached_response(self, response: str, boost: float) -> str:
        """Optimize responses from cache with freshness indicators"""
        
        if boost > 0.1:  # High cache boost means very relevant
            return f"{response}\n\n⚡ **Fast Response**: Retrieved from optimized knowledge base."
        
        return response
    
    def _evaluate_response_quality(self, response: str, query: str, response_data: Dict[str, Any]) -> float:
        """Evaluate response quality using multiple metrics"""
        
        quality_components = {}
        
        # 1. Length appropriateness (0-1)
        response_length = len(response.split())
        if 50 <= response_length <= 500:
            quality_components["length"] = 1.0
        elif response_length < 50:
            quality_components["length"] = response_length / 50.0
        else:
            quality_components["length"] = max(0.5, 500 / response_length)
        
        # 2. Information completeness (0-1)
        completeness_indicators = [
            "contact" in response.lower(),
            "department" in response.lower() or "office" in response.lower(),
            any(char.isdigit() for char in response),  # Contains numbers (contacts, fees, etc.)
            "process" in response.lower() or "step" in response.lower(),
        ]
        quality_components["completeness"] = sum(completeness_indicators) / len(completeness_indicators)
        
        # 3. Query relevance (0-1)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split()) 
        overlap = len(query_words.intersection(response_words))
        quality_components["relevance"] = min(1.0, overlap / max(len(query_words), 1))
        
        # 4. Helpfulness indicators (0-1)
        helpful_indicators = [
            "tip" in response.lower(),
            "note" in response.lower(),
            "contact" in response.lower(),
            "portal" in response.lower() or "website" in response.lower(),
            "📞" in response or "💡" in response or "🌐" in response,
        ]
        quality_components["helpfulness"] = sum(helpful_indicators) / len(helpful_indicators)
        
        # 5. Emergency handling (bonus for emergency queries)
        emergency_bonus = 0.0
        if any(word in query.lower() for word in ["emergency", "urgent", "fire", "ambulance"]):
            if "🚨" in response or "urgent" in response.lower():
                emergency_bonus = 0.2
        
        # 6. Specialization bonus
        specialization_bonus = 0.0
        if response_data.get("specialization_match", False):
            if "specialized" in response.lower() or "🎯" in response:
                specialization_bonus = 0.1
        
        # Weighted combination
        weights = {
            "length": 0.2,
            "completeness": 0.3,
            "relevance": 0.3,
            "helpfulness": 0.2
        }
        
        base_score = sum(quality_components[component] * weights[component] 
                        for component in weights)
        
        final_score = min(1.0, base_score + emergency_bonus + specialization_bonus)
        
        return final_score
    
    def _trigger_hyperparameter_optimization(self):
        """Trigger Optuna-based hyperparameter optimization"""
        
        if len(self.response_evaluations) < 20:  # Need minimum data
            return
        
        try:
            # Run optimization
            self.study.optimize(self._objective_function, n_trials=10, timeout=30)
            
            # Update current parameters with best trial
            best_trial = self.study.best_trial
            if best_trial.value > self.best_score:
                self.best_score = best_trial.value
                self.best_params = best_trial.params
                self.current_params.update(best_trial.params)
                
                # Log optimization result
                self.optimization_history.append({
                    "timestamp": datetime.now(),
                    "best_score": self.best_score,
                    "best_params": self.best_params.copy(),
                    "trial_count": len(self.study.trials)
                })
        
        except Exception as e:
            # Log error but continue operation
            print(f"Optimization error: {e}")
    
    def _objective_function(self, trial) -> float:
        """Optuna objective function to maximize response quality"""
        
        # Define hyperparameter search space
        params = {
            "rag_weight": trial.suggest_float("rag_weight", 0.3, 0.7),
            "kag_weight": trial.suggest_float("kag_weight", 0.2, 0.4),
            "cag_weight": trial.suggest_float("cag_weight", 0.1, 0.3),
            "min_confidence_threshold": trial.suggest_float("min_confidence_threshold", 0.5, 0.9),
            "relevance_threshold": trial.suggest_float("relevance_threshold", 0.4, 0.8),
            "completeness_threshold": trial.suggest_float("completeness_threshold", 0.6, 0.9),
            "max_response_length": trial.suggest_int("max_response_length", 300, 700),
            "min_response_length": trial.suggest_int("min_response_length", 30, 100),
            "include_contact_probability": trial.suggest_float("include_contact_probability", 0.5, 1.0),
            "cache_hit_boost": trial.suggest_float("cache_hit_boost", 0.05, 0.25),
            "location_relevance_boost": trial.suggest_float("location_relevance_boost", 0.1, 0.3),
            "specialization_boost": trial.suggest_float("specialization_boost", 0.05, 0.25),
            "emergency_priority_multiplier": trial.suggest_float("emergency_priority_multiplier", 1.5, 3.0),
        }
        
        # Constraint: weights should sum approximately to 1
        weight_sum = params["rag_weight"] + params["kag_weight"] + params["cag_weight"]
        if abs(weight_sum - 1.0) > 0.1:
            return 0.0  # Invalid parameter combination
        
        # Evaluate parameters on recent responses
        recent_evaluations = self.response_evaluations[-50:]  # Last 50 responses
        if len(recent_evaluations) < 10:
            return 0.0
        
        # Simulate applying these parameters
        total_score = 0.0
        valid_evaluations = 0
        
        for eval_data in recent_evaluations:
            # Simulate response optimization with trial parameters
            simulated_response = self._simulate_response_optimization(
                eval_data["response"], 
                eval_data["query"],
                params
            )
            
            # Evaluate simulated response
            simulated_score = self._evaluate_response_quality(
                simulated_response,
                eval_data["query"], 
                {}
            )
            
            total_score += simulated_score
            valid_evaluations += 1
        
        if valid_evaluations == 0:
            return 0.0
        
        average_score = total_score / valid_evaluations
        
        # Add regularization to prevent overfitting
        regularization_penalty = sum(
            abs(params[key] - self._initialize_default_params()[key]) * 0.01
            for key in params if key in self._initialize_default_params()
        )
        
        return average_score - regularization_penalty
    
    def _simulate_response_optimization(self, response: str, query: str, params: Dict[str, Any]) -> str:
        """Simulate response optimization with given parameters"""
        
        # Apply length constraints
        words = response.split()
        if len(words) > params["max_response_length"]:
            response = " ".join(words[:params["max_response_length"]]) + "..."
        elif len(words) < params["min_response_length"]:
            response += " Additional information available upon request."
        
        # Apply contact inclusion probability
        if (np.random.random() < params["include_contact_probability"] and 
            "contact" not in response.lower()):
            response += " Contact: 1913 for assistance."
        
        # Apply emergency handling
        if any(word in query.lower() for word in ["emergency", "urgent"]):
            if params["emergency_priority_multiplier"] > 2.0:
                response = f"🚨 URGENT - {response}"
        
        return response
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance"""
        
        if not self.optimization_history:
            return {
                "message": "No optimization runs completed yet",
                "current_params": self.current_params,
                "responses_evaluated": len(self.response_evaluations)
            }
        
        # Calculate improvement metrics
        recent_scores = [eval_data["quality_score"] for eval_data in self.response_evaluations[-50:]]
        overall_scores = [eval_data["quality_score"] for eval_data in self.response_evaluations]
        
        improvement_trend = []
        if len(overall_scores) >= 100:
            # Compare first 50 vs last 50 scores
            early_avg = np.mean(overall_scores[:50])
            recent_avg = np.mean(overall_scores[-50:])
            improvement_trend.append({
                "period": "Early vs Recent",
                "early_average": round(early_avg, 3),
                "recent_average": round(recent_avg, 3),
                "improvement": round(recent_avg - early_avg, 3)
            })
        
        # Parameter sensitivity analysis
        param_impact = self._analyze_parameter_impact()
        
        return {
            "optimization_summary": {
                "total_trials": len(self.study.trials) if hasattr(self.study, 'trials') else 0,
                "best_score": round(self.best_score, 3),
                "optimization_runs": len(self.optimization_history),
                "responses_evaluated": len(self.response_evaluations)
            },
            "performance_metrics": {
                "current_average_quality": round(np.mean(recent_scores), 3) if recent_scores else 0,
                "quality_std": round(np.std(recent_scores), 3) if recent_scores else 0,
                "improvement_trend": improvement_trend
            },
            "best_parameters": self.best_params,
            "current_parameters": self.current_params,
            "parameter_impact": param_impact,
            "next_optimization_in": max(0, self.optimization_frequency - (self.response_count % self.optimization_frequency))
        }
    
    def _analyze_parameter_impact(self) -> Dict[str, float]:
        """Analyze which parameters have the most impact on quality"""
        
        if len(self.optimization_history) < 3:
            return {"message": "Need more optimization history for analysis"}
        
        # Simple impact analysis based on parameter changes
        param_impact = {}
        
        for i in range(1, len(self.optimization_history)):
            current = self.optimization_history[i]
            previous = self.optimization_history[i-1]
            
            score_change = current["best_score"] - previous["best_score"]
            
            for param_name in current["best_params"]:
                if param_name in previous["best_params"]:
                    param_change = abs(current["best_params"][param_name] - previous["best_params"][param_name])
                    if param_change > 0:
                        impact = abs(score_change) / param_change
                        if param_name not in param_impact:
                            param_impact[param_name] = []
                        param_impact[param_name].append(impact)
        
        # Average impact per parameter
        avg_impact = {}
        for param_name, impacts in param_impact.items():
            avg_impact[param_name] = round(np.mean(impacts), 4)
        
        # Sort by impact
        sorted_impact = dict(sorted(avg_impact.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_impact
    
    def manual_parameter_update(self, param_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Manually update optimization parameters"""
        
        updated_params = []
        invalid_params = []
        
        for param_name, param_value in param_updates.items():
            if param_name in self.current_params:
                old_value = self.current_params[param_name]
                self.current_params[param_name] = param_value
                updated_params.append({
                    "parameter": param_name,
                    "old_value": old_value,
                    "new_value": param_value
                })
            else:
                invalid_params.append(param_name)
        
        # Log manual update
        if updated_params:
            self.optimization_history.append({
                "timestamp": datetime.now(),
                "type": "manual_update",
                "updates": updated_params,
                "current_params": self.current_params.copy()
            })
        
        return {
            "success": len(updated_params) > 0,
            "updated_parameters": updated_params,
            "invalid_parameters": invalid_params,
            "current_parameters": self.current_params
        }
    
    def reset_optimization(self) -> Dict[str, Any]:
        """Reset optimization to default parameters"""
        
        old_params = self.current_params.copy()
        self.current_params = self._initialize_default_params()
        self.best_params = self.current_params.copy()
        self.best_score = 0.0
        
        # Clear history but keep evaluations for learning
        self.optimization_history = [{
            "timestamp": datetime.now(),
            "type": "reset",
            "message": "Optimization reset to default parameters"
        }]
        
        return {
            "success": True,
            "message": "Optimization parameters reset to defaults",
            "previous_parameters": old_params,
            "current_parameters": self.current_params
        }
    
    def export_optimization_data(self) -> Dict[str, Any]:
        """Export comprehensive optimization data for analysis"""
        
        return {
            "export_timestamp": datetime.now().isoformat(),
            "optimization_summary": {
                "total_responses_evaluated": len(self.response_evaluations),
                "optimization_runs": len(self.optimization_history),
                "current_best_score": self.best_score,
                "avg_recent_quality": np.mean([e["quality_score"] for e in self.response_evaluations[-100:]]) if len(self.response_evaluations) >= 100 else None
            },
            "parameter_history": self.optimization_history,
            "current_parameters": self.current_params,
            "best_parameters": self.best_params,
            "quality_metrics": {
                "scores": [e["quality_score"] for e in self.response_evaluations[-200:]],  # Last 200 scores
                "timestamps": [e["timestamp"].isoformat() for e in self.response_evaluations[-200:]]
            }
        }
