"""
FL Manager - Federated Learning Simulation Manager
Simulates federated learning for continuous improvement across Chennai zones
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
import hashlib

class FLManager:
    def __init__(self):
        """Initialize Federated Learning Manager"""
        self.federated_nodes = self._initialize_federated_nodes()
        self.global_model_params = self._initialize_global_model()
        self.interaction_history = []
        self.node_contributions = defaultdict(list)
        self.aggregation_frequency = 100  # Aggregate every 100 interactions
        self.privacy_threshold = 5  # Minimum interactions before sharing
        
        # Performance metrics
        self.metrics = {
            "total_interactions": 0,
            "successful_responses": 0,
            "user_satisfaction": [],
            "response_accuracy": [],
            "node_performance": defaultdict(dict)
        }
        
    def _initialize_federated_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize federated learning nodes representing Chennai zones"""
        nodes = {}
        
        # Create nodes for each Chennai zone
        zone_info = [
            {"zone": 1, "name": "Tiruvottiyur", "population": 250000, "node_type": "coastal_industrial"},
            {"zone": 2, "name": "Manali", "population": 280000, "node_type": "industrial"},
            {"zone": 3, "name": "Madhavaram", "population": 320000, "node_type": "transport_hub"},
            {"zone": 4, "name": "Tondiarpet", "population": 290000, "node_type": "port_commercial"},
            {"zone": 5, "name": "Royapuram", "population": 270000, "node_type": "heritage_commercial"},
            {"zone": 6, "name": "Thiru Vi Ka Nagar", "population": 350000, "node_type": "mixed_residential"},
            {"zone": 7, "name": "Ambattur", "population": 380000, "node_type": "it_residential"},
            {"zone": 8, "name": "Anna Nagar", "population": 300000, "node_type": "planned_commercial"},
            {"zone": 9, "name": "Teynampet", "population": 280000, "node_type": "commercial_hub"},
            {"zone": 10, "name": "Kodambakkam", "population": 290000, "node_type": "media_entertainment"},
            {"zone": 11, "name": "Valasaravakkam", "population": 340000, "node_type": "suburban_residential"},
            {"zone": 12, "name": "Alandur", "population": 250000, "node_type": "transport_airport"},
            {"zone": 13, "name": "Adyar", "population": 320000, "node_type": "upscale_coastal"},
            {"zone": 14, "name": "Perungudi", "population": 360000, "node_type": "it_corridor"},
            {"zone": 15, "name": "Sholinganallur", "population": 290000, "node_type": "tech_hub"}
        ]
        
        for zone in zone_info:
            node_id = f"zone_{zone['zone']}"
            nodes[node_id] = {
                "zone_number": zone["zone"],
                "zone_name": zone["name"],
                "population": zone["population"],
                "node_type": zone["node_type"],
                "local_model": self._initialize_local_model(),
                "interaction_count": 0,
                "last_update": datetime.now(),
                "specialization": self._get_zone_specialization(zone["node_type"]),
                "performance_score": 0.75,  # Initial score
                "privacy_budget": 1.0,  # Differential privacy budget
                "data_quality": np.random.uniform(0.7, 0.95),  # Simulated data quality
                "contribution_weight": 1.0
            }
        
        return nodes
    
    def _initialize_global_model(self) -> Dict[str, Any]:
        """Initialize global model parameters"""
        return {
            "query_classification": {
                "complaint": 0.25,
                "information": 0.30,
                "emergency": 0.15,
                "procedure": 0.20,
                "general": 0.10
            },
            "response_quality": {
                "accuracy_threshold": 0.85,
                "completeness_weight": 0.4,
                "relevance_weight": 0.35,
                "timeliness_weight": 0.25
            },
            "service_priorities": {
                "water_supply": 0.22,
                "waste_management": 0.20,
                "electricity": 0.18,
                "property_tax": 0.15,
                "roads": 0.12,
                "health": 0.08,
                "others": 0.05
            },
            "last_aggregation": datetime.now(),
            "aggregation_count": 0,
            "model_version": 1.0
        }
    
    def _get_zone_specialization(self, node_type: str) -> List[str]:
        """Get specialization areas for each zone type"""
        specializations = {
            "coastal_industrial": ["water_supply", "pollution", "industrial_waste"],
            "industrial": ["waste_management", "air_quality", "industrial_permits"],
            "transport_hub": ["traffic", "roads", "public_transport"],
            "port_commercial": ["trade_licenses", "commercial_permits", "logistics"],
            "heritage_commercial": ["heritage_conservation", "tourism", "cultural_events"],
            "mixed_residential": ["residential_services", "schools", "healthcare"],
            "it_residential": ["IT_infrastructure", "residential_planning", "tech_services"],
            "planned_commercial": ["urban_planning", "commercial_services", "infrastructure"],
            "commercial_hub": ["business_licenses", "commercial_disputes", "retail"],
            "media_entertainment": ["entertainment_permits", "media_regulations", "events"],
            "suburban_residential": ["residential_development", "suburban_services", "community"],
            "transport_airport": ["airport_services", "transport_connectivity", "logistics"],
            "upscale_coastal": ["coastal_management", "upscale_services", "environmental"],
            "it_corridor": ["IT_services", "tech_infrastructure", "innovation"],
            "tech_hub": ["technology", "startups", "innovation_services"]
        }
        return specializations.get(node_type, ["general_civic_services"])
    
    def _initialize_local_model(self) -> Dict[str, Any]:
        """Initialize local model parameters for a node"""
        return {
            "local_weights": np.random.uniform(0.5, 1.0, 10).tolist(),
            "local_bias": np.random.uniform(-0.1, 0.1, 5).tolist(),
            "specialization_scores": {},
            "local_accuracy": 0.75,
            "training_samples": 0,
            "last_training": datetime.now()
        }
    
    def update_from_interaction(self, query: str, response: str, user_area: str = None, 
                              user_feedback: Optional[float] = None) -> Dict[str, Any]:
        """Update federated learning from user interaction"""
        
        interaction_data = {
            "query": query,
            "response": response,
            "user_area": user_area,
            "timestamp": datetime.now(),
            "feedback": user_feedback or np.random.uniform(3.5, 4.8),  # Simulated feedback
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8]
        }
        
        # Add to interaction history
        self.interaction_history.append(interaction_data)
        self.metrics["total_interactions"] += 1
        
        # Determine relevant nodes based on user area and query content
        relevant_nodes = self._identify_relevant_nodes(query, user_area)
        
        # Update local nodes
        update_results = []
        for node_id in relevant_nodes:
            node_update = self._update_local_node(node_id, interaction_data)
            update_results.append(node_update)
            
            # Store contribution for aggregation
            self.node_contributions[node_id].append({
                "interaction": interaction_data,
                "update": node_update,
                "timestamp": datetime.now()
            })
        
        # Check if global aggregation is needed
        if self.metrics["total_interactions"] % self.aggregation_frequency == 0:
            aggregation_result = self._perform_global_aggregation()
            return {
                "local_updates": update_results,
                "global_aggregation": aggregation_result,
                "triggered_global_update": True
            }
        
        return {
            "local_updates": update_results,
            "triggered_global_update": False,
            "next_aggregation_in": self.aggregation_frequency - (self.metrics["total_interactions"] % self.aggregation_frequency)
        }
    
    def _identify_relevant_nodes(self, query: str, user_area: str = None) -> List[str]:
        """Identify which federated nodes are relevant for this query"""
        relevant_nodes = []
        query_lower = query.lower()
        
        # Geographic relevance
        if user_area:
            # Extract zone from user area if possible
            for node_id, node_data in self.federated_nodes.items():
                if (user_area.lower() in node_data["zone_name"].lower() or 
                    f"zone {node_data['zone_number']}" in user_area.lower()):
                    relevant_nodes.append(node_id)
                    break
        
        # Content-based relevance
        service_keywords = {
            "water": ["zone_1", "zone_13"],  # Coastal zones have more water issues
            "waste": ["zone_2", "zone_6", "zone_11"],  # High population zones
            "electricity": ["zone_7", "zone_14", "zone_15"],  # IT corridors
            "traffic": ["zone_3", "zone_8", "zone_12"],  # Transport hubs
            "commercial": ["zone_4", "zone_5", "zone_9"],  # Commercial areas
            "industrial": ["zone_1", "zone_2"],  # Industrial zones
            "tech": ["zone_7", "zone_14", "zone_15"],  # Tech hubs
            "residential": ["zone_6", "zone_8", "zone_11", "zone_13"]  # Major residential areas
        }
        
        for service, zones in service_keywords.items():
            if service in query_lower:
                relevant_nodes.extend([zone for zone in zones if zone not in relevant_nodes])
        
        # If no specific relevance found, include top performing nodes
        if not relevant_nodes:
            top_performers = sorted(
                self.federated_nodes.keys(), 
                key=lambda x: self.federated_nodes[x]["performance_score"], 
                reverse=True
            )[:3]
            relevant_nodes.extend(top_performers)
        
        return relevant_nodes[:5]  # Limit to 5 nodes for efficiency
    
    def _update_local_node(self, node_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a specific local node with interaction data"""
        
        if node_id not in self.federated_nodes:
            return {"error": f"Node {node_id} not found"}
        
        node = self.federated_nodes[node_id]
        
        # Update interaction count
        node["interaction_count"] += 1
        node["last_update"] = datetime.now()
        
        # Simulate local learning update
        feedback_score = interaction_data["feedback"]
        
        # Update local model weights based on feedback
        learning_rate = 0.01
        feedback_normalized = (feedback_score - 3.0) / 2.0  # Normalize to [-1, 1]
        
        # Update weights with differential privacy noise
        for i in range(len(node["local_model"]["local_weights"])):
            noise = np.random.normal(0, 0.02) if node["privacy_budget"] > 0 else 0
            node["local_model"]["local_weights"][i] += learning_rate * feedback_normalized + noise
            # Keep weights in reasonable range
            node["local_model"]["local_weights"][i] = np.clip(
                node["local_model"]["local_weights"][i], 0.1, 1.5
            )
        
        # Update local accuracy estimate
        accuracy_update = 0.05 if feedback_score > 4.0 else -0.02 if feedback_score < 3.5 else 0
        node["local_model"]["local_accuracy"] = np.clip(
            node["local_model"]["local_accuracy"] + accuracy_update, 0.5, 1.0
        )
        
        # Update performance score
        recent_interactions = min(node["interaction_count"], 50)
        performance_decay = 0.95  # Slight decay to encourage recent performance
        node["performance_score"] = (
            node["performance_score"] * performance_decay + 
            (feedback_score / 5.0) * (1 - performance_decay)
        )
        
        # Update specialization scores
        query_category = self._classify_query_category(interaction_data["query"])
        if query_category in node["specialization"]:
            current_score = node["local_model"]["specialization_scores"].get(query_category, 0.5)
            node["local_model"]["specialization_scores"][query_category] = (
                current_score * 0.9 + (feedback_score / 5.0) * 0.1
            )
        
        # Consume privacy budget
        node["privacy_budget"] = max(0, node["privacy_budget"] - 0.001)
        
        return {
            "node_id": node_id,
            "update_success": True,
            "new_performance_score": node["performance_score"],
            "new_accuracy": node["local_model"]["local_accuracy"],
            "privacy_budget_remaining": node["privacy_budget"],
            "interaction_count": node["interaction_count"]
        }
    
    def _classify_query_category(self, query: str) -> str:
        """Classify query into service categories"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["water", "supply", "pressure"]):
            return "water_supply"
        elif any(word in query_lower for word in ["garbage", "waste", "trash"]):
            return "waste_management"
        elif any(word in query_lower for word in ["power", "electricity", "outage"]):
            return "electricity"
        elif any(word in query_lower for word in ["tax", "property", "payment"]):
            return "property_tax"
        elif any(word in query_lower for word in ["road", "traffic", "pothole"]):
            return "roads"
        elif any(word in query_lower for word in ["health", "hospital", "medical"]):
            return "health"
        else:
            return "general"
    
    def _perform_global_aggregation(self) -> Dict[str, Any]:
        """Perform federated averaging to update global model"""
        
        if not self.node_contributions:
            return {"error": "No node contributions available for aggregation"}
        
        aggregation_result = {
            "aggregation_timestamp": datetime.now(),
            "participating_nodes": list(self.node_contributions.keys()),
            "total_contributions": sum(len(contributions) for contributions in self.node_contributions.values())
        }
        
        # Calculate weighted averages for global model parameters
        total_weight = 0
        weighted_params = {}
        
        # Aggregate query classification probabilities
        query_class_updates = defaultdict(list)
        service_priority_updates = defaultdict(list)
        accuracy_updates = []
        
        for node_id, contributions in self.node_contributions.items():
            node = self.federated_nodes[node_id]
            node_weight = self._calculate_node_weight(node)
            total_weight += node_weight
            
            # Collect updates for averaging
            for contrib in contributions[-10:]:  # Use last 10 contributions per node
                feedback = contrib["interaction"]["feedback"]
                category = self._classify_query_category(contrib["interaction"]["query"])
                
                query_class_updates[category].append(feedback * node_weight)
                accuracy_updates.append(node["local_model"]["local_accuracy"] * node_weight)
        
        # Update global model parameters
        if total_weight > 0:
            # Update query classification weights
            for category, updates in query_class_updates.items():
                if updates:
                    avg_performance = sum(updates) / len(updates) / total_weight
                    current_weight = self.global_model_params["query_classification"].get(category, 0.1)
                    self.global_model_params["query_classification"][category] = (
                        current_weight * 0.9 + avg_performance * 0.1
                    )
            
            # Normalize classification weights
            total_class_weight = sum(self.global_model_params["query_classification"].values())
            if total_class_weight > 0:
                for category in self.global_model_params["query_classification"]:
                    self.global_model_params["query_classification"][category] /= total_class_weight
            
            # Update accuracy threshold
            if accuracy_updates:
                global_accuracy = sum(accuracy_updates) / total_weight
                self.global_model_params["response_quality"]["accuracy_threshold"] = (
                    self.global_model_params["response_quality"]["accuracy_threshold"] * 0.8 + 
                    global_accuracy * 0.2
                )
        
        # Update global model metadata
        self.global_model_params["last_aggregation"] = datetime.now()
        self.global_model_params["aggregation_count"] += 1
        self.global_model_params["model_version"] += 0.1
        
        # Clear processed contributions (keep only recent ones)
        for node_id in list(self.node_contributions.keys()):
            self.node_contributions[node_id] = self.node_contributions[node_id][-5:]
        
        # Update aggregation result
        aggregation_result.update({
            "new_model_version": self.global_model_params["model_version"],
            "global_accuracy_threshold": self.global_model_params["response_quality"]["accuracy_threshold"],
            "participating_node_count": len(self.node_contributions),
            "aggregation_success": True
        })
        
        return aggregation_result
    
    def _calculate_node_weight(self, node: Dict[str, Any]) -> float:
        """Calculate weight for a node in federated aggregation"""
        
        # Factors for weighting:
        # 1. Data quality
        # 2. Recent performance
        # 3. Interaction count (but not too dominant)
        # 4. Privacy budget remaining
        
        data_quality_weight = node["data_quality"]
        performance_weight = node["performance_score"]
        
        # Interaction count weight (logarithmic to prevent dominance)
        interaction_weight = min(1.0, np.log(max(1, node["interaction_count"])) / 10)
        
        # Privacy weight (nodes with more privacy budget contribute more)
        privacy_weight = max(0.1, node["privacy_budget"])
        
        # Combined weight
        combined_weight = (
            data_quality_weight * 0.3 +
            performance_weight * 0.4 +
            interaction_weight * 0.2 +
            privacy_weight * 0.1
        )
        
        return max(0.1, combined_weight)  # Minimum weight threshold
    
    def get_personalized_response_params(self, query: str, user_area: str = None) -> Dict[str, Any]:
        """Get personalized response parameters based on federated learning"""
        
        # Identify relevant nodes
        relevant_nodes = self._identify_relevant_nodes(query, user_area)
        
        # Get specialized parameters from relevant nodes
        specialized_params = {
            "confidence_boost": 0.0,
            "specialization_match": False,
            "local_expertise": [],
            "recommended_contact": None,
            "area_specific_info": None
        }
        
        query_category = self._classify_query_category(query)
        
        for node_id in relevant_nodes:
            node = self.federated_nodes[node_id]
            
            # Check if node specializes in this query category
            if query_category in node["specialization"]:
                specialized_params["specialization_match"] = True
                specialized_params["confidence_boost"] += 0.1
                specialized_params["local_expertise"].append(node["zone_name"])
                
                # Get specialization score if available
                spec_score = node["local_model"]["specialization_scores"].get(query_category, 0.5)
                specialized_params["confidence_boost"] += spec_score * 0.1
            
            # Add area-specific information
            if user_area and user_area.lower() in node["zone_name"].lower():
                specialized_params["area_specific_info"] = {
                    "zone": node["zone_name"],
                    "population": node["population"],
                    "node_type": node["node_type"],
                    "performance": node["performance_score"]
                }
        
        # Cap confidence boost
        specialized_params["confidence_boost"] = min(0.3, specialized_params["confidence_boost"])
        
        return specialized_params
    
    def get_fl_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning statistics"""
        
        # Calculate global statistics
        total_interactions = sum(node["interaction_count"] for node in self.federated_nodes.values())
        avg_performance = np.mean([node["performance_score"] for node in self.federated_nodes.values()])
        avg_accuracy = np.mean([node["local_model"]["local_accuracy"] for node in self.federated_nodes.values()])
        
        # Node performance ranking
        node_ranking = sorted(
            [(node_id, node["performance_score"], node["interaction_count"]) 
             for node_id, node in self.federated_nodes.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Privacy budget status
        privacy_status = {
            node_id: {
                "remaining_budget": node["privacy_budget"],
                "status": "good" if node["privacy_budget"] > 0.5 else "low" if node["privacy_budget"] > 0.1 else "critical"
            }
            for node_id, node in self.federated_nodes.items()
        }
        
        return {
            "global_model": {
                "version": self.global_model_params["model_version"],
                "last_aggregation": self.global_model_params["last_aggregation"].isoformat(),
                "aggregation_count": self.global_model_params["aggregation_count"],
                "accuracy_threshold": self.global_model_params["response_quality"]["accuracy_threshold"]
            },
            "federation_stats": {
                "total_nodes": len(self.federated_nodes),
                "total_interactions": total_interactions,
                "average_performance": round(avg_performance, 3),
                "average_accuracy": round(avg_accuracy, 3),
                "interactions_until_next_aggregation": self.aggregation_frequency - (self.metrics["total_interactions"] % self.aggregation_frequency)
            },
            "node_performance": {
                "top_performers": [{"node": rank[0], "score": round(rank[1], 3), "interactions": rank[2]} 
                                 for rank in node_ranking[:5]],
                "bottom_performers": [{"node": rank[0], "score": round(rank[1], 3), "interactions": rank[2]} 
                                    for rank in node_ranking[-3:]]
            },
            "privacy_status": privacy_status,
            "specialization_coverage": self._get_specialization_coverage(),
            "recent_activity": self._get_recent_activity_summary()
        }
    
    def _get_specialization_coverage(self) -> Dict[str, List[str]]:
        """Get which nodes specialize in which areas"""
        specialization_map = defaultdict(list)
        
        for node_id, node in self.federated_nodes.items():
            for specialization in node["specialization"]:
                specialization_map[specialization].append(node["zone_name"])
        
        return dict(specialization_map)
    
    def _get_recent_activity_summary(self) -> Dict[str, Any]:
        """Get summary of recent federated learning activity"""
        recent_cutoff = datetime.now() - timedelta(hours=24)
        
        recent_interactions = [
            interaction for interaction in self.interaction_history
            if interaction["timestamp"] > recent_cutoff
        ]
        
        if not recent_interactions:
            return {"message": "No recent activity in the last 24 hours"}
        
        # Analyze recent activity
        avg_feedback = np.mean([interaction["feedback"] for interaction in recent_interactions])
        query_categories = defaultdict(int)
        
        for interaction in recent_interactions:
            category = self._classify_query_category(interaction["query"])
            query_categories[category] += 1
        
        return {
            "last_24_hours": {
                "total_interactions": len(recent_interactions),
                "average_feedback": round(avg_feedback, 2),
                "most_common_categories": dict(sorted(query_categories.items(), key=lambda x: x[1], reverse=True)[:5])
            }
        }
    
    def simulate_federated_training_round(self) -> Dict[str, Any]:
        """Simulate a complete federated training round for testing"""
        
        # Generate simulated interactions for each node
        simulation_results = []
        
        for node_id, node in self.federated_nodes.items():
            # Generate 5-15 simulated interactions per node
            num_interactions = np.random.randint(5, 16)
            
            node_results = {
                "node_id": node_id,
                "zone_name": node["zone_name"],
                "simulated_interactions": num_interactions,
                "performance_updates": []
            }
            
            for _ in range(num_interactions):
                # Generate realistic queries based on zone specialization
                query = self._generate_realistic_query(node["specialization"])
                feedback = np.random.uniform(3.0, 5.0)
                
                # Simulate interaction update
                interaction_result = self.update_from_interaction(
                    query=query,
                    response="Simulated response",
                    user_area=node["zone_name"],
                    user_feedback=feedback
                )
                
                node_results["performance_updates"].append({
                    "query_category": self._classify_query_category(query),
                    "feedback": round(feedback, 2)
                })
            
            simulation_results.append(node_results)
        
        # Force global aggregation
        aggregation_result = self._perform_global_aggregation()
        
        return {
            "simulation_complete": True,
            "node_results": simulation_results,
            "global_aggregation": aggregation_result,
            "total_simulated_interactions": sum(result["simulated_interactions"] for result in simulation_results)
        }
    
    def _generate_realistic_query(self, specializations: List[str]) -> str:
        """Generate realistic queries based on zone specializations"""
        
        query_templates = {
            "water_supply": [
                "Water pressure is low in my area",
                "How to report water supply interruption",
                "Water quality issues in the locality"
            ],
            "waste_management": [
                "Garbage collection delayed for 3 days",
                "How to complain about waste pickup",
                "Waste segregation guidelines"
            ],
            "electricity": [
                "Power outage in the neighborhood",
                "Streetlight not working on my street",
                "How to apply for new EB connection"
            ],
            "industrial_waste": [
                "Industrial waste disposal regulations",
                "Air pollution from nearby factory",
                "How to report industrial violations"
            ],
            "traffic": [
                "Traffic signal not working properly",
                "Road construction causing traffic jam",
                "How to request speed breakers"
            ]
        }
        
        # Pick a specialization and generate relevant query
        if specializations:
            chosen_spec = np.random.choice([spec for spec in specializations if spec in query_templates])
            if chosen_spec in query_templates:
                return np.random.choice(query_templates[chosen_spec])
        
        # Fallback to general queries
        general_queries = [
            "How to pay property tax online",
            "Birth certificate application process",
            "Contact number for corporation office"
        ]
        return np.random.choice(general_queries)
