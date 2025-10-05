"""
KAG Engine - Knowledge Graph Augmented Generation
NetworkX-based knowledge graph for Chennai civic services
"""

import networkx as nx
import json
from datetime import datetime
from typing import Dict, Any, List, Set, Optional, Tuple
import pandas as pd

class KAGEngine:
    def __init__(self):
        """Initialize Knowledge Graph for Chennai civic ecosystem"""
        self.civic_graph = nx.MultiDiGraph()
        self.entity_embeddings = {}
        self.relationship_weights = {}
        self._build_comprehensive_knowledge_graph()
        
    def _build_comprehensive_knowledge_graph(self):
        """Build comprehensive knowledge graph of Chennai civic services"""
        
        # Add government entities
        self._add_government_entities()
        
        # Add geographical entities
        self._add_geographical_entities()
        
        # Add service entities
        self._add_service_entities()
        
        # Add personnel entities
        self._add_personnel_entities()
        
        # Add relationships
        self._add_entity_relationships()
        
        # Calculate centrality measures
        self._calculate_graph_metrics()
    
    def _add_government_entities(self):
        """Add government departments and bodies"""
        departments = [
            {
                "id": "gcc", "name": "Greater Chennai Corporation", "type": "municipal_body",
                "head": "Commissioner", "budget": "₹8,000 crores", "employees": 75000,
                "zones": 15, "wards": 200, "services": 45
            },
            {
                "id": "cmwssb", "name": "Chennai Metro Water Supply & Sewerage Board", 
                "type": "water_authority", "head": "Managing Director", "coverage": "426 sq km",
                "daily_supply": "830 MLD", "connections": 650000
            },
            {
                "id": "tneb", "name": "Tamil Nadu Electricity Board", "type": "power_utility",
                "head": "Chief Engineer Chennai", "divisions": 8, "substations": 245,
                "consumers": 1200000
            },
            {
                "id": "health_dept", "name": "Public Health Department", "type": "health_authority",
                "head": "City Health Officer", "hospitals": 23, "health_centers": 147,
                "beds": 8500
            },
            {
                "id": "revenue_dept", "name": "Revenue Department", "type": "revenue_authority",
                "head": "Commissioner Revenue", "registrar_offices": 200, 
                "property_assessments": 1200000
            },
            {
                "id": "police_dept", "name": "Chennai City Police", "type": "law_enforcement",
                "head": "Police Commissioner", "stations": 121, "divisions": 4,
                "personnel": 25000
            },
            {
                "id": "fire_dept", "name": "Tamil Nadu Fire & Rescue", "type": "emergency_services",
                "head": "Director Fire Services", "stations": 32, "personnel": 2500,
                "response_time": "8-12 minutes"
            }
        ]
        
        for dept in departments:
            dept_id = dept.pop("id")
            self.civic_graph.add_node(dept_id, **dept)
    
    def _add_geographical_entities(self):
        """Add zones, wards, and localities"""
        
        # Add zones with detailed information
        zones_data = [
            {"id": "zone_1", "name": "Tiruvottiyur", "type": "zone", "wards": "1-14", 
             "areas": ["Tiruvottiyur", "Kathivakkam", "Ennore"], "population": 250000,
             "industrial": True, "coastal": True, "port_area": True},
            {"id": "zone_2", "name": "Manali", "type": "zone", "wards": "15-28",
             "areas": ["Manali", "Madhavaram", "Puzhal"], "population": 280000,
             "industrial": True, "petrochemical": True},
            {"id": "zone_3", "name": "Madhavaram", "type": "zone", "wards": "29-42",
             "areas": ["Madhavaram", "Perambur", "Korattur"], "population": 320000,
             "transport_hub": True, "railway": True},
            {"id": "zone_4", "name": "Tondiarpet", "type": "zone", "wards": "43-56",
             "areas": ["Tondiarpet", "Washermenpet", "Royapuram"], "population": 290000,
             "port_area": True, "commercial": True},
            {"id": "zone_5", "name": "Royapuram", "type": "zone", "wards": "57-70",
             "areas": ["Royapuram", "Sowcarpet", "Georgetown"], "population": 270000,
             "heritage": True, "commercial": True, "wholesale": True},
            {"id": "zone_6", "name": "Thiru Vi Ka Nagar", "type": "zone", "wards": "71-84",
             "areas": ["Ambattur", "Avadi", "Padi"], "population": 350000,
             "industrial": True, "residential": True},
            {"id": "zone_7", "name": "Ambattur", "type": "zone", "wards": "85-98",
             "areas": ["Ambattur", "Mogappair", "Anna Nagar West"], "population": 380000,
             "it_corridor": True, "residential": True},
            {"id": "zone_8", "name": "Anna Nagar", "type": "zone", "wards": "99-112",
             "areas": ["Anna Nagar", "Kilpauk", "Purasawalkam"], "population": 300000,
             "planned_city": True, "commercial": True},
            {"id": "zone_9", "name": "Teynampet", "type": "zone", "wards": "113-126",
             "areas": ["Teynampet", "T.Nagar", "Nandanam"], "population": 280000,
             "commercial_hub": True, "shopping": True},
            {"id": "zone_10", "name": "Kodambakkam", "type": "zone", "wards": "127-140",
             "areas": ["Kodambakkam", "Vadapalani", "Saidapet"], "population": 290000,
             "film_city": True, "media": True},
            {"id": "zone_11", "name": "Valasaravakkam", "type": "zone", "wards": "141-154",
             "areas": ["Valasaravakkam", "Porur", "Ramapuram"], "population": 340000,
             "outer_ring_road": True, "residential": True},
            {"id": "zone_12", "name": "Alandur", "type": "zone", "wards": "155-168",
             "areas": ["Alandur", "St. Thomas Mount", "Guindy"], "population": 250000,
             "airport_zone": True, "transport_hub": True},
            {"id": "zone_13", "name": "Adyar", "type": "zone", "wards": "169-182",
             "areas": ["Adyar", "Besant Nagar", "Thiruvanmiyur"], "population": 320000,
             "coastal": True, "upscale": True, "educational": True},
            {"id": "zone_14", "name": "Perungudi", "type": "zone", "wards": "183-196",
             "areas": ["Perungudi", "Velachery", "Pallikaranai"], "population": 360000,
             "it_corridor": True, "marshland": True},
            {"id": "zone_15", "name": "Sholinganallur", "type": "zone", "wards": "197-200",
             "areas": ["Sholinganallur", "Navalur", "Siruseri"], "population": 290000,
             "omr_corridor": True, "it_hub": True}
        ]
        
        for zone in zones_data:
            zone_id = zone.pop("id")
            self.civic_graph.add_node(zone_id, **zone)
        
        # Add sample wards (representative set)
        sample_wards = [
            {"id": "ward_1", "number": 1, "type": "ward", "zone": "zone_1", 
             "population": 18000, "households": 4500, "area_sq_km": 2.1},
            {"id": "ward_50", "number": 50, "type": "ward", "zone": "zone_4",
             "population": 21000, "households": 5250, "area_sq_km": 1.8},
            {"id": "ward_100", "number": 100, "type": "ward", "zone": "zone_8",
             "population": 19500, "households": 4875, "area_sq_km": 2.3},
            {"id": "ward_150", "number": 150, "type": "ward", "zone": "zone_11",
             "population": 22000, "households": 5500, "area_sq_km": 2.7},
            {"id": "ward_200", "number": 200, "type": "ward", "zone": "zone_15",
             "population": 25000, "households": 6250, "area_sq_km": 3.2}
        ]
        
        for ward in sample_wards:
            ward_id = ward.pop("id")
            self.civic_graph.add_node(ward_id, **ward)
    
    def _add_service_entities(self):
        """Add civic services and their attributes"""
        services = [
            {
                "id": "water_supply", "name": "Water Supply", "type": "civic_service",
                "department": "cmwssb", "sla": "24 hours", "digital": True,
                "complaints_per_month": 2500, "satisfaction": 4.1
            },
            {
                "id": "waste_management", "name": "Solid Waste Management", "type": "civic_service",
                "department": "gcc", "sla": "24 hours", "digital": True,
                "complaints_per_month": 3200, "satisfaction": 3.8
            },
            {
                "id": "electricity_supply", "name": "Electricity Supply", "type": "civic_service",
                "department": "tneb", "sla": "4 hours", "digital": True,
                "complaints_per_month": 2800, "satisfaction": 4.0
            },
            {
                "id": "property_tax", "name": "Property Tax", "type": "civic_service",
                "department": "revenue_dept", "sla": "15 days", "digital": True,
                "queries_per_month": 15000, "satisfaction": 4.3
            },
            {
                "id": "birth_certificate", "name": "Birth Certificate", "type": "civic_service",
                "department": "revenue_dept", "sla": "7 days", "digital": True,
                "applications_per_month": 8500, "satisfaction": 4.4
            },
            {
                "id": "road_maintenance", "name": "Road Maintenance", "type": "civic_service",
                "department": "gcc", "sla": "7 days", "digital": True,
                "complaints_per_month": 2400, "satisfaction": 3.5
            },
            {
                "id": "health_services", "name": "Public Health Services", "type": "civic_service",
                "department": "health_dept", "sla": "immediate", "digital": False,
                "patients_per_month": 45000, "satisfaction": 3.9
            }
        ]
        
        for service in services:
            service_id = service.pop("id")
            self.civic_graph.add_node(service_id, **service)
    
    def _add_personnel_entities(self):
        """Add key personnel and their roles"""
        personnel = [
            {
                "id": "commissioner", "name": "Corporation Commissioner", "type": "personnel",
                "level": "city", "department": "gcc", "authority": "executive"
            },
            {
                "id": "md_metro_water", "name": "Managing Director CMWSSB", "type": "personnel",
                "level": "city", "department": "cmwssb", "authority": "executive"
            },
            {
                "id": "police_commissioner", "name": "Police Commissioner", "type": "personnel",
                "level": "city", "department": "police_dept", "authority": "executive"
            }
        ]
        
        # Add zone officers
        for i in range(1, 16):
            personnel.append({
                "id": f"zone_officer_{i}", "name": f"Zone {i} Officer", "type": "personnel",
                "level": "zone", "zone": f"zone_{i}", "authority": "administrative"
            })
        
        for person in personnel:
            person_id = person.pop("id")
            self.civic_graph.add_node(person_id, **person)
    
    def _add_entity_relationships(self):
        """Add relationships between entities"""
        
        # Department to zone relationships
        for i in range(1, 16):
            self.civic_graph.add_edge("gcc", f"zone_{i}", relationship="administers", weight=1.0)
            self.civic_graph.add_edge(f"zone_officer_{i}", f"zone_{i}", relationship="manages", weight=1.0)
        
        # Service to department relationships
        service_dept_mapping = {
            "water_supply": "cmwssb",
            "waste_management": "gcc", 
            "electricity_supply": "tneb",
            "property_tax": "revenue_dept",
            "birth_certificate": "revenue_dept",
            "road_maintenance": "gcc",
            "health_services": "health_dept"
        }
        
        for service, dept in service_dept_mapping.items():
            self.civic_graph.add_edge(dept, service, relationship="provides", weight=1.0)
        
        # Ward to zone relationships
        ward_zone_mapping = {
            "ward_1": "zone_1", "ward_50": "zone_4", "ward_100": "zone_8",
            "ward_150": "zone_11", "ward_200": "zone_15"
        }
        
        for ward, zone in ward_zone_mapping.items():
            self.civic_graph.add_edge(zone, ward, relationship="contains", weight=1.0)
        
        # Personnel hierarchy
        self.civic_graph.add_edge("commissioner", "gcc", relationship="heads", weight=1.0)
        self.civic_graph.add_edge("md_metro_water", "cmwssb", relationship="heads", weight=1.0)
        self.civic_graph.add_edge("police_commissioner", "police_dept", relationship="heads", weight=1.0)
    
    def _calculate_graph_metrics(self):
        """Calculate graph centrality and importance metrics"""
        
        # Calculate various centrality measures
        self.pagerank_scores = nx.pagerank(self.civic_graph)
        self.betweenness_centrality = nx.betweenness_centrality(self.civic_graph)
        self.degree_centrality = nx.degree_centrality(self.civic_graph)
        
        # Add centrality scores as node attributes
        for node in self.civic_graph.nodes():
            self.civic_graph.nodes[node]['pagerank'] = self.pagerank_scores.get(node, 0)
            self.civic_graph.nodes[node]['betweenness'] = self.betweenness_centrality.get(node, 0)
            self.civic_graph.nodes[node]['degree_centrality'] = self.degree_centrality.get(node, 0)
    
    def knowledge_reasoning(self, query: str, user_area: str = None) -> Dict[str, Any]:
        """Perform knowledge graph reasoning for civic queries"""
        
        # Step 1: Extract entities from query
        extracted_entities = self._extract_query_entities(query, user_area)
        
        # Step 2: Find relevant subgraph
        relevant_subgraph = self._find_relevant_subgraph(extracted_entities)
        
        # Step 3: Apply graph reasoning
        reasoning_result = self._apply_graph_reasoning(query, extracted_entities, relevant_subgraph)
        
        # Step 4: Generate structured response
        if reasoning_result:
            response = self._generate_knowledge_response(query, reasoning_result)
            success = True
        else:
            response = self._generate_fallback_knowledge_response(query, extracted_entities)
            success = False
        
        return {
            "success": success,
            "response": response,
            "entities": extracted_entities,
            "subgraph_size": len(relevant_subgraph.nodes()),
            "reasoning_paths": reasoning_result.get("paths", []),
            "confidence": reasoning_result.get("confidence", 0.5),
            "source": "KAG - Knowledge Graph Reasoning"
        }
    
    def _extract_query_entities(self, query: str, user_area: str = None) -> List[str]:
        """Extract relevant entities from query"""
        query_lower = query.lower()
        extracted_entities = []
        
        # Extract service entities
        service_keywords = {
            "water_supply": ["water", "supply", "pressure", "quality", "metro water"],
            "waste_management": ["garbage", "waste", "trash", "collection", "dustbin"],
            "electricity_supply": ["power", "electricity", "outage", "streetlight", "eb"],
            "property_tax": ["property", "tax", "assessment", "payment"],
            "birth_certificate": ["birth", "certificate", "birth certificate"],
            "road_maintenance": ["road", "pothole", "repair", "maintenance"],
            "health_services": ["hospital", "health", "clinic", "medical"]
        }
        
        for service, keywords in service_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_entities.append(service)
        
        # Extract geographical entities
        for node in self.civic_graph.nodes():
            node_data = self.civic_graph.nodes[node]
            if node_data.get("type") in ["zone", "ward"]:
                # Check if zone/ward name or areas mentioned
                if node_data.get("name") and node_data["name"].lower() in query_lower:
                    extracted_entities.append(node)
                elif node_data.get("areas"):
                    for area in node_data["areas"]:
                        if area.lower() in query_lower:
                            extracted_entities.append(node)
                            break
        
        # Extract department entities
        dept_keywords = {
            "gcc": ["corporation", "gcc", "municipal"],
            "cmwssb": ["metro water", "water board", "cmwssb"],
            "tneb": ["electricity board", "tneb", "eb"],
            "revenue_dept": ["revenue", "certificate", "tax"],
            "health_dept": ["health", "hospital", "medical"],
            "police_dept": ["police"],
            "fire_dept": ["fire", "rescue"]
        }
        
        for dept, keywords in dept_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_entities.append(dept)
        
        return list(set(extracted_entities))  # Remove duplicates
    
    def _find_relevant_subgraph(self, entities: List[str]) -> nx.Graph:
        """Find relevant subgraph based on extracted entities"""
        if not entities:
            return nx.Graph()
        
        # Start with extracted entities
        relevant_nodes = set(entities)
        
        # Add immediate neighbors
        for entity in entities:
            if entity in self.civic_graph:
                neighbors = list(self.civic_graph.neighbors(entity))
                relevant_nodes.update(neighbors[:5])  # Limit to top 5 neighbors
        
        # Create subgraph
        subgraph = self.civic_graph.subgraph(relevant_nodes)
        return subgraph
    
    def _apply_graph_reasoning(self, query: str, entities: List[str], subgraph: nx.Graph) -> Dict[str, Any]:
        """Apply graph-based reasoning"""
        
        if not entities or subgraph.number_of_nodes() == 0:
            return {}
        
        reasoning_result = {
            "primary_entities": entities[:3],  # Top 3 most relevant
            "paths": [],
            "related_services": [],
            "departments": [],
            "contacts": [],
            "procedures": [],
            "confidence": 0.0
        }
        
        # Find paths between entities
        entity_pairs = [(entities[i], entities[j]) for i in range(len(entities)) 
                       for j in range(i+1, len(entities)) if i != j]
        
        for source, target in entity_pairs[:5]:  # Limit path finding
            if source in subgraph and target in subgraph:
                try:
                    path = nx.shortest_path(subgraph, source, target)
                    reasoning_result["paths"].append({
                        "source": source,
                        "target": target,
                        "path": path,
                        "length": len(path) - 1
                    })
                except nx.NetworkXNoPath:
                    continue
        
        # Extract departments and services
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            node_type = node_data.get("type", "")
            
            if node_type == "municipal_body" or node_type in ["water_authority", "power_utility", "health_authority"]:
                reasoning_result["departments"].append({
                    "id": node,
                    "name": node_data.get("name", ""),
                    "type": node_type,
                    "head": node_data.get("head", ""),
                    "importance": node_data.get("pagerank", 0)
                })
            
            elif node_type == "civic_service":
                reasoning_result["related_services"].append({
                    "id": node,
                    "name": node_data.get("name", ""),
                    "department": node_data.get("department", ""),
                    "sla": node_data.get("sla", ""),
                    "satisfaction": node_data.get("satisfaction", 0)
                })
        
        # Extract contact information
        contact_mapping = {
            "gcc": "1913", "cmwssb": "044-45671200", "tneb": "94987-94987",
            "revenue_dept": "044-25619515", "health_dept": "044-25619671",
            "police_dept": "100", "fire_dept": "101"
        }
        
        for dept in reasoning_result["departments"]:
            if dept["id"] in contact_mapping:
                reasoning_result["contacts"].append({
                    "department": dept["name"],
                    "contact": contact_mapping[dept["id"]]
                })
        
        # Calculate confidence based on graph connectivity and entity relevance
        base_confidence = min(0.8, len(entities) * 0.2)
        path_bonus = min(0.15, len(reasoning_result["paths"]) * 0.05)
        reasoning_result["confidence"] = base_confidence + path_bonus
        
        return reasoning_result
    
    def _generate_knowledge_response(self, query: str, reasoning_result: Dict) -> str:
        """Generate response from knowledge graph reasoning"""
        
        response_parts = []
        
        # Add department information
        if reasoning_result["departments"]:
            primary_dept = reasoning_result["departments"][0]
            response_parts.append(f"🏢 **Responsible Department**: {primary_dept['name']} (Head: {primary_dept['head']})")
        
        # Add service information
        if reasoning_result["related_services"]:
            primary_service = reasoning_result["related_services"][0]
            response_parts.append(f"⚙️ **Service**: {primary_service['name']} (SLA: {primary_service['sla']})")
            if primary_service.get("satisfaction"):
                response_parts.append(f"📊 **Satisfaction Rating**: {primary_service['satisfaction']}/5.0")
        
        # Add contact information
        if reasoning_result["contacts"]:
            contact_info = reasoning_result["contacts"][0]
            response_parts.append(f"📞 **Contact**: {contact_info['department']} - {contact_info['contact']}")
        
        # Add procedural guidance based on query type
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "process", "procedure", "steps"]):
            if "property tax" in query_lower:
                response_parts.append("📋 **Process**: Online payment available at chennaicorporation.gov.in/propertytax")
            elif "birth certificate" in query_lower:
                response_parts.append("📋 **Process**: Apply online at tnreginet.gov.in or visit registrar office")
            elif "water connection" in query_lower:
                response_parts.append("📋 **Process**: Submit application → Site inspection → Payment → Installation")
        
        # Add relationship insights from graph paths
        if reasoning_result["paths"]:
            path_insight = reasoning_result["paths"][0]
            if len(path_insight["path"]) > 2:
                middle_entity = path_insight["path"][1]
                if middle_entity in self.civic_graph:
                    entity_data = self.civic_graph.nodes[middle_entity]
                    if entity_data.get("name"):
                        response_parts.append(f"🔗 **Related**: This also involves {entity_data['name']}")
        
        if not response_parts:
            return "I found relevant information in the knowledge graph but need more specific details to provide a complete answer."
        
        return "\n\n".join(response_parts)
    
    def _generate_fallback_knowledge_response(self, query: str, entities: List[str]) -> str:
        """Generate fallback response when reasoning fails"""
        
        if entities:
            # Try to provide information about recognized entities
            entity_info = []
            for entity in entities[:2]:  # Limit to 2 entities
                if entity in self.civic_graph:
                    node_data = self.civic_graph.nodes[entity]
                    if node_data.get("name"):
                        entity_info.append(f"• {node_data['name']}: {node_data.get('type', 'civic entity')}")
            
            if entity_info:
                return f"I recognize these entities from your query:\n" + "\n".join(entity_info) + f"\n\nFor specific assistance, please contact Chennai Corporation at 1913."
        
        return "I couldn't find specific matches in the knowledge graph. For general civic assistance, contact Chennai Corporation at 1913."
    
    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        if entity_id in self.civic_graph:
            node_data = dict(self.civic_graph.nodes[entity_id])
            
            # Add relationship information
            relationships = []
            for neighbor in self.civic_graph.neighbors(entity_id):
                edge_data = self.civic_graph[entity_id][neighbor]
                relationships.append({
                    "target": neighbor,
                    "relationship": edge_data.get("relationship", "connected"),
                    "weight": edge_data.get("weight", 0.5)
                })
            
            node_data["relationships"] = relationships
            return node_data
        
        return {}
    
    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two entities"""
        try:
            return nx.shortest_path(self.civic_graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_similar_entities(self, entity_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar entities based on graph structure"""
        if entity_id not in self.civic_graph:
            return []
        
        # Get entities with similar attributes
        target_attributes = self.civic_graph.nodes[entity_id]
        similar_entities = []
        
        for node in self.civic_graph.nodes():
            if node == entity_id:
                continue
            
            node_attributes = self.civic_graph.nodes[node]
            similarity_score = self._calculate_attribute_similarity(target_attributes, node_attributes)
            
            if similarity_score > 0.3:  # Similarity threshold
                similar_entities.append({
                    "id": node,
                    "attributes": node_attributes,
                    "similarity": similarity_score
                })
        
        # Sort by similarity and return top results
        similar_entities.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_entities[:limit]
    
    def _calculate_attribute_similarity(self, attr1: Dict, attr2: Dict) -> float:
        """Calculate similarity between two entity attribute sets"""
        
        # Compare type similarity (highest weight)
        type_similarity = 1.0 if attr1.get("type") == attr2.get("type") else 0.0
        
        # Compare common attributes
        common_attributes = set(attr1.keys()) & set(attr2.keys())
        attribute_matches = sum(1 for attr in common_attributes 
                              if attr1[attr] == attr2[attr])
        
        attribute_similarity = attribute_matches / max(len(common_attributes), 1)
        
        # Weighted combination
        return 0.6 * type_similarity + 0.4 * attribute_similarity
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return {
            "total_nodes": self.civic_graph.number_of_nodes(),
            "total_edges": self.civic_graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "average_degree": sum(dict(self.civic_graph.degree()).values()) / self.civic_graph.number_of_nodes(),
            "density": nx.density(self.civic_graph),
            "most_central_nodes": self._get_most_central_nodes(),
            "connected_components": nx.number_weakly_connected_components(self.civic_graph)
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = {}
        for node in self.civic_graph.nodes():
            node_type = self.civic_graph.nodes[node].get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _get_most_central_nodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most central nodes by PageRank score"""
        most_central = []
        for node, score in sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            node_data = self.civic_graph.nodes[node]
            most_central.append({
                "id": node,
                "name": node_data.get("name", node),
                "type": node_data.get("type", "unknown"),
                "pagerank_score": score
            })
        return most_central"""
KAG Engine - Knowledge Graph Augmented Generation
NetworkX-based knowledge graph for Chennai civic services
"""

import networkx as nx
import json
from datetime import datetime
from typing import Dict, Any, List, Set, Optional, Tuple
import pandas as pd

class KAGEngine:
    def __init__(self):
        """Initialize Knowledge Graph for Chennai civic ecosystem"""
        self.civic_graph = nx.MultiDiGraph()
        self.entity_embeddings = {}
        self.relationship_weights = {}
        self._build_comprehensive_knowledge_graph()
        
    def _build_comprehensive_knowledge_graph(self):
        """Build comprehensive knowledge graph of Chennai civic services"""
        
        # Add government entities
        self._add_government_entities()
        
        # Add geographical entities
        self._add_geographical_entities()
        
        # Add service entities
        self._add_service_entities()
        
        # Add personnel entities
        self._add_personnel_entities()
        
        # Add relationships
        self._add_entity_relationships()
        
        # Calculate centrality measures
        self._calculate_graph_metrics()
    
    def _add_government_entities(self):
        """Add government departments and bodies"""
        departments = [
            {
                "id": "gcc", "name": "Greater Chennai Corporation", "type": "municipal_body",
                "head": "Commissioner", "budget": "₹8,000 crores", "employees": 75000,
                "zones": 15, "wards": 200, "services": 45
            },
            {
                "id": "cmwssb", "name": "Chennai Metro Water Supply & Sewerage Board", 
                "type": "water_authority", "head": "Managing Director", "coverage": "426 sq km",
                "daily_supply": "830 MLD", "connections": 650000
            },
            {
                "id": "tneb", "name": "Tamil Nadu Electricity Board", "type": "power_utility",
                "head": "Chief Engineer Chennai", "divisions": 8, "substations": 245,
                "consumers": 1200000
            },
            {
                "id": "health_dept", "name": "Public Health Department", "type": "health_authority",
                "head": "City Health Officer", "hospitals": 23, "health_centers": 147,
                "beds": 8500
            },
            {
                "id": "revenue_dept", "name": "Revenue Department", "type": "revenue_authority",
                "head": "Commissioner Revenue", "registrar_offices": 200, 
                "property_assessments": 1200000
            },
            {
                "id": "police_dept", "name": "Chennai City Police", "type": "law_enforcement",
                "head": "Police Commissioner", "stations": 121, "divisions": 4,
                "personnel": 25000
            },
            {
                "id": "fire_dept", "name": "Tamil Nadu Fire & Rescue", "type": "emergency_services",
                "head": "Director Fire Services", "stations": 32, "personnel": 2500,
                "response_time": "8-12 minutes"
            }
        ]
        
        for dept in departments:
            dept_id = dept.pop("id")
            self.civic_graph.add_node(dept_id, **dept)
    
    def _add_geographical_entities(self):
        """Add zones, wards, and localities"""
        
        # Add zones with detailed information
        zones_data = [
            {"id": "zone_1", "name": "Tiruvottiyur", "type": "zone", "wards": "1-14", 
             "areas": ["Tiruvottiyur", "Kathivakkam", "Ennore"], "population": 250000,
             "industrial": True, "coastal": True, "port_area": True},
            {"id": "zone_2", "name": "Manali", "type": "zone", "wards": "15-28",
             "areas": ["Manali", "Madhavaram", "Puzhal"], "population": 280000,
             "industrial": True, "petrochemical": True},
            {"id": "zone_3", "name": "Madhavaram", "type": "zone", "wards": "29-42",
             "areas": ["Madhavaram", "Perambur", "Korattur"], "population": 320000,
             "transport_hub": True, "railway": True},
            {"id": "zone_4", "name": "Tondiarpet", "type": "zone", "wards": "43-56",
             "areas": ["Tondiarpet", "Washermenpet", "Royapuram"], "population": 290000,
             "port_area": True, "commercial": True},
            {"id": "zone_5", "name": "Royapuram", "type": "zone", "wards": "57-70",
             "areas": ["Royapuram", "Sowcarpet", "Georgetown"], "population": 270000,
             "heritage": True, "commercial": True, "wholesale": True},
            {"id": "zone_6", "name": "Thiru Vi Ka Nagar", "type": "zone", "wards": "71-84",
             "areas": ["Ambattur", "Avadi", "Padi"], "population": 350000,
             "industrial": True, "residential": True},
            {"id": "zone_7", "name": "Ambattur", "type": "zone", "wards": "85-98",
             "areas": ["Ambattur", "Mogappair", "Anna Nagar West"], "population": 380000,
             "it_corridor": True, "residential": True},
            {"id": "zone_8", "name": "Anna Nagar", "type": "zone", "wards": "99-112",
             "areas": ["Anna Nagar", "Kilpauk", "Purasawalkam"], "population": 300000,
             "planned_city": True, "commercial": True},
            {"id": "zone_9", "name": "Teynampet", "type": "zone", "wards": "113-126",
             "areas": ["Teynampet", "T.Nagar", "Nandanam"], "population": 280000,
             "commercial_hub": True, "shopping": True},
            {"id": "zone_10", "name": "Kodambakkam", "type": "zone", "wards": "127-140",
             "areas": ["Kodambakkam", "Vadapalani", "Saidapet"], "population": 290000,
             "film_city": True, "media": True},
            {"id": "zone_11", "name": "Valasaravakkam", "type": "zone", "wards": "141-154",
             "areas": ["Valasaravakkam", "Porur", "Ramapuram"], "population": 340000,
             "outer_ring_road": True, "residential": True},
            {"id": "zone_12", "name": "Alandur", "type": "zone", "wards": "155-168",
             "areas": ["Alandur", "St. Thomas Mount", "Guindy"], "population": 250000,
             "airport_zone": True, "transport_hub": True},
            {"id": "zone_13", "name": "Adyar", "type": "zone", "wards": "169-182",
             "areas": ["Adyar", "Besant Nagar", "Thiruvanmiyur"], "population": 320000,
             "coastal": True, "upscale": True, "educational": True},
            {"id": "zone_14", "name": "Perungudi", "type": "zone", "wards": "183-196",
             "areas": ["Perungudi", "Velachery", "Pallikaranai"], "population": 360000,
             "it_corridor": True, "marshland": True},
            {"id": "zone_15", "name": "Sholinganallur", "type": "zone", "wards": "197-200",
             "areas": ["Sholinganallur", "Navalur", "Siruseri"], "population": 290000,
             "omr_corridor": True, "it_hub": True}
        ]
        
        for zone in zones_data:
            zone_id = zone.pop("id")
            self.civic_graph.add_node(zone_id, **zone)
        
        # Add sample wards (representative set)
        sample_wards = [
            {"id": "ward_1", "number": 1, "type": "ward", "zone": "zone_1", 
             "population": 18000, "households": 4500, "area_sq_km": 2.1},
            {"id": "ward_50", "number": 50, "type": "ward", "zone": "zone_4",
             "population": 21000, "households": 5250, "area_sq_km": 1.8},
            {"id": "ward_100", "number": 100, "type": "ward", "zone": "zone_8",
             "population": 19500, "households": 4875, "area_sq_km": 2.3},
            {"id": "ward_150", "number": 150, "type": "ward", "zone": "zone_11",
             "population": 22000, "households": 5500, "area_sq_km": 2.7},
            {"id": "ward_200", "number": 200, "type": "ward", "zone": "zone_15",
             "population": 25000, "households": 6250, "area_sq_km": 3.2}
        ]
        
        for ward in sample_wards:
            ward_id = ward.pop("id")
            self.civic_graph.add_node(ward_id, **ward)
    
    def _add_service_entities(self):
        """Add civic services and their attributes"""
        services = [
            {
                "id": "water_supply", "name": "Water Supply", "type": "civic_service",
                "department": "cmwssb", "sla": "24 hours", "digital": True,
                "complaints_per_month": 2500, "satisfaction": 4.1
            },
            {
                "id": "waste_management", "name": "Solid Waste Management", "type": "civic_service",
                "department": "gcc", "sla": "24 hours", "digital": True,
                "complaints_per_month": 3200, "satisfaction": 3.8
            },
            {
                "id": "electricity_supply", "name": "Electricity Supply", "type": "civic_service",
                "department": "tneb", "sla": "4 hours", "digital": True,
                "complaints_per_month": 2800, "satisfaction": 4.0
            },
            {
                "id": "property_tax", "name": "Property Tax", "type": "civic_service",
                "department": "revenue_dept", "sla": "15 days", "digital": True,
                "queries_per_month": 15000, "satisfaction": 4.3
            },
            {
                "id": "birth_certificate", "name": "Birth Certificate", "type": "civic_service",
                "department": "revenue_dept", "sla": "7 days", "digital": True,
                "applications_per_month": 8500, "satisfaction": 4.4
            },
            {
                "id": "road_maintenance", "name": "Road Maintenance", "type": "civic_service",
                "department": "gcc", "sla": "7 days", "digital": True,
                "complaints_per_month": 2400, "satisfaction": 3.5
            },
            {
                "id": "health_services", "name": "Public Health Services", "type": "civic_service",
                "department": "health_dept", "sla": "immediate", "digital": False,
                "patients_per_month": 45000, "satisfaction": 3.9
            }
        ]
        
        for service in services:
            service_id = service.pop("id")
            self.civic_graph.add_node(service_id, **service)
    
    def _add_personnel_entities(self):
        """Add key personnel and their roles"""
        personnel = [
            {
                "id": "commissioner", "name": "Corporation Commissioner", "type": "personnel",
                "level": "city", "department": "gcc", "authority": "executive"
            },
            {
                "id": "md_metro_water", "name": "Managing Director CMWSSB", "type": "personnel",
                "level": "city", "department": "cmwssb", "authority": "executive"
            },
            {
                "id": "police_commissioner", "name": "Police Commissioner", "type": "personnel",
                "level": "city", "department": "police_dept", "authority": "executive"
            }
        ]
        
        # Add zone officers
        for i in range(1, 16):
            personnel.append({
                "id": f"zone_officer_{i}", "name": f"Zone {i} Officer", "type": "personnel",
                "level": "zone", "zone": f"zone_{i}", "authority": "administrative"
            })
        
        for person in personnel:
            person_id = person.pop("id")
            self.civic_graph.add_node(person_id, **person)
    
    def _add_entity_relationships(self):
        """Add relationships between entities"""
        
        # Department to zone relationships
        for i in range(1, 16):
            self.civic_graph.add_edge("gcc", f"zone_{i}", relationship="administers", weight=1.0)
            self.civic_graph.add_edge(f"zone_officer_{i}", f"zone_{i}", relationship="manages", weight=1.0)
        
        # Service to department relationships
        service_dept_mapping = {
            "water_supply": "cmwssb",
            "waste_management": "gcc", 
            "electricity_supply": "tneb",
            "property_tax": "revenue_dept",
            "birth_certificate": "revenue_dept",
            "road_maintenance": "gcc",
            "health_services": "health_dept"
        }
        
        for service, dept in service_dept_mapping.items():
            self.civic_graph.add_edge(dept, service, relationship="provides", weight=1.0)
        
        # Ward to zone relationships
        ward_zone_mapping = {
            "ward_1": "zone_1", "ward_50": "zone_4", "ward_100": "zone_8",
            "ward_150": "zone_11", "ward_200": "zone_15"
        }
        
        for ward, zone in ward_zone_mapping.items():
            self.civic_graph.add_edge(zone, ward, relationship="contains", weight=1.0)
        
        # Personnel hierarchy
        self.civic_graph.add_edge("commissioner", "gcc", relationship="heads", weight=1.0)
        self.civic_graph.add_edge("md_metro_water", "cmwssb", relationship="heads", weight=1.0)
        self.civic_graph.add_edge("police_commissioner", "police_dept", relationship="heads", weight=1.0)
    
    def _calculate_graph_metrics(self):
        """Calculate graph centrality and importance metrics"""
        
        # Calculate various centrality measures
        self.pagerank_scores = nx.pagerank(self.civic_graph)
        self.betweenness_centrality = nx.betweenness_centrality(self.civic_graph)
        self.degree_centrality = nx.degree_centrality(self.civic_graph)
        
        # Add centrality scores as node attributes
        for node in self.civic_graph.nodes():
            self.civic_graph.nodes[node]['pagerank'] = self.pagerank_scores.get(node, 0)
            self.civic_graph.nodes[node]['betweenness'] = self.betweenness_centrality.get(node, 0)
            self.civic_graph.nodes[node]['degree_centrality'] = self.degree_centrality.get(node, 0)
    
    def knowledge_reasoning(self, query: str, user_area: str = None) -> Dict[str, Any]:
        """Perform knowledge graph reasoning for civic queries"""
        
        # Step 1: Extract entities from query
        extracted_entities = self._extract_query_entities(query, user_area)
        
        # Step 2: Find relevant subgraph
        relevant_subgraph = self._find_relevant_subgraph(extracted_entities)
        
        # Step 3: Apply graph reasoning
        reasoning_result = self._apply_graph_reasoning(query, extracted_entities, relevant_subgraph)
        
        # Step 4: Generate structured response
        if reasoning_result:
            response = self._generate_knowledge_response(query, reasoning_result)
            success = True
        else:
            response = self._generate_fallback_knowledge_response(query, extracted_entities)
            success = False
        
        return {
            "success": success,
            "response": response,
            "entities": extracted_entities,
            "subgraph_size": len(relevant_subgraph.nodes()),
            "reasoning_paths": reasoning_result.get("paths", []),
            "confidence": reasoning_result.get("confidence", 0.5),
            "source": "KAG - Knowledge Graph Reasoning"
        }
    
    def _extract_query_entities(self, query: str, user_area: str = None) -> List[str]:
        """Extract relevant entities from query"""
        query_lower = query.lower()
        extracted_entities = []
        
        # Extract service entities
        service_keywords = {
            "water_supply": ["water", "supply", "pressure", "quality", "metro water"],
            "waste_management": ["garbage", "waste", "trash", "collection", "dustbin"],
            "electricity_supply": ["power", "electricity", "outage", "streetlight", "eb"],
            "property_tax": ["property", "tax", "assessment", "payment"],
            "birth_certificate": ["birth", "certificate", "birth certificate"],
            "road_maintenance": ["road", "pothole", "repair", "maintenance"],
            "health_services": ["hospital", "health", "clinic", "medical"]
        }
        
        for service, keywords in service_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_entities.append(service)
        
        # Extract geographical entities
        for node in self.civic_graph.nodes():
            node_data = self.civic_graph.nodes[node]
            if node_data.get("type") in ["zone", "ward"]:
                # Check if zone/ward name or areas mentioned
                if node_data.get("name") and node_data["name"].lower() in query_lower:
                    extracted_entities.append(node)
                elif node_data.get("areas"):
                    for area in node_data["areas"]:
                        if area.lower() in query_lower:
                            extracted_entities.append(node)
                            break
        
        # Extract department entities
        dept_keywords = {
            "gcc": ["corporation", "gcc", "municipal"],
            "cmwssb": ["metro water", "water board", "cmwssb"],
            "tneb": ["electricity board", "tneb", "eb"],
            "revenue_dept": ["revenue", "certificate", "tax"],
            "health_dept": ["health", "hospital", "medical"],
            "police_dept": ["police"],
            "fire_dept": ["fire", "rescue"]
        }
        
        for dept, keywords in dept_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_entities.append(dept)
        
        return list(set(extracted_entities))  # Remove duplicates
    
    def _find_relevant_subgraph(self, entities: List[str]) -> nx.Graph:
        """Find relevant subgraph based on extracted entities"""
        if not entities:
            return nx.Graph()
        
        # Start with extracted entities
        relevant_nodes = set(entities)
        
        # Add immediate neighbors
        for entity in entities:
            if entity in self.civic_graph:
                neighbors = list(self.civic_graph.neighbors(entity))
                relevant_nodes.update(neighbors[:5])  # Limit to top 5 neighbors
        
        # Create subgraph
        subgraph = self.civic_graph.subgraph(relevant_nodes)
        return subgraph
    
    def _apply_graph_reasoning(self, query: str, entities: List[str], subgraph: nx.Graph) -> Dict[str, Any]:
        """Apply graph-based reasoning"""
        
        if not entities or subgraph.number_of_nodes() == 0:
            return {}
        
        reasoning_result = {
            "primary_entities": entities[:3],  # Top 3 most relevant
            "paths": [],
            "related_services": [],
            "departments": [],
            "contacts": [],
            "procedures": [],
            "confidence": 0.0
        }
        
        # Find paths between entities
        entity_pairs = [(entities[i], entities[j]) for i in range(len(entities)) 
                       for j in range(i+1, len(entities)) if i != j]
        
        for source, target in entity_pairs[:5]:  # Limit path finding
            if source in subgraph and target in subgraph:
                try:
                    path = nx.shortest_path(subgraph, source, target)
                    reasoning_result["paths"].append({
                        "source": source,
                        "target": target,
                        "path": path,
                        "length": len(path) - 1
                    })
                except nx.NetworkXNoPath:
                    continue
        
        # Extract departments and services
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            node_type = node_data.get("type", "")
            
            if node_type == "municipal_body" or node_type in ["water_authority", "power_utility", "health_authority"]:
                reasoning_result["departments"].append({
                    "id": node,
                    "name": node_data.get("name", ""),
                    "type": node_type,
                    "head": node_data.get("head", ""),
                    "importance": node_data.get("pagerank", 0)
                })
            
            elif node_type == "civic_service":
                reasoning_result["related_services"].append({
                    "id": node,
                    "name": node_data.get("name", ""),
                    "department": node_data.get("department", ""),
                    "sla": node_data.get("sla", ""),
                    "satisfaction": node_data.get("satisfaction", 0)
                })
        
        # Extract contact information
        contact_mapping = {
            "gcc": "1913", "cmwssb": "044-45671200", "tneb": "94987-94987",
            "revenue_dept": "044-25619515", "health_dept": "044-25619671",
            "police_dept": "100", "fire_dept": "101"
        }
        
        for dept in reasoning_result["departments"]:
            if dept["id"] in contact_mapping:
                reasoning_result["contacts"].append({
                    "department": dept["name"],
                    "contact": contact_mapping[dept["id"]]
                })
        
        # Calculate confidence based on graph connectivity and entity relevance
        base_confidence = min(0.8, len(entities) * 0.2)
        path_bonus = min(0.15, len(reasoning_result["paths"]) * 0.05)
        reasoning_result["confidence"] = base_confidence + path_bonus
        
        return reasoning_result
    
    def _generate_knowledge_response(self, query: str, reasoning_result: Dict) -> str:
        """Generate response from knowledge graph reasoning"""
        
        response_parts = []
        
        # Add department information
        if reasoning_result["departments"]:
            primary_dept = reasoning_result["departments"][0]
            response_parts.append(f"🏢 **Responsible Department**: {primary_dept['name']} (Head: {primary_dept['head']})")
        
        # Add service information
        if reasoning_result["related_services"]:
            primary_service = reasoning_result["related_services"][0]
            response_parts.append(f"⚙️ **Service**: {primary_service['name']} (SLA: {primary_service['sla']})")
            if primary_service.get("satisfaction"):
                response_parts.append(f"📊 **Satisfaction Rating**: {primary_service['satisfaction']}/5.0")
        
        # Add contact information
        if reasoning_result["contacts"]:
            contact_info = reasoning_result["contacts"][0]
            response_parts.append(f"📞 **Contact**: {contact_info['department']} - {contact_info['contact']}")
        
        # Add procedural guidance based on query type
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "process", "procedure", "steps"]):
            if "property tax" in query_lower:
                response_parts.append("📋 **Process**: Online payment available at chennaicorporation.gov.in/propertytax")
            elif "birth certificate" in query_lower:
                response_parts.append("📋 **Process**: Apply online at tnreginet.gov.in or visit registrar office")
            elif "water connection" in query_lower:
                response_parts.append("📋 **Process**: Submit application → Site inspection → Payment → Installation")
        
        # Add relationship insights from graph paths
        if reasoning_result["paths"]:
            path_insight = reasoning_result["paths"][0]
            if len(path_insight["path"]) > 2:
                middle_entity = path_insight["path"][1]
                if middle_entity in self.civic_graph:
                    entity_data = self.civic_graph.nodes[middle_entity]
                    if entity_data.get("name"):
                        response_parts.append(f"🔗 **Related**: This also involves {entity_data['name']}")
        
        if not response_parts:
            return "I found relevant information in the knowledge graph but need more specific details to provide a complete answer."
        
        return "\n\n".join(response_parts)
    
    def _generate_fallback_knowledge_response(self, query: str, entities: List[str]) -> str:
        """Generate fallback response when reasoning fails"""
        
        if entities:
            # Try to provide information about recognized entities
            entity_info = []
            for entity in entities[:2]:  # Limit to 2 entities
                if entity in self.civic_graph:
                    node_data = self.civic_graph.nodes[entity]
                    if node_data.get("name"):
                        entity_info.append(f"• {node_data['name']}: {node_data.get('type', 'civic entity')}")
            
            if entity_info:
                return f"I recognize these entities from your query:\n" + "\n".join(entity_info) + f"\n\nFor specific assistance, please contact Chennai Corporation at 1913."
        
        return "I couldn't find specific matches in the knowledge graph. For general civic assistance, contact Chennai Corporation at 1913."
    
    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        if entity_id in self.civic_graph:
            node_data = dict(self.civic_graph.nodes[entity_id])
            
            # Add relationship information
            relationships = []
            for neighbor in self.civic_graph.neighbors(entity_id):
                edge_data = self.civic_graph[entity_id][neighbor]
                relationships.append({
                    "target": neighbor,
                    "relationship": edge_data.get("relationship", "connected"),
                    "weight": edge_data.get("weight", 0.5)
                })
            
            node_data["relationships"] = relationships
            return node_data
        
        return {}
    
    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two entities"""
        try:
            return nx.shortest_path(self.civic_graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_similar_entities(self, entity_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar entities based on graph structure"""
        if entity_id not in self.civic_graph:
            return []
        
        # Get entities with similar attributes
        target_attributes = self.civic_graph.nodes[entity_id]
        similar_entities = []
        
        for node in self.civic_graph.nodes():
            if node == entity_id:
                continue
            
            node_attributes = self.civic_graph.nodes[node]
            similarity_score = self._calculate_attribute_similarity(target_attributes, node_attributes)
            
            if similarity_score > 0.3:  # Similarity threshold
                similar_entities.append({
                    "id": node,
                    "attributes": node_attributes,
                    "similarity": similarity_score
                })
        
        # Sort by similarity and return top results
        similar_entities.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_entities[:limit]
    
    def _calculate_attribute_similarity(self, attr1: Dict, attr2: Dict) -> float:
        """Calculate similarity between two entity attribute sets"""
        
        # Compare type similarity (highest weight)
        type_similarity = 1.0 if attr1.get("type") == attr2.get("type") else 0.0
        
        # Compare common attributes
        common_attributes = set(attr1.keys()) & set(attr2.keys())
        attribute_matches = sum(1 for attr in common_attributes 
                              if attr1[attr] == attr2[attr])
        
        attribute_similarity = attribute_matches / max(len(common_attributes), 1)
        
        # Weighted combination
        return 0.6 * type_similarity + 0.4 * attribute_similarity
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return {
            "total_nodes": self.civic_graph.number_of_nodes(),
            "total_edges": self.civic_graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "average_degree": sum(dict(self.civic_graph.degree()).values()) / self.civic_graph.number_of_nodes(),
            "density": nx.density(self.civic_graph),
            "most_central_nodes": self._get_most_central_nodes(),
            "connected_components": nx.number_weakly_connected_components(self.civic_graph)
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = {}
        for node in self.civic_graph.nodes():
            node_type = self.civic_graph.nodes[node].get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _get_most_central_nodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most central nodes by PageRank score"""
        most_central = []
        for node, score in sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            node_data = self.civic_graph.nodes[node]
            most_central.append({
                "id": node,
                "name": node_data.get("name", node),
                "type": node_data.get("type", "unknown"),
                "pagerank_score": score
            })
        return most_central
