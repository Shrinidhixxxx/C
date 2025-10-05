"""
CAG Engine - Cache-Augmented Generation
Intelligent caching layer for Chennai civic data with 24-hour expiry
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import pandas as pd

class CAGEngine:
    def __init__(self):
        """Initialize CAG engine with intelligent caching"""
        self.cache_dir = "cache"
        self.cache_expiry_hours = 24
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "total_queries": 0}
        
        # Initialize cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load static civic data
        self.static_data = self._load_static_civic_data()
        self.frequent_queries = self._load_frequent_queries()
        self.emergency_data = self._load_emergency_data()
        
        # Initialize cache with frequently accessed data
        self._warm_up_cache()
    
    def _load_static_civic_data(self) -> Dict[str, Any]:
        """Load static civic information that rarely changes"""
        return {
            "office_timings": {
                "corporation_offices": {
                    "weekdays": "10:00 AM - 5:00 PM (Monday-Friday)",
                    "saturday": "10:00 AM - 1:00 PM (Selected offices)",
                    "sunday": "Closed",
                    "lunch_break": "1:00 PM - 2:00 PM"
                },
                "zone_offices": {
                    "weekdays": "9:30 AM - 5:30 PM (Monday-Saturday)",
                    "sunday": "Closed",
                    "lunch_break": "1:00 PM - 2:00 PM"
                },
                "metro_water": {
                    "weekdays": "9:30 AM - 5:30 PM (Monday-Friday)",
                    "saturday": "9:30 AM - 1:00 PM",
                    "emergency": "24/7 complaint line"
                }
            },
            "contact_directory": {
                "primary_helplines": {
                    "chennai_corporation": "1913",
                    "metro_water": "044-45671200",
                    "tneb": "94987-94987",
                    "police": "100",
                    "fire": "101",
                    "ambulance": "108"
                },
                "specialized_helplines": {
                    "women_helpline": "1091",
                    "child_helpline": "1098", 
                    "tourist_helpline": "1363",
                    "cyber_crime": "155620",
                    "senior_citizen": "14567",
                    "anti_corruption": "1064"
                },
                "department_contacts": {
                    "revenue_department": "044-25619515",
                    "health_department": "044-25619671",
                    "education_department": "044-25619600",
                    "traffic_police": "044-28447777"
                }
            },
            "fee_structure": {
                "certificates": {
                    "birth_certificate": {"first_copy": "₹10", "additional": "₹5"},
                    "death_certificate": {"first_copy": "₹10", "additional": "₹5"},
                    "marriage_certificate": {"first_copy": "₹15", "additional": "₹10"}
                },
                "licenses": {
                    "trade_license": {"shops": "₹500-2000", "restaurants": "₹1000-5000"},
                    "building_permit": {"residential": "₹500-5000", "commercial": "₹2000-25000"}
                },
                "connections": {
                    "water_connection": {"domestic": "₹1500-3000", "commercial": "₹5000-15000"},
                    "eb_connection": {"domestic": "₹500-1500", "commercial": "₹2000-8000"}
                }
            },
            "document_requirements": {
                "birth_certificate": [
                    "Hospital discharge summary/Medical certificate",
                    "Parents' Aadhaar/Photo ID", 
                    "Address proof",
                    "Marriage certificate of parents (if applicable)"
                ],
                "death_certificate": [
                    "Medical certificate of cause of death",
                    "Aadhaar/Photo ID of applicant",
                    "Address proof",
                    "Relationship proof with deceased"
                ],
                "property_tax_payment": [
                    "Property tax demand notice",
                    "Previous receipt (if available)",
                    "Property ownership documents",
                    "Bank account details for online payment"
                ],
                "water_connection": [
                    "Application form with photos",
                    "Property tax receipt",
                    "Building plan approval",
                    "Aadhaar card",
                    "Address proof"
                ]
            }
        }
    
    def _load_frequent_queries(self) -> Dict[str, Dict[str, Any]]:
        """Load frequently asked questions and their cached responses"""
        return {
            "property_tax_online_payment": {
                "query_patterns": ["property tax online", "pay property tax", "tax payment online"],
                "response": "Property tax can be paid online at chennaicorporation.gov.in/propertytax. You'll need your assessment number and can pay via net banking, debit/credit cards, or UPI. Due dates are April 30 and October 31 each year.",
                "category": "property_tax",
                "department": "Revenue Department",
                "contact": "044-25619515",
                "portal": "chennaicorporation.gov.in/propertytax",
                "cache_priority": "high",
                "access_count": 1250
            },
            "garbage_collection_schedule": {
                "query_patterns": ["garbage collection", "waste pickup", "when is garbage collected"],
                "response": "Garbage collection happens daily between 6:00 AM - 12:00 PM in residential areas and 7:00 AM - 2:00 PM in commercial areas. Please segregate waste into wet (green), dry (blue), and hazardous (red) bins.",
                "category": "waste_management",
                "department": "Solid Waste Management",
                "contact": "1913",
                "cache_priority": "high",
                "access_count": 980
            },
            "birth_certificate_process": {
                "query_patterns": ["birth certificate", "how to get birth certificate", "birth registration"],
                "response": "Birth certificates can be obtained by applying at the Registrar office or online at tnreginet.gov.in. Required documents: Hospital discharge summary, parents' ID, address proof. Fee: ₹10 for first copy. Processing time: 7 working days.",
                "category": "certificates",
                "department": "Revenue Department", 
                "contact": "044-25619515",
                "portal": "tnreginet.gov.in",
                "cache_priority": "high",
                "access_count": 875
            },
            "water_supply_complaint": {
                "query_patterns": ["water problem", "no water supply", "water pressure low"],
                "response": "Water supply issues can be reported to Chennai Metro Water at 044-45671200 or through their mobile app. For emergency water supply, call 044-25619417. Tanker water can be requested for areas with supply disruption.",
                "category": "water_supply",
                "department": "Chennai Metro Water",
                "contact": "044-45671200",
                "emergency": "044-25619417",
                "cache_priority": "high",
                "access_count": 1120
            },
            "electricity_complaint": {
                "query_patterns": ["power outage", "electricity complaint", "streetlight not working"],
                "response": "Electricity complaints can be filed with TNEB at 94987-94987 or through tnebnet.org. Power outages are typically resolved within 2-4 hours. Streetlight issues take 48-72 hours for resolution.",
                "category": "electricity",
                "department": "Tamil Nadu Electricity Board",
                "contact": "94987-94987",
                "portal": "tnebnet.org",
                "cache_priority": "high",
                "access_count": 760
            },
            "trade_license_renewal": {
                "query_patterns": ["trade license", "business license", "shop license"],
                "response": "Trade licenses can be renewed online at chennaicorporation.gov.in or at zone offices. Renewal should be done before expiry. Required: Previous license, tax compliance certificate, updated documents. Fee varies from ₹500-5000 based on business type.",
                "category": "licenses",
                "department": "Revenue Department",
                "contact": "044-25619515",
                "portal": "chennaicorporation.gov.in",
                "cache_priority": "medium",
                "access_count": 450
            },
            "emergency_contacts": {
                "query_patterns": ["emergency numbers", "helpline", "emergency contact"],
                "response": "Emergency contacts: Fire-101, Police-100, Ambulance-108, Chennai Corporation-1913, Metro Water Emergency-044-25619417, Women Helpline-1091, Child Helpline-1098, Cyber Crime-155620",
                "category": "emergency",
                "department": "Multiple",
                "contact": "Various emergency numbers",
                "cache_priority": "critical",
                "access_count": 2100
            }
        }
    
    def _load_emergency_data(self) -> Dict[str, Any]:
        """Load critical emergency information"""
        return {
            "fire_emergency": {
                "primary_contact": "101",
                "stations": 32,
                "response_time": "8-12 minutes",
                "headquarters": "044-28527004",
                "procedure": [
                    "Call 101 immediately",
                    "Provide exact location and nature of fire",
                    "Evacuate the area safely", 
                    "Do not use elevators",
                    "Wait for fire personnel at safe distance"
                ]
            },
            "medical_emergency": {
                "primary_contact": "108",
                "service": "24/7 free ambulance",
                "response_time": "15-20 minutes",
                "major_hospitals": {
                    "government_general": "044-25305000",
                    "stanley_medical": "044-25281341",
                    "rajiv_gandhi": "044-22359090",
                    "kilpauk_medical": "044-26640307"
                },
                "procedure": [
                    "Call 108 for ambulance",
                    "Provide patient condition details",
                    "Give exact address with landmarks",
                    "Keep patient stable until help arrives",
                    "Have medical documents ready"
                ]
            },
            "police_emergency": {
                "primary_contact": "100",
                "control_rooms": {
                    "central": "044-23452300",
                    "north": "044-26242640", 
                    "south": "044-24912340",
                    "west": "044-23746000"
                },
                "specialized_contacts": {
                    "traffic_police": "044-28447777",
                    "women_helpline": "1091",
                    "cyber_crime": "155620"
                }
            },
            "natural_disasters": {
                "flood_control": "044-25619000",
                "cyclone_warning": "044-25361721",
                "disaster_management": "044-25619515",
                "relief_camps": "1070"
            }
        }
    
    def _warm_up_cache(self):
        """Pre-load frequently accessed data into memory cache"""
        high_priority_queries = [
            query_id for query_id, data in self.frequent_queries.items()
            if data.get("cache_priority") in ["critical", "high"]
        ]
        
        for query_id in high_priority_queries:
            cache_key = f"frequent_{query_id}"
            self.memory_cache[cache_key] = {
                "data": self.frequent_queries[query_id],
                "timestamp": datetime.now(),
                "hit_count": 0
            }
    
    def get_cached_response(self, query: str, user_area: str = None) -> Dict[str, Any]:
        """Get response from cache or generate new cached response"""
        start_time = datetime.now()
        self.cache_stats["total_queries"] += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, user_area)
        
        # Check memory cache first
        cached_result = self._check_memory_cache(cache_key)
        if cached_result:
            self.cache_stats["hits"] += 1
            return self._format_cache_response(cached_result, True, start_time)
        
        # Check frequent queries
        frequent_result = self._check_frequent_queries(query)
        if frequent_result:
            self.cache_stats["hits"] += 1
            # Update memory cache
            self._update_memory_cache(cache_key, frequent_result)
            return self._format_cache_response(frequent_result, True, start_time)
        
        # Check file cache
        file_cached_result = self._check_file_cache(cache_key)
        if file_cached_result:
            self.cache_stats["hits"] += 1
            return self._format_cache_response(file_cached_result, True, start_time)
        
        # Generate new response and cache it
        self.cache_stats["misses"] += 1
        new_response = self._generate_and_cache_response(query, user_area, cache_key)
        
        return self._format_cache_response(new_response, False, start_time)
    
    def _generate_cache_key(self, query: str, user_area: str = None) -> str:
        """Generate unique cache key for query"""
        content = f"{query.lower()}_{user_area or 'general'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_memory_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check in-memory cache"""
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            
            # Check expiry
            age = datetime.now() - cached_item["timestamp"]
            if age.total_seconds() < (self.cache_expiry_hours * 3600):
                cached_item["hit_count"] += 1
                return cached_item["data"]
            else:
                # Remove expired item
                del self.memory_cache[cache_key]
        
        return None
    
    def _check_frequent_queries(self, query: str) -> Optional[Dict[str, Any]]:
        """Check against frequent queries patterns"""
        query_lower = query.lower()
        
        for query_id, query_data in self.frequent_queries.items():
            patterns = query_data.get("query_patterns", [])
            if any(pattern in query_lower for pattern in patterns):
                return query_data
        
        return None
    
    def _check_file_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check persistent file cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check expiry
                age = datetime.now() - cached_data["timestamp"]
                if age.total_seconds() < (self.cache_expiry_hours * 3600):
                    return cached_data["data"]
                else:
                    # Remove expired file
                    os.remove(cache_file)
            except Exception:
                # Handle corrupted cache file
                if os.path.exists(cache_file):
                    os.remove(cache_file)
        
        return None
    
    def _generate_and_cache_response(self, query: str, user_area: str, cache_key: str) -> Dict[str, Any]:
        """Generate new response and cache it"""
        
        # Classify query type
        query_category = self._classify_query(query)
        
        # Generate response based on category
        if query_category == "emergency":
            response_data = self._handle_emergency_query(query)
        elif query_category == "contact":
            response_data = self._handle_contact_query(query)
        elif query_category == "procedure":
            response_data = self._handle_procedure_query(query)
        elif query_category == "timing":
            response_data = self._handle_timing_query(query)
        elif query_category == "fees":
            response_data = self._handle_fees_query(query)
        else:
            response_data = self._handle_general_query(query, user_area)
        
        # Cache the response
        if response_data.get("success"):
            self._save_to_cache(cache_key, response_data)
        
        return response_data
    
    def _classify_query(self, query: str) -> str:
        """Classify query into categories"""
        query_lower = query.lower()
        
        emergency_keywords = ["emergency", "urgent", "fire", "ambulance", "police", "accident", "help"]
        contact_keywords = ["contact", "phone", "number", "helpline", "call"]
        procedure_keywords = ["how to", "process", "procedure", "steps", "apply"]
        timing_keywords = ["timings", "hours", "when open", "office hours"]
        fees_keywords = ["fee", "cost", "charge", "payment", "price"]
        
        if any(keyword in query_lower for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in query_lower for keyword in contact_keywords):
            return "contact"  
        elif any(keyword in query_lower for keyword in procedure_keywords):
            return "procedure"
        elif any(keyword in query_lower for keyword in timing_keywords):
            return "timing"
        elif any(keyword in query_lower for keyword in fees_keywords):
            return "fees"
        else:
            return "general"
    
    def _handle_emergency_query(self, query: str) -> Dict[str, Any]:
        """Handle emergency-related queries"""
        query_lower = query.lower()
        
        if "fire" in query_lower:
            emergency_info = self.emergency_data["fire_emergency"]
            response = f"""🚨 **FIRE EMERGENCY**
            
**Primary Contact:** {emergency_info['primary_contact']}
**Response Time:** {emergency_info['response_time']}
**Headquarters:** {emergency_info['headquarters']}

**Emergency Procedure:**
{chr(10).join(['• ' + step for step in emergency_info['procedure']])}

**Additional Info:** Chennai has {emergency_info['stations']} fire stations strategically located across the city."""
            
        elif "ambulance" in query_lower or "medical" in query_lower:
            medical_info = self.emergency_data["medical_emergency"]
            response = f"""🚑 **MEDICAL EMERGENCY**
            
**Primary Contact:** {medical_info['primary_contact']} ({medical_info['service']})
**Response Time:** {medical_info['response_time']}

**Major Hospitals:**
{chr(10).join([f'• {name.replace("_", " ").title()}: {phone}' for name, phone in medical_info['major_hospitals'].items()])}

**Emergency Procedure:**
{chr(10).join(['• ' + step for step in medical_info['procedure']])}"""
            
        elif "police" in query_lower:
            police_info = self.emergency_data["police_emergency"]
            response = f"""🚓 **POLICE EMERGENCY**
            
**Primary Contact:** {police_info['primary_contact']}

**Control Rooms:**
{chr(10).join([f'• {name.title()}: {phone}' for name, phone in police_info['control_rooms'].items()])}

**Specialized Contacts:**
{chr(10).join([f'• {name.replace("_", " ").title()}: {phone}' for name, phone in police_info['specialized_contacts'].items()])}"""
            
        else:
            response = """🚨 **ALL EMERGENCY CONTACTS**
            
• **Fire:** 101
• **Police:** 100  
• **Ambulance:** 108
• **Women Helpline:** 1091
• **Child Helpline:** 1098
• **Disaster Management:** 044-25619515
• **Chennai Corporation:** 1913"""
        
        return {
            "success": True,
            "response": response,
            "category": "emergency",
            "priority": "critical",
            "contact": "Multiple emergency numbers"
        }
    
    def _handle_contact_query(self, query: str) -> Dict[str, Any]:
        """Handle contact information queries"""
        query_lower = query.lower()
        contacts = self.static_data["contact_directory"]
        
        if "water" in query_lower:
            response = f"💧 **Metro Water Contacts:**\n• Main: {contacts['primary_helplines']['metro_water']}\n• Emergency: 044-25619417"
        elif "electricity" in query_lower or "power" in query_lower:
            response = f"⚡ **TNEB Contact:**\n• Helpline: {contacts['primary_helplines']['tneb']}\n• Portal: tnebnet.org"
        elif "corporation" in query_lower:
            response = f"🏛️ **Chennai Corporation:**\n• General Helpline: {contacts['primary_helplines']['chennai_corporation']}\n• Revenue Dept: {contacts['department_contacts']['revenue_department']}"
        else:
            response = "📞 **Important Contacts:**\n"
            for service, number in contacts['primary_helplines'].items():
                response += f"• {service.replace('_', ' ').title()}: {number}\n"
        
        return {
            "success": True,
            "response": response,
            "category": "contact",
            "priority": "high"
        }
    
    def _handle_procedure_query(self, query: str) -> Dict[str, Any]:
        """Handle procedure-related queries"""
        query_lower = query.lower()
        
        if "birth certificate" in query_lower:
            docs = self.static_data["document_requirements"]["birth_certificate"]
            fees = self.static_data["fee_structure"]["certificates"]["birth_certificate"]
            
            response = f"""📄 **Birth Certificate Process:**

**Required Documents:**
{chr(10).join(['• ' + doc for doc in docs])}

**Fee Structure:**
• First copy: {fees['first_copy']}
• Additional copies: {fees['additional']}

**Process:**
1. Collect application form from Registrar office
2. Submit with required documents
3. Pay prescribed fee
4. Get acknowledgment receipt
5. Collect certificate in 7 working days

**Online Portal:** tnreginet.gov.in
**Contact:** Revenue Department - 044-25619515"""
            
        elif "property tax" in query_lower:
            docs = self.static_data["document_requirements"]["property_tax_payment"]
            
            response = f"""🏠 **Property Tax Payment Process:**

**Required Documents:**
{chr(10).join(['• ' + doc for doc in docs])}

**Payment Methods:**
• Online: chennaicorporation.gov.in/propertytax
• Bank payment at authorized branches
• Direct payment at Corporation offices

**Due Dates:** April 30 and October 31
**Penalty:** 1% per month for delayed payment
**Contact:** Revenue Department - 044-25619515"""
            
        elif "water connection" in query_lower:
            docs = self.static_data["document_requirements"]["water_connection"]
            fees = self.static_data["fee_structure"]["connections"]["water_connection"]
            
            response = f"""💧 **New Water Connection Process:**

**Required Documents:**
{chr(10).join(['• ' + doc for doc in docs])}

**Fee Range:**
• Domestic: {fees['domestic']}
• Commercial: {fees['commercial']}

**Process:**
1. Submit application with documents
2. Site inspection by engineer
3. Pay connection charges
4. Schedule installation work
5. Get meter installed and activated

**Contact:** Metro Water - 044-45671200"""
            
        else:
            response = "For specific procedures, please mention the service you need help with (birth certificate, property tax, water connection, etc.) or contact Chennai Corporation at 1913."
        
        return {
            "success": bool("certificate" in query_lower or "tax" in query_lower or "connection" in query_lower),
            "response": response,
            "category": "procedure",
            "priority": "medium"
        }
    
    def _handle_timing_query(self, query: str) -> Dict[str, Any]:
        """Handle office timing queries"""
        timings = self.static_data["office_timings"]
        
        response = """🕐 **Office Timings:**

**Corporation Offices:**
• Weekdays: """ + timings["corporation_offices"]["weekdays"] + """
• Saturday: """ + timings["corporation_offices"]["saturday"] + """
• Lunch Break: """ + timings["corporation_offices"]["lunch_break"] + """

**Zone Offices:**
• """ + timings["zone_offices"]["weekdays"] + """
• Lunch Break: """ + timings["zone_offices"]["lunch_break"] + """

**Metro Water:**
• Weekdays: """ + timings["metro_water"]["weekdays"] + """
• Saturday: """ + timings["metro_water"]["saturday"] + """
• Emergency: """ + timings["metro_water"]["emergency"] + """

**Note:** All offices closed on Sundays and public holidays."""
        
        return {
            "success": True,
            "response": response,
            "category": "timing",
            "priority": "low"
        }
    
    def _handle_fees_query(self, query: str) -> Dict[str, Any]:
        """Handle fee-related queries"""
        query_lower = query.lower()
        fees = self.static_data["fee_structure"]
        
        if "certificate" in query_lower:
            cert_fees = fees["certificates"]
            response = """💰 **Certificate Fees:**

**Birth/Death Certificate:**
• First copy: """ + cert_fees["birth_certificate"]["first_copy"] + """
• Additional copies: """ + cert_fees["birth_certificate"]["additional"] + """

**Marriage Certificate:**
• First copy: """ + cert_fees["marriage_certificate"]["first_copy"] + """
• Additional copies: """ + cert_fees["marriage_certificate"]["additional"] + """
            
**Note:** Fees are subject to periodic revision by the government."""
            
        elif "license" in query_lower:
            license_fees = fees["licenses"]
            response = """💰 **License Fees:**

**Trade License:**
• Shops: """ + license_fees["trade_license"]["shops"] + """
• Restaurants: """ + license_fees["trade_license"]["restaurants"] + """

**Building Permit:**
• Residential: """ + license_fees["building_permit"]["residential"] + """
• Commercial: """ + license_fees["building_permit"]["commercial"] + """
            
**Note:** Exact fee depends on area, type, and local regulations."""
            
        elif "connection" in query_lower:
            conn_fees = fees["connections"]
            response = """💰 **Connection Fees:**

**Water Connection:**
• Domestic: """ + conn_fees["water_connection"]["domestic"] + """
• Commercial: """ + conn_fees["water_connection"]["commercial"] + """

**Electricity Connection:**
• Domestic: """ + conn_fees["eb_connection"]["domestic"] + """
• Commercial: """ + conn_fees["eb_connection"]["commercial"] + """
            
**Note:** Additional charges may apply for materials and labor."""
        else:
            response = "Please specify which service fees you're asking about (certificates, licenses, connections) or contact the relevant department for detailed fee information."
        
        return {
            "success": bool(any(keyword in query_lower for keyword in ["certificate", "license", "connection"])),
            "response": response,
            "category": "fees",
            "priority": "low"
        }
    
    def _handle_general_query(self, query: str, user_area: str) -> Dict[str, Any]:
        """Handle general queries not fitting other categories"""
        
        # Try to provide some general civic information
        response = f"""I understand you're asking about civic services in Chennai. While I don't have specific cached information for your query, here are some general resources:

🏛️ **Chennai Corporation:** 1913
🌐 **Official Website:** chennaicorporation.gov.in  
📱 **Mobile App:** Chennai One
💧 **Metro Water:** 044-45671200
⚡ **TNEB:** 94987-94987

For specific assistance with your query about "{query[:50]}...", please contact the relevant department or visit the nearest zone office."""
        
        if user_area:
            response += f"\n\n📍 **Your Area:** {user_area} - Contact your zone office for location-specific services."
        
        return {
            "success": False,
            "response": response,
            "category": "general",
            "priority": "low",
            "contact": "1913"
        }
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save response to both memory and file cache"""
        
        # Save to memory cache
        self.memory_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now(),
            "hit_count": 0
        }
        
        # Save to file cache for persistence
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    "data": data,
                    "timestamp": datetime.now()
                }, f)
        except Exception as e:
            # Log error but don't fail the request
            pass
    
    def _update_memory_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Update memory cache with fresh data"""
        self.memory_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now(),
            "hit_count": 1
        }
    
    def _format_cache_response(self, data: Dict[str, Any], from_cache: bool, start_time: datetime) -> Dict[str, Any]:
        """Format final cache response"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": data.get("success", True),
            "response": data.get("response", ""),
            "category": data.get("category", "general"),
            "contact": data.get("contact", "1913"),
            "priority": data.get("priority", "medium"),
            "from_cache": from_cache,
            "processing_time": processing_time,
            "source": "CAG - Intelligent Cache"
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        hit_rate = (self.cache_stats["hits"] / max(self.cache_stats["total_queries"], 1)) * 100
        
        return {
            "total_queries": self.cache_stats["total_queries"],
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate_percentage": round(hit_rate, 2),
            "memory_cache_size": len(self.memory_cache),
            "frequent_queries_loaded": len(self.frequent_queries),
            "cache_expiry_hours": self.cache_expiry_hours
        }
    
    def clear_cache(self, cache_type: str = "all") -> Dict[str, Any]:
        """Clear cache based on type"""
        cleared = {"memory": 0, "files": 0}
        
        if cache_type in ["all", "memory"]:
            cleared["memory"] = len(self.memory_cache)
            self.memory_cache.clear()
        
        if cache_type in ["all", "files"]:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            for cache_file in cache_files:
                os.remove(os.path.join(self.cache_dir, cache_file))
                cleared["files"] += 1
        
        return {
            "success": True,
            "cleared": cleared,
            "message": f"Cleared {cache_type} cache successfully"
        }
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache by removing old and least accessed items"""
        
        optimization_stats = {"removed_memory": 0, "removed_files": 0}
        current_time = datetime.now()
        
        # Optimize memory cache
        memory_items_to_remove = []
        for cache_key, cache_item in self.memory_cache.items():
            age_hours = (current_time - cache_item["timestamp"]).total_seconds() / 3600
            hit_count = cache_item["hit_count"]
            
            # Remove if old and rarely accessed
            if age_hours > self.cache_expiry_hours or (age_hours > 12 and hit_count < 2):
                memory_items_to_remove.append(cache_key)
        
        for key in memory_items_to_remove:
            del self.memory_cache[key]
            optimization_stats["removed_memory"] += 1
        
        # Optimize file cache
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        for cache_file in cache_files:
            file_path = os.path.join(self.cache_dir, cache_file)
            try:
                with open(file_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                age_hours = (current_time - cached_data["timestamp"]).total_seconds() / 3600
                if age_hours > self.cache_expiry_hours:
                    os.remove(file_path)
                    optimization_stats["removed_files"] += 1
            except Exception:
                # Remove corrupted files
                os.remove(file_path)
                optimization_stats["removed_files"] += 1
        
        return {
            "success": True,
            "optimization_stats": optimization_stats,
            "message": "Cache optimization completed"
        }
