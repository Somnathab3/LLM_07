#!/usr/bin/env python3
"""
Example usage of the Ollama LLM Client for conflict detection and resolution
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext


def main():
    """Demonstrate LLM Client usage"""
    
    print("🚀 Ollama LLM Client Example")
    print("=" * 50)
    
    # Configure the LLM client (optimized settings)
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",  # Change this to your preferred model
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=300,   # Reduced for faster responses
        timeout=60.0      # Increased timeout
    )
    
    try:
        # Initialize the client
        print("🔧 Initializing LLM client...")
        client = LLMClient(config)
        
        # Test connection
        print("🔌 Testing connection...")
        test_result = client.test_connection()
        
        if not test_result['generation_test']:
            print("❌ Connection failed. Please ensure Ollama is running.")
            print("   Start with: ollama run llama3.1:8b")
            return
        
        print("✅ Connected successfully!")
        
        # Create a realistic conflict scenario
        print("\\n📊 Creating conflict scenario...")
        context = ConflictContext(
            ownship_callsign="AAL123",
            ownship_state={
                'latitude': 40.7128,
                'longitude': -74.0060,
                'altitude': 35000,
                'heading': 90,  # Eastbound
                'speed': 450
            },
            intruders=[
                {
                    'callsign': 'UAL456',
                    'latitude': 40.7150,
                    'longitude': -73.9950,  # Slightly northeast
                    'altitude': 35000,      # Same altitude - potential conflict
                    'heading': 270,         # Westbound - head-on situation
                    'speed': 440,
                    'position': '40.7150°, -73.9950°'
                }
            ],
            scenario_time=0.0,
            lookahead_minutes=5.0,  # Reduced for faster processing
            constraints={}
        )
        
        print(f"   Ownship: {context.ownship_callsign} at FL350, heading {context.ownship_state['heading']}°")
        print(f"   Intruder: UAL456 at FL350, heading 270° (potential head-on conflict)")
        
        # Detect conflicts (using simplified prompts for faster response)
        print("\\n🔍 Detecting conflicts...")
        conflict_result = client.detect_conflicts(context, use_simple_prompt=True)
        
        print(f"   Conflicts detected: {conflict_result.get('conflicts_detected', False)}")
        
        if conflict_result.get('conflicts'):
            for i, conflict in enumerate(conflict_result['conflicts']):
                print(f"   Conflict {i+1}:")
                print(f"     - Aircraft: {conflict.get('intruder_callsign')}")
                print(f"     - Time to conflict: {conflict.get('time_to_conflict_minutes', 'N/A')} minutes")
                print(f"     - Type: {conflict.get('conflict_type', 'N/A')}")
        
        # Generate resolution if conflicts exist
        if conflict_result.get('conflicts_detected'):
            print("\\n⚡ Generating resolution...")
            resolution = client.generate_resolution(context, conflict_result, use_simple_prompt=True)
            
            if resolution.success:
                print(f"   Resolution type: {resolution.resolution_type}")
                print(f"   Parameters: {resolution.parameters}")
                print(f"   Reasoning: {resolution.reasoning}")
                print(f"   Confidence: {resolution.confidence:.2f}")
                
                # Format the resolution as ATC instruction
                if resolution.resolution_type == "heading_change":
                    new_heading = resolution.parameters.get('new_heading_deg')
                    if new_heading:
                        print(f"\\n📢 ATC Instruction:")
                        print(f"   '{context.ownship_callsign}, turn right heading {new_heading:03d}°'")
                
                elif resolution.resolution_type == "altitude_change":
                    new_altitude = resolution.parameters.get('target_altitude_ft')
                    if new_altitude:
                        fl = int(new_altitude / 100)
                        print(f"\\n📢 ATC Instruction:")
                        print(f"   '{context.ownship_callsign}, climb and maintain flight level {fl}'")
                
                elif resolution.resolution_type == "speed_change":
                    new_speed = resolution.parameters.get('target_speed_kt')
                    if new_speed:
                        print(f"\\n📢 ATC Instruction:")
                        print(f"   '{context.ownship_callsign}, reduce speed to {new_speed} knots'")
            else:
                print("   ❌ Failed to generate resolution")
        else:
            print("\\n✅ No conflicts detected - no resolution needed")
        
        print("\\n🎯 Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\nTroubleshooting:")
        print("1. Ensure Ollama is installed and running")
        print("2. Pull the model: ollama pull llama3.1:8b")
        print("3. Start the model: ollama run llama3.1:8b")
        print("4. Check that Ollama is accessible at http://localhost:11434")


if __name__ == "__main__":
    main()
