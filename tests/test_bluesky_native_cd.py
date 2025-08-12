#!/usr/bin/env python3
"""
Test BlueSky's Built-in ConflictDetection System

Based on the detection.py file, this tests access to BlueSky's native
conflict detection system: bluesky.traffic.asas.detection.ConflictDetection
"""

import sys
sys.path.append('.')

import bluesky as bs
from src.cdr.simulation.bluesky_client import SimpleBlueSkyClient


def test_bluesky_conflict_detection_access():
    """Test access to BlueSky's built-in ConflictDetection system"""
    print("🔍 Testing BlueSky Built-in ConflictDetection System")
    print("=" * 60)
    
    # Initialize BlueSky
    client = SimpleBlueSkyClient()
    client.initialize()
    client.reset()
    
    print("✅ BlueSky initialized")
    
    # Test 1: Access ConflictDetection through traffic
    print("\n📊 Checking ConflictDetection access...")
    try:
        # Check if conflict detection is available in traffic
        if hasattr(bs.traf, 'cd'):
            cd = bs.traf.cd
            print(f"✅ ConflictDetection available: {type(cd)}")
            print(f"   Class: {cd.__class__.__name__}")
            print(f"   Module: {cd.__class__.__module__}")
            
            # Check ConflictDetection attributes from detection.py
            cd_attrs = [
                'confpairs', 'lospairs', 'qdr', 'dist', 'dcpa', 'tcpa', 'tLOS',
                'confpairs_unique', 'lospairs_unique', 'confpairs_all', 'lospairs_all',
                'inconf', 'tcpamax', 'rpz', 'hpz', 'dtlookahead', 'dtnolook'
            ]
            
            print(f"\n📋 ConflictDetection attributes:")
            for attr in cd_attrs:
                if hasattr(cd, attr):
                    value = getattr(cd, attr)
                    print(f"   ✅ {attr}: {type(value)} = {value}")
                else:
                    print(f"   ❌ {attr}: Not available")
                    
        else:
            print("❌ ConflictDetection not available in bs.traf.cd")
            
    except Exception as e:
        print(f"❌ Error accessing ConflictDetection: {e}")
    
    # Test 2: Check ASAS settings and commands
    print("\n🛡️  Testing ASAS Configuration...")
    try:
        # Test ASAS commands mentioned in detection.py
        asas_commands = ['CDMETHOD', 'ASAS', 'ZONER', 'PZR', 'ZONEDH', 'DTLOOK', 'DTNOLOOK']
        
        for cmd in asas_commands:
            try:
                # Test if command exists
                result = bs.stack.stack(f"{cmd}")
                print(f"   ✅ {cmd}: Available")
            except Exception as e:
                print(f"   ❌ {cmd}: {e}")
                
    except Exception as e:
        print(f"❌ Error testing ASAS commands: {e}")
    
    # Test 3: Enable conflict detection and test
    print("\n🎯 Testing Conflict Detection Activation...")
    try:
        # Try to enable conflict detection
        print("   Attempting to enable CD with CDMETHOD ON...")
        result = bs.stack.stack("CDMETHOD ON")
        print(f"   CDMETHOD ON result: {result}")
        
        # Check current CD method
        result = bs.stack.stack("CDMETHOD")
        print(f"   Current CD method: {result}")
        
        # Set protection zone parameters
        print("   Setting protection zone parameters...")
        bs.stack.stack("ZONER 5.0")  # 5 NM horizontal
        bs.stack.stack("ZONEDH 1000")  # 1000 ft vertical
        bs.stack.stack("DTLOOK 300")  # 5 minute lookahead
        
    except Exception as e:
        print(f"❌ Error activating conflict detection: {e}")
    
    return cd if 'cd' in locals() else None


def test_conflict_detection_with_aircraft(cd):
    """Test conflict detection with actual aircraft"""
    print("\n✈️  Testing ConflictDetection with Aircraft...")
    
    client = SimpleBlueSkyClient()
    
    # Create aircraft on collision course
    print("   Creating aircraft on collision course...")
    client.create_aircraft("CONF1", lat=42.0, lon=-87.0, hdg=90, alt=35000, spd=400)
    client.create_aircraft("CONF2", lat=42.0, lon=-86.8, hdg=270, alt=35000, spd=400)
    
    print(f"   Created 2 aircraft")
    
    # Monitor for conflicts using built-in system
    print("   Monitoring ConflictDetection for 60 seconds...")
    
    for i in range(60):
        client.step_simulation(1.0)
        
        try:
            # Check conflict detection results
            if cd:
                # Check various conflict indicators
                if hasattr(cd, 'confpairs') and cd.confpairs:
                    print(f"   🚨 Conflicts detected at {i}s: {cd.confpairs}")
                    
                if hasattr(cd, 'lospairs') and cd.lospairs:
                    print(f"   🚨 LoS detected at {i}s: {cd.lospairs}")
                    
                if hasattr(cd, 'inconf') and len(cd.inconf) > 0 and any(cd.inconf):
                    print(f"   🚨 Aircraft in conflict at {i}s: {cd.inconf}")
                    
                if hasattr(cd, 'confpairs_unique') and cd.confpairs_unique:
                    print(f"   🚨 Unique conflicts at {i}s: {cd.confpairs_unique}")
                    
        except Exception as e:
            if i == 0:  # Only print error once
                print(f"   ❌ Error checking conflicts: {e}")
        
        # Progress indicator
        if i % 10 == 0:
            print(f"   {i}s: Monitoring...")
    
    # Final conflict summary
    try:
        if cd:
            print(f"\n📊 Final ConflictDetection Status:")
            print(f"   Total unique conflicts: {len(cd.confpairs_unique) if hasattr(cd, 'confpairs_unique') else 'N/A'}")
            print(f"   Total unique LoS: {len(cd.lospairs_unique) if hasattr(cd, 'lospairs_unique') else 'N/A'}")
            print(f"   All conflicts since start: {len(cd.confpairs_all) if hasattr(cd, 'confpairs_all') else 'N/A'}")
            print(f"   All LoS since start: {len(cd.lospairs_all) if hasattr(cd, 'lospairs_all') else 'N/A'}")
    except Exception as e:
        print(f"   ❌ Error getting final status: {e}")


def enhance_bluesky_client():
    """Enhance the BlueSky client with native conflict detection"""
    print("\n🔧 Enhancing BlueSky Client with Native Conflict Detection...")
    
    enhancement_code = '''
def get_bluesky_conflicts(self):
    """Get conflicts from BlueSky's native ConflictDetection system"""
    try:
        if hasattr(bs.traf, 'cd'):
            cd = bs.traf.cd
            return {
                'confpairs': getattr(cd, 'confpairs', []),
                'lospairs': getattr(cd, 'lospairs', []),
                'confpairs_unique': list(getattr(cd, 'confpairs_unique', set())),
                'lospairs_unique': list(getattr(cd, 'lospairs_unique', set())),
                'inconf': getattr(cd, 'inconf', np.array([])),
                'qdr': getattr(cd, 'qdr', np.array([])),
                'dist': getattr(cd, 'dist', np.array([])),
                'dcpa': getattr(cd, 'dcpa', np.array([])),
                'tcpa': getattr(cd, 'tcpa', np.array([]))
            }
    except Exception as e:
        print(f"Error accessing BlueSky conflicts: {e}")
    return None

def configure_conflict_detection(self, pz_radius_nm=5.0, pz_height_ft=1000, lookahead_sec=300):
    """Configure BlueSky's native conflict detection parameters"""
    try:
        bs.stack.stack("CDMETHOD ON")
        bs.stack.stack(f"ZONER {pz_radius_nm}")
        bs.stack.stack(f"ZONEDH {pz_height_ft}")
        bs.stack.stack(f"DTLOOK {lookahead_sec}")
        return True
    except Exception as e:
        print(f"Error configuring conflict detection: {e}")
        return False
'''
    
    print("   Suggested methods to add to SimpleBlueSkyClient:")
    print(enhancement_code)


if __name__ == "__main__":
    print("🛡️  BlueSky Native ConflictDetection Integration Test")
    print("=" * 70)
    
    # Test access to conflict detection
    cd = test_bluesky_conflict_detection_access()
    
    # Test with actual aircraft
    if cd:
        test_conflict_detection_with_aircraft(cd)
    else:
        print("\n❌ Cannot test with aircraft - ConflictDetection not accessible")
    
    # Show enhancement suggestions
    enhance_bluesky_client()
    
    print("\n✅ ConflictDetection integration test completed")
