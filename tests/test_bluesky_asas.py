#!/usr/bin/env python3
"""
Test BlueSky's Internal ASAS (Automatic Separation Assurance System)
"""

import sys
sys.path.append('.')

import bluesky as bs
from src.cdr.simulation.bluesky_client import BlueSkyClient


def test_bluesky_asas():
    """Test BlueSky's internal ASAS system"""
    print("🛡️  Testing BlueSky Internal ASAS System")
    print("=" * 50)
    
    # Initialize BlueSky
    client = BlueSkyClient()
    client.initialize()
    client.reset()
    
    # Check ASAS availability and configuration
    print("\n📊 ASAS System Information:")
    try:
        if hasattr(bs.traf, 'asas'):
            asas = bs.traf.asas
            print(f"   ASAS Object: {type(asas)}")
            
            # Check various ASAS attributes
            attributes = ['dtlook', 'dtconf', 'rpz', 'active', 'enabled']
            for attr in attributes:
                if hasattr(asas, attr):
                    value = getattr(asas, attr)
                    print(f"   {attr}: {value}")
            
            # Try to activate ASAS
            print(f"\n🔧 Attempting to configure ASAS...")
            
            # Enable ASAS if possible
            if hasattr(asas, 'setseparation'):
                print("   Found setseparation method")
            if hasattr(asas, 'setasas'):
                print("   Found setasas method")
                
            # Check command stack for ASAS commands
            print(f"\n📝 Available ASAS-related commands:")
            if hasattr(bs.stack, 'commands'):
                asas_commands = [cmd for cmd in bs.stack.commands.keys() if 'asas' in cmd.lower()]
                print(f"   ASAS commands: {asas_commands}")
            
            # Try some ASAS commands
            print(f"\n🎯 Testing ASAS commands...")
            
            # Enable ASAS
            result = bs.stack.stack("ASAS ON")
            print(f"   ASAS ON result: {result}")
            
            # Check if conflicts are being detected after enabling
            print(f"\n✈️  Creating test aircraft...")
            
            # Create two aircraft on collision course
            client.create_aircraft("TEST1", lat=42.0, lon=-87.0, hdg=90, alt=35000, spd=400)
            client.create_aircraft("TEST2", lat=42.0, lon=-86.5, hdg=270, alt=35000, spd=400)
            
            # Step simulation and check for internal conflicts
            for i in range(60):  # 1 minute
                client.step_simulation(1.0)
                
                if hasattr(asas, 'confpairs') and asas.confpairs is not None:
                    if len(asas.confpairs) > 0:
                        print(f"   🚨 BlueSky ASAS detected conflicts at {i}s: {len(asas.confpairs)}")
                        break
                elif hasattr(asas, 'conflicts') and asas.conflicts is not None:
                    if len(asas.conflicts) > 0:
                        print(f"   🚨 BlueSky ASAS detected conflicts at {i}s: {len(asas.conflicts)}")
                        break
                        
                if i % 10 == 0:
                    print(f"   {i}s: No internal conflicts detected yet")
            
        else:
            print("   ❌ ASAS not available in bs.traf")
            
    except Exception as e:
        print(f"   ❌ Error checking ASAS: {e}")
    
    # Check other conflict detection systems
    print(f"\n🔍 Other Conflict Detection Systems:")
    try:
        # Check for other conflict detection attributes
        conflict_attrs = []
        for attr in dir(bs.traf):
            if 'conf' in attr.lower() or 'asas' in attr.lower():
                conflict_attrs.append(attr)
        
        print(f"   Conflict-related attributes: {conflict_attrs}")
        
        for attr in conflict_attrs:
            try:
                value = getattr(bs.traf, attr)
                print(f"   {attr}: {type(value)} = {value}")
            except:
                print(f"   {attr}: (error accessing)")
                
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    test_bluesky_asas()
