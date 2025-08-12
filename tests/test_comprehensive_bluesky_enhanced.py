#!/usr/bin/env python3
"""
Comprehensive BlueSky Client Function Testing

Tests all major BlueSky client functions based on the command reference:
- Aircraft creation and management
- Autopilot and FMS systems  
- Flight commands (ALT, SPD, HDG, VS)
- Throttle management (THR)
- Navigation modes (LNAV, VNAV)
- Simulation control
- Aircraft state monitoring

Reference: BlueSky base commands and command table
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

try:
    from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
except ImportError:
    print("❌ Could not import BlueSky client - check src/cdr/simulation/bluesky_client.py")
    sys.exit(1)


class ComprehensiveBlueSkyTester:
    """Comprehensive tester for all BlueSky client functions"""
    
    def __init__(self):
        self.bs_client = None
        self.test_results = {}
        self.created_aircraft = []
        
    def connect(self):
        """Initialize BlueSky connection"""
        try:
            config = BlueSkyConfig()
            config.headless = True
            config.reso_off = True
            config.dtmult = 1.0
            config.dt = 1.0
            
            self.bs_client = BlueSkyClient(config)
            return self.bs_client.connect()
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def test_simulation_control(self):
        """Test basic simulation control commands"""
        print("\n🎮 Testing Simulation Control Commands...")
        
        tests = {
            "RESET": lambda: self.bs_client.bs.stack.stack("RESET"),
            "OP (Start)": lambda: self.bs_client.bs.stack.stack("OP"),
            "HOLD (Pause)": lambda: self.bs_client.bs.stack.stack("HOLD"),
            "OP (Resume)": lambda: self.bs_client.bs.stack.stack("OP"),
            "DT (Set timestep)": lambda: self.bs_client.bs.stack.stack("DT 1.0"),
            "DTMULT (Set time multiplier)": lambda: self.bs_client.bs.stack.stack("DTMULT 1.0"),
            "NOISE OFF": lambda: self.bs_client.bs.stack.stack("NOISE OFF"),
            "TIME": lambda: self.bs_client.bs.stack.stack("TIME RUN")
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                test_func()
                time.sleep(0.2)  # Brief pause between commands
                results[test_name] = "✅ PASS"
                print(f"   {test_name}: ✅ PASS")
            except Exception as e:
                results[test_name] = f"❌ FAIL: {e}"
                print(f"   {test_name}: ❌ FAIL: {e}")
        
        self.test_results["Simulation Control"] = results
        return results
    
    def test_aircraft_creation(self):
        """Test aircraft creation with different types"""
        print("\n✈️ Testing Aircraft Creation...")
        
        aircraft_configs = [
            {"id": "TEST001", "type": "B737", "lat": 42.0, "lon": -86.0, "hdg": 90, "alt": 35000, "spd": 450},
            {"id": "TEST002", "type": "A320", "lat": 41.5, "lon": -86.5, "hdg": 180, "alt": 36000, "spd": 420},
            {"id": "TEST003", "type": "B777", "lat": 42.5, "lon": -86.2, "hdg": 270, "alt": 34000, "spd": 480},
            {"id": "TEST004", "type": "A330", "lat": 41.8, "lon": -86.8, "hdg": 45, "alt": 37000, "spd": 460}
        ]
        
        results = {}
        created = []
        
        for config in aircraft_configs:
            try:
                # Use CRE command for aircraft creation
                cmd = f"CRE {config['id']},{config['type']},{config['lat']},{config['lon']},{config['hdg']},{config['alt']},{config['spd']}"
                self.bs_client.bs.stack.stack(cmd)
                time.sleep(0.5)  # Allow creation to complete
                
                # Verify aircraft exists by checking state
                state = self.bs_client.get_aircraft_state(config['id'])
                if state:
                    created.append(config['id'])
                    results[config['id']] = "✅ CREATED"
                    print(f"   {config['id']} ({config['type']}): ✅ CREATED")
                else:
                    results[config['id']] = "❌ CREATION FAILED"
                    print(f"   {config['id']} ({config['type']}): ❌ CREATION FAILED")
                    
            except Exception as e:
                results[config['id']] = f"❌ ERROR: {e}"
                print(f"   {config['id']} ({config['type']}): ❌ ERROR: {e}")
        
        self.created_aircraft = created
        self.test_results["Aircraft Creation"] = results
        print(f"   📊 Summary: {len(created)}/{len(aircraft_configs)} aircraft created successfully")
        return results
    
    def test_autopilot_systems(self):
        """Test autopilot and FMS system commands"""
        print("\n🤖 Testing Autopilot and FMS Systems...")
        
        if not self.created_aircraft:
            print("   ⚠️ No aircraft available for autopilot testing")
            return {}
        
        # Test autopilot commands on first aircraft
        test_aircraft = self.created_aircraft[0]
        results = {}
        
        autopilot_tests = {
            "THR AUTO (Autothrottle)": lambda: self.bs_client.bs.stack.stack(f"THR {test_aircraft},AUTO"),
            "LNAV ON (Lateral Navigation)": lambda: self.bs_client.bs.stack.stack(f"LNAV {test_aircraft},ON"),
            "VNAV ON (Vertical Navigation)": lambda: self.bs_client.bs.stack.stack(f"VNAV {test_aircraft},ON"),
            "ALT Command": lambda: self.bs_client.bs.stack.stack(f"ALT {test_aircraft},35000"),
            "SPD Command": lambda: self.bs_client.bs.stack.stack(f"SPD {test_aircraft},450"),
            "HDG Command": lambda: self.bs_client.bs.stack.stack(f"HDG {test_aircraft},90"),
            "VS Command": lambda: self.bs_client.bs.stack.stack(f"VS {test_aircraft},0"),
            "LNAV OFF": lambda: self.bs_client.bs.stack.stack(f"LNAV {test_aircraft},OFF"),
            "VNAV OFF": lambda: self.bs_client.bs.stack.stack(f"VNAV {test_aircraft},OFF"),
            "THR IDLE": lambda: self.bs_client.bs.stack.stack(f"THR {test_aircraft},IDLE")
        }
        
        for test_name, test_func in autopilot_tests.items():
            try:
                test_func()
                time.sleep(0.3)  # Brief pause for command processing
                results[test_name] = "✅ COMMAND ACCEPTED"
                print(f"   {test_name}: ✅ COMMAND ACCEPTED")
            except Exception as e:
                results[test_name] = f"❌ COMMAND FAILED: {e}"
                print(f"   {test_name}: ❌ COMMAND FAILED: {e}")
        
        self.test_results["Autopilot Systems"] = results
        return results
    
    def test_flight_commands(self):
        """Test flight control commands on all aircraft"""
        print("\n🎯 Testing Flight Control Commands...")
        
        if not self.created_aircraft:
            print("   ⚠️ No aircraft available for flight command testing")
            return {}
        
        results = {}
        
        # Test basic flight commands on each aircraft
        for i, aircraft_id in enumerate(self.created_aircraft):
            print(f"   Testing commands on {aircraft_id}...")
            aircraft_results = {}
            
            # Test altitude commands
            test_alt = 35000 + (i * 1000)  # Different altitudes for each aircraft
            try:
                self.bs_client.bs.stack.stack(f"ALT {aircraft_id},{test_alt}")
                aircraft_results["ALT Command"] = "✅ SENT"
            except Exception as e:
                aircraft_results["ALT Command"] = f"❌ FAILED: {e}"
            
            # Test speed commands
            test_spd = 450 + (i * 20)  # Different speeds for each aircraft
            try:
                self.bs_client.bs.stack.stack(f"SPD {aircraft_id},{test_spd}")
                aircraft_results["SPD Command"] = "✅ SENT"
            except Exception as e:
                aircraft_results["SPD Command"] = f"❌ FAILED: {e}"
            
            # Test heading commands
            test_hdg = 90 + (i * 30)  # Different headings for each aircraft
            try:
                self.bs_client.bs.stack.stack(f"HDG {aircraft_id},{test_hdg}")
                aircraft_results["HDG Command"] = "✅ SENT"
            except Exception as e:
                aircraft_results["HDG Command"] = f"❌ FAILED: {e}"
            
            # Test vertical speed commands
            try:
                self.bs_client.bs.stack.stack(f"VS {aircraft_id},500")  # 500 fpm climb
                time.sleep(2)
                self.bs_client.bs.stack.stack(f"VS {aircraft_id},0")    # Level off
                aircraft_results["VS Command"] = "✅ SENT"
            except Exception as e:
                aircraft_results["VS Command"] = f"❌ FAILED: {e}"
            
            results[aircraft_id] = aircraft_results
            
            # Brief pause between aircraft
            time.sleep(0.5)
        
        self.test_results["Flight Commands"] = results
        return results
    
    def test_aircraft_state_monitoring(self):
        """Test aircraft state retrieval and monitoring"""
        print("\n📊 Testing Aircraft State Monitoring...")
        
        if not self.created_aircraft:
            print("   ⚠️ No aircraft available for state monitoring")
            return {}
        
        results = {}
        
        # Test state retrieval for each aircraft
        for aircraft_id in self.created_aircraft:
            try:
                state = self.bs_client.get_aircraft_state(aircraft_id)
                
                if state:
                    # Check that we got reasonable values
                    lat = state.get('lat', 0)
                    lon = state.get('lon', 0)
                    alt = state.get('alt', 0)
                    hdg = state.get('hdg', 0)
                    spd = state.get('tas', 0)  # True airspeed
                    vs = state.get('vs', 0)    # Vertical speed
                    
                    if lat != 0 and lon != 0 and alt > 1000:  # Reasonable values check
                        results[aircraft_id] = {
                            "Status": "✅ STATE RETRIEVED",
                            "Position": f"({lat:.4f}, {lon:.4f})",
                            "Altitude": f"{alt:.0f}ft",
                            "Heading": f"{hdg:.1f}°",
                            "Speed": f"{spd:.1f}kt",
                            "VS": f"{vs:.1f}fpm"
                        }
                        print(f"   {aircraft_id}: ✅ {lat:.4f},{lon:.4f} | {alt:.0f}ft | {hdg:.1f}° | {spd:.1f}kt")
                    else:
                        results[aircraft_id] = "⚠️ INVALID STATE DATA"
                        print(f"   {aircraft_id}: ⚠️ INVALID STATE DATA")
                else:
                    results[aircraft_id] = "❌ NO STATE DATA"
                    print(f"   {aircraft_id}: ❌ NO STATE DATA")
                    
            except Exception as e:
                results[aircraft_id] = f"❌ ERROR: {e}"
                print(f"   {aircraft_id}: ❌ ERROR: {e}")
        
        self.test_results["State Monitoring"] = results
        return results
    
    def test_advanced_commands(self):
        """Test advanced BlueSky commands"""
        print("\n🔧 Testing Advanced Commands...")
        
        advanced_tests = {
            "POS Command": lambda: self.bs_client.bs.stack.stack(f"POS {self.created_aircraft[0] if self.created_aircraft else 'TEST001'}"),
            "DIST Calculation": lambda: self.bs_client.bs.stack.stack("DIST 42.0,-86.0,41.0,-87.0"),
            "WIND Definition": lambda: self.bs_client.bs.stack.stack("WIND 42.0,-86.0,35000,270,50"),
            "SEED Set": lambda: self.bs_client.bs.stack.stack("SEED 12345"),
            "CALC Command": lambda: self.bs_client.bs.stack.stack("CALC 2+2*3"),
            "ECHO Command": lambda: self.bs_client.bs.stack.stack("ECHO Test message from comprehensive test")
        }
        
        results = {}
        for test_name, test_func in advanced_tests.items():
            try:
                test_func()
                time.sleep(0.2)
                results[test_name] = "✅ COMMAND SENT"
                print(f"   {test_name}: ✅ COMMAND SENT")
            except Exception as e:
                results[test_name] = f"❌ FAILED: {e}"
                print(f"   {test_name}: ❌ FAILED: {e}")
        
        self.test_results["Advanced Commands"] = results
        return results
    
    def test_aircraft_deletion(self):
        """Test aircraft deletion"""
        print("\n🗑️ Testing Aircraft Deletion...")
        
        if not self.created_aircraft:
            print("   ⚠️ No aircraft available for deletion testing")
            return {}
        
        results = {}
        
        # Delete one aircraft for testing
        if len(self.created_aircraft) > 1:
            test_aircraft = self.created_aircraft[-1]  # Delete the last one
            
            try:
                self.bs_client.bs.stack.stack(f"DEL {test_aircraft}")
                time.sleep(1)
                
                # Check if aircraft is gone
                state = self.bs_client.get_aircraft_state(test_aircraft)
                if not state:
                    results[test_aircraft] = "✅ SUCCESSFULLY DELETED"
                    print(f"   {test_aircraft}: ✅ SUCCESSFULLY DELETED")
                    self.created_aircraft.remove(test_aircraft)
                else:
                    results[test_aircraft] = "⚠️ STILL EXISTS AFTER DELETE"
                    print(f"   {test_aircraft}: ⚠️ STILL EXISTS AFTER DELETE")
                    
            except Exception as e:
                results[test_aircraft] = f"❌ DELETE FAILED: {e}"
                print(f"   {test_aircraft}: ❌ DELETE FAILED: {e}")
        else:
            results["Skip"] = "⚠️ Insufficient aircraft for deletion test"
            print("   ⚠️ Insufficient aircraft for deletion test")
        
        self.test_results["Aircraft Deletion"] = results
        return results
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 COMPREHENSIVE BLUESKY CLIENT FUNCTION TEST")
        print("=" * 60)
        print("Testing all major BlueSky client functions based on command reference")
        print()
        
        if not self.connect():
            return False
        
        # Run all test categories
        test_categories = [
            self.test_simulation_control,
            self.test_aircraft_creation,
            self.test_autopilot_systems,
            self.test_flight_commands,
            self.test_aircraft_state_monitoring,
            self.test_advanced_commands,
            self.test_aircraft_deletion
        ]
        
        for test_category in test_categories:
            try:
                test_category()
                time.sleep(1)  # Brief pause between test categories
            except Exception as e:
                print(f"❌ Test category failed: {e}")
        
        # Print summary
        self.print_test_summary()
        
        # Clean up
        try:
            self.bs_client.disconnect()
        except:
            pass
        
        return True
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        total_passed = 0
        
        for category, tests in self.test_results.items():
            print(f"\n🔍 {category}:")
            
            if isinstance(tests, dict):
                category_passed = 0
                category_total = len(tests)
                
                for test_name, result in tests.items():
                    if isinstance(result, dict):
                        # Handle nested results (like aircraft state monitoring)
                        status = result.get("Status", str(result))
                        extra = " | ".join([f"{k}: {v}" for k, v in result.items() if k != "Status"])
                        print(f"   {test_name}: {status}")
                        if extra:
                            print(f"      └─ {extra}")
                        if "✅" in str(status):
                            category_passed += 1
                    else:
                        print(f"   {test_name}: {result}")
                        if "✅" in str(result):
                            category_passed += 1
                
                total_tests += category_total
                total_passed += category_passed
                
                # Category summary
                pass_rate = (category_passed / category_total * 100) if category_total > 0 else 0
                print(f"   📈 Category Score: {category_passed}/{category_total} ({pass_rate:.1f}%)")
        
        # Overall summary
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 OVERALL TEST RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_tests - total_passed}")
        print(f"   Success Rate: {overall_pass_rate:.1f}%")
        
        if overall_pass_rate >= 90:
            print(f"   🎉 EXCELLENT: BlueSky client functions working very well!")
        elif overall_pass_rate >= 75:
            print(f"   ✅ GOOD: Most BlueSky client functions working properly")
        elif overall_pass_rate >= 50:
            print(f"   ⚠️ FAIR: Some BlueSky client functions need attention")
        else:
            print(f"   ❌ POOR: Significant issues with BlueSky client functions")


def main():
    """Main test execution"""
    print("🚁 BlueSky Comprehensive Client Function Test")
    print("Based on BlueSky command reference and base commands")
    print()
    
    tester = ComprehensiveBlueSkyTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 Comprehensive test completed!")
    else:
        print("\n💥 Comprehensive test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
