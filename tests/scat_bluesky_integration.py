#!/usr/bin/env python3
"""Integration example: SCAT data replay in BlueSky simulator"""

import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.adapters.scat_adapter import SCATAdapter
from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
from src.cdr.schemas.scat_schemas import TrackPoint


class SCATReplayIntegration:
    """Integration class for replaying SCAT data in BlueSky"""
    
    def __init__(self, bluesky_config: BlueSkyConfig):
        self.bluesky_client = BlueSkyClient(bluesky_config)
        self.scat_adapters: Dict[str, SCATAdapter] = {}
        self.replay_speed = 1.0
        self.replay_active = False
    
    def load_scat_file(self, scat_file: Path) -> bool:
        """Load a SCAT file for replay"""
        try:
            adapter = SCATAdapter(scat_file)
            adapter.load_file()
            
            callsign = adapter.get_callsign()
            self.scat_adapters[callsign] = adapter
            
            print(f"✅ Loaded SCAT data for {callsign}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SCAT file {scat_file}: {e}")
            return False
    
    def load_scat_directory(self, scat_dir: Path, max_files: int = 10) -> int:
        """Load multiple SCAT files from directory"""
        loaded_count = 0
        
        for scat_file in scat_dir.glob("*.json"):
            if loaded_count >= max_files:
                break
                
            if self.load_scat_file(scat_file):
                loaded_count += 1
        
        print(f"✅ Loaded {loaded_count} SCAT files")
        return loaded_count
    
    def setup_simulation(self) -> bool:
        """Setup BlueSky simulation for SCAT replay"""
        try:
            print("🔧 Setting up BlueSky simulation...")
            
            if not self.bluesky_client.connect(timeout=45):
                print("❌ Failed to connect to BlueSky")
                return False
            
            # Reset simulation
            self.bluesky_client.reset_simulation()
            
            # Configure for replay
            self.bluesky_client.set_fast_time_factor(self.replay_speed)
            
            # Create aircraft in BlueSky for each SCAT track
            for callsign, adapter in self.scat_adapters.items():
                track_points = list(adapter.ownship_track())
                if track_points:
                    first_point = track_points[0]
                    
                    # Create aircraft at starting position
                    success = self.bluesky_client.create_aircraft(
                        callsign=callsign,
                        aircraft_type=adapter.get_aircraft_type() or "B737",
                        lat=first_point.latitude,
                        lon=first_point.longitude,
                        heading=first_point.heading_deg,
                        altitude_ft=first_point.altitude_ft,
                        speed_kt=first_point.speed_kt
                    )
                    
                    if success:
                        print(f"✅ Created aircraft {callsign} in BlueSky")
                    else:
                        print(f"⚠️ Failed to create aircraft {callsign}")
            
            print("✅ BlueSky simulation setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def replay_scat_data(self, duration_seconds: float = 300) -> bool:
        """Replay SCAT data in BlueSky simulation"""
        try:
            print(f"▶️ Starting SCAT data replay for {duration_seconds}s...")
            self.replay_active = True
            
            # Get all track points organized by time
            all_tracks = {}
            min_time = float('inf')
            max_time = 0
            
            for callsign, adapter in self.scat_adapters.items():
                track_points = list(adapter.ownship_track())
                if track_points:
                    all_tracks[callsign] = track_points
                    times = [p.timestamp for p in track_points]
                    min_time = min(min_time, min(times))
                    max_time = max(max_time, max(times))
            
            if not all_tracks:
                print("❌ No track data to replay")
                return False
            
            print(f"📊 Replay data span: {max_time - min_time:.0f} seconds")
            print(f"   Aircraft: {list(all_tracks.keys())}")
            
            # Start simulation
            self.bluesky_client.continue_simulation()
            
            # Replay loop
            start_time = time.time()
            replay_start_time = min_time
            last_update_time = {}
            
            while (time.time() - start_time) < duration_seconds and self.replay_active:
                current_real_time = time.time()
                elapsed_real_time = current_real_time - start_time
                current_sim_time = replay_start_time + (elapsed_real_time * self.replay_speed)
                
                # Update aircraft positions based on SCAT data
                for callsign, track_points in all_tracks.items():
                    # Find the appropriate track point for current time
                    target_point = self._find_track_point_at_time(track_points, current_sim_time)
                    
                    if target_point and callsign not in last_update_time or \
                       (current_real_time - last_update_time.get(callsign, 0)) > 5.0:
                        
                        # Update aircraft position in BlueSky
                        self.bluesky_client.move_aircraft(
                            callsign=callsign,
                            lat=target_point.latitude,
                            lon=target_point.longitude,
                            altitude_ft=target_point.altitude_ft,
                            heading_deg=target_point.heading_deg,
                            speed_kt=target_point.speed_kt
                        )
                        
                        last_update_time[callsign] = current_real_time
                
                # Check for conflicts
                if int(elapsed_real_time) % 10 == 0:  # Every 10 seconds
                    conflicts = self.bluesky_client.get_conflicts()
                    if conflicts:
                        print(f"⚠️ Detected {len(conflicts)} conflicts at t={elapsed_real_time:.0f}s")
                        for conflict in conflicts:
                            print(f"   {conflict.aircraft1} vs {conflict.aircraft2}: "
                                  f"{conflict.horizontal_distance:.1f}NM")
                
                # Progress update
                if int(elapsed_real_time) % 30 == 0 and elapsed_real_time > 0:
                    print(f"📈 Replay progress: {elapsed_real_time:.0f}s / {duration_seconds:.0f}s")
                
                time.sleep(1.0)  # Update every second
            
            self.bluesky_client.hold_simulation()
            print("✅ SCAT replay completed")
            return True
            
        except Exception as e:
            print(f"❌ Replay failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.replay_active = False
    
    def _find_track_point_at_time(self, track_points: List[TrackPoint], 
                                  target_time: float) -> TrackPoint:
        """Find the track point closest to the target time"""
        if not track_points:
            return None
        
        # Binary search for efficiency with large datasets
        left, right = 0, len(track_points) - 1
        best_point = track_points[0]
        min_diff = abs(track_points[0].timestamp - target_time)
        
        while left <= right:
            mid = (left + right) // 2
            point = track_points[mid]
            diff = abs(point.timestamp - target_time)
            
            if diff < min_diff:
                min_diff = diff
                best_point = point
            
            if point.timestamp < target_time:
                left = mid + 1
            else:
                right = mid - 1
        
        return best_point
    
    def analyze_conflicts(self) -> Dict[str, any]:
        """Analyze conflicts in the loaded SCAT data"""
        print("🔍 Analyzing conflicts in SCAT data...")
        
        conflict_analysis = {
            "total_conflicts": 0,
            "conflict_pairs": [],
            "critical_conflicts": 0,
            "time_periods": []
        }
        
        # Get current aircraft states
        states = self.bluesky_client.get_aircraft_states()
        
        # Perform conflict analysis
        conflicts = self.bluesky_client.get_conflicts()
        conflict_analysis["total_conflicts"] = len(conflicts)
        
        for conflict in conflicts:
            conflict_analysis["conflict_pairs"].append({
                "aircraft1": conflict.aircraft1,
                "aircraft2": conflict.aircraft2,
                "horizontal_distance": conflict.horizontal_distance,
                "vertical_distance": conflict.vertical_distance,
                "severity": conflict.severity
            })
            
            if conflict.severity == "high":
                conflict_analysis["critical_conflicts"] += 1
        
        print(f"📊 Analysis complete: {len(conflicts)} conflicts found")
        print(f"   Critical conflicts: {conflict_analysis['critical_conflicts']}")
        
        return conflict_analysis
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.replay_active = False
        
        # Delete all aircraft
        states = self.bluesky_client.get_aircraft_states()
        for callsign in states.keys():
            self.bluesky_client.delete_aircraft(callsign)
        
        self.bluesky_client.disconnect()
        print("✅ Cleanup complete")


def main():
    """Main integration demo"""
    print("🚀 SCAT-BlueSky Integration Demo")
    print("=" * 50)
    
    # Configuration
    config = BlueSkyConfig(
        headless=True,
        fast_time_factor=5.0,  # 5x speed for demo
        conflict_detection=True,
        lookahead_time=180.0,  # 3 minutes
        protected_zone_radius=5.0,
        protected_zone_height=1000.0
    )
    
    # Create integration instance
    integration = SCATReplayIntegration(config)
    
    try:
        # Load SCAT data
        print("\n1️⃣ Loading SCAT data...")
        
        sample_file = Path("data/sample_scat.json")
        if sample_file.exists():
            integration.load_scat_file(sample_file)
        else:
            print("❌ Sample SCAT file not found")
            return
        
        # Setup simulation
        print("\n2️⃣ Setting up simulation...")
        if not integration.setup_simulation():
            return
        
        # Run replay
        print("\n3️⃣ Running replay...")
        integration.replay_scat_data(duration_seconds=120)  # 2 minutes
        
        # Analyze conflicts
        print("\n4️⃣ Analyzing conflicts...")
        analysis = integration.analyze_conflicts()
        
        print(f"\n📊 Final Analysis:")
        print(f"   Total conflicts: {analysis['total_conflicts']}")
        print(f"   Critical conflicts: {analysis['critical_conflicts']}")
        
        # Show conflict details
        for i, conflict in enumerate(analysis['conflict_pairs'][:5]):  # Show first 5
            print(f"   Conflict {i+1}: {conflict['aircraft1']} vs {conflict['aircraft2']}")
            print(f"     Distance: {conflict['horizontal_distance']:.1f}NM, "
                  f"{conflict['vertical_distance']:.0f}ft")
        
        print("\n✅ Integration demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        integration.cleanup()


if __name__ == "__main__":
    main()
