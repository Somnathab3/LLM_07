"""Advanced Experience Memory System for Air Traffic Control

This module implements a comprehensive memory system that stores and retrieves
past conflict resolution experiences to improve decision-making quality.
"""

import numpy as np
import logging
import warnings

# Configure Faiss to use CPU only and suppress GPU warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from Faiss

# Suppress specific Faiss GPU warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*GPU.*")
    warnings.filterwarnings("ignore", message=".*GpuIndexIVFFlat.*")
    import faiss

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from builtins import open  # Explicitly import open function

# Suppress GPU warnings for SentenceTransformers as well
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*GPU.*")
    from sentence_transformers import SentenceTransformer

from .llm_client import ConflictContext


@dataclass
class MemoryRecord:
    """A single experience record in the memory system"""
    record_id: str
    timestamp: datetime
    conflict_context: Dict[str, Any]  # Serialized ConflictContext
    resolution_taken: Dict[str, Any]
    outcome_metrics: Dict[str, Any]  # Success rate, safety metrics, etc.
    scenario_features: Dict[str, Any]  # Geometric and contextual features
    task_type: str  # 'conflict_detection', 'resolution', 'verification'
    success_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any]  # Additional context


@dataclass
class ConflictGeometry:
    """Geometric features of a conflict scenario"""
    approach_angle: float  # Relative approach angle in degrees
    speed_ratio: float  # Intruder speed / ownship speed
    altitude_separation: float  # Current altitude separation in ft
    horizontal_separation: float  # Current horizontal separation in nm
    time_to_cpa: float  # Time to closest point of approach in minutes
    conflict_type: str  # 'head_on', 'crossing', 'overtaking', 'vertical'
    urgency_level: str  # 'low', 'medium', 'high', 'critical'


class ExperienceMemory:
    """
    Advanced experience memory system using FAISS for efficient similarity search.
    
    Features:
    - Vector embeddings of conflict scenarios
    - Temporal and spatial feature encoding
    - Success-weighted similarity search
    - Efficient batch updates and indexing
    - Geometric conflict pattern recognition
    """
    
    def __init__(self, 
                 memory_dir: Path = Path("data/memory"),
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_type: str = "Flat",  # Changed default to Flat for better reliability
                 embedding_dim: int = 384,
                 max_records: int = 10000,
                 device: str = "cpu"):  # Force CPU to avoid GPU contention
        """
        Initialize the experience memory system.
        
        Args:
            memory_dir: Directory to store memory files
            embedding_model: Sentence transformer model name
            index_type: FAISS index type ('Flat', 'IVFFlat', 'HNSW')
            embedding_dim: Dimension of embeddings
            max_records: Maximum number of records to store
            device: Device for embeddings ('cpu' or 'cuda')
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure memory subdirectories exist
        (self.memory_dir / "backups").mkdir(exist_ok=True)
        (self.memory_dir / "exports").mkdir(exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.device = device  # Store device preference
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.max_records = max_records
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_faiss_index()
        self._load_existing_records()
        
        # Performance tracking
        self.query_count = 0
        self.hit_count = 0
        
        # Suppress Faiss GPU warnings after initialization
        self._suppress_faiss_warnings()
    
    def _suppress_faiss_warnings(self):
        """Suppress Faiss GPU-related warnings"""
        # Configure logging to suppress Faiss GPU warnings
        faiss_logger = logging.getLogger('faiss')
        faiss_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
        
        # Also suppress warnings from the faiss loader
        faiss_loader_logger = logging.getLogger('faiss.loader')
        faiss_loader_logger.setLevel(logging.ERROR)
        
        self.logger.debug("Suppressed Faiss GPU warnings")
        
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name} on {self.device}")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model {self.embedding_model_name}: {e}")
            # Fallback to simple feature vectors
            self.embedding_model = None
            self.embedding_dim = 64  # Reduced dimension for manual features
            
    def _initialize_faiss_index(self):
        """Initialize FAISS index for similarity search (CPU-only)"""
        try:
            # Explicitly use CPU-only Faiss indexes to avoid GPU warnings
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == "IVFFlat":
                # IVF with 100 clusters (CPU only)
                nlist = min(100, max(10, self.max_records // 100))
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            elif self.index_type == "HNSW":
                # Hierarchical Navigable Small World index (CPU only)
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Ensure index stays on CPU
            if hasattr(self.index, 'set_direct_map_type'):
                # Some indexes support direct mapping for better performance
                self.index.set_direct_map_type(faiss.DirectMap.Hashtable)
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize {self.index_type} index: {e}")
            # Fallback to simple flat index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index_type = "Flat"
            
        # Initialize ID mapping
        self.record_ids: List[str] = []
        self.records: Dict[str, MemoryRecord] = {}
        
        self.logger.info(f"Initialized FAISS index: {self.index_type} (CPU-only)")
        
    def _load_existing_records(self):
        """Load existing memory records from disk"""
        records_file = self.memory_dir / "records.json"
        index_file = self.memory_dir / "faiss_index.idx"
        ids_file = self.memory_dir / "record_ids.pkl"
        
        if records_file.exists() and index_file.exists() and ids_file.exists():
            try:
                # Load records
                with open(str(records_file), 'r', encoding='utf-8') as f:
                    records_data = json.load(f)
                
                self.records = {}
                for record_data in records_data:
                    try:
                        # Convert timestamp back to datetime
                        record_data['timestamp'] = datetime.fromisoformat(record_data['timestamp'])
                        record = MemoryRecord(**record_data)
                        self.records[record.record_id] = record
                    except Exception as e:
                        self.logger.warning(f"Failed to load record: {e}")
                        continue
                
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load record IDs
                with open(str(ids_file), 'rb') as f:
                    self.record_ids = pickle.load(f)
                
                self.logger.info(f"Loaded {len(self.records)} existing memory records")
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing records: {e}")
                import traceback
                self.logger.warning(f"Traceback: {traceback.format_exc()}")
                self.records = {}
                self.record_ids = []
        else:
            self.logger.info("No existing memory files found, starting fresh")
            self.records = {}
            self.record_ids = []
                
    def _save_memory_state(self):
        """Save memory state to disk"""
        try:
            records_file = self.memory_dir / "records.json"
            index_file = self.memory_dir / "faiss_index.idx"
            ids_file = self.memory_dir / "record_ids.pkl"
            
            # Ensure directory exists
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Save records (convert datetime to string for JSON)
            records_data = []
            for record in self.records.values():
                try:
                    record_dict = asdict(record)
                    record_dict['timestamp'] = record.timestamp.isoformat()
                    records_data.append(record_dict)
                except Exception as e:
                    self.logger.warning(f"Failed to serialize record {record.record_id}: {e}")
                    continue
            
            # Save records to JSON with explicit file handling
            try:
                with open(str(records_file), 'w', encoding='utf-8') as f:
                    json.dump(records_data, f, indent=2)
                self.logger.debug(f"Saved {len(records_data)} records to {records_file}")
            except Exception as e:
                self.logger.error(f"Failed to save records to {records_file}: {e}")
                raise
            
            # Save FAISS index
            try:
                faiss.write_index(self.index, str(index_file))
                self.logger.debug(f"Saved FAISS index to {index_file}")
            except Exception as e:
                self.logger.error(f"Failed to save FAISS index to {index_file}: {e}")
                raise
            
            # Save record IDs
            try:
                with open(str(ids_file), 'wb') as f:
                    pickle.dump(self.record_ids, f)
                self.logger.debug(f"Saved record IDs to {ids_file}")
            except Exception as e:
                self.logger.error(f"Failed to save record IDs to {ids_file}: {e}")
                raise
                
            self.logger.info(f"Saved memory state with {len(self.records)} records")
            
        except Exception as e:
            self.logger.error(f"Failed to save memory state: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _extract_geometric_features(self, context: ConflictContext) -> ConflictGeometry:
        """Extract geometric features from conflict context"""
        ownship = context.ownship_state
        
        if not context.intruders:
            # No intruders - return default geometry
            return ConflictGeometry(
                approach_angle=0.0,
                speed_ratio=1.0,
                altitude_separation=10000.0,
                horizontal_separation=50.0,
                time_to_cpa=30.0,
                conflict_type='none',
                urgency_level='low'
            )
        
        # Use the first (closest) intruder for geometry calculation
        intruder = context.intruders[0]
        
        # Calculate approach angle
        own_hdg = ownship.get('heading', 0)
        int_hdg = intruder.get('heading', 0)
        approach_angle = abs(own_hdg - int_hdg)
        if approach_angle > 180:
            approach_angle = 360 - approach_angle
            
        # Speed ratio
        own_speed = ownship.get('speed', 400)
        int_speed = intruder.get('speed', 400)
        speed_ratio = int_speed / max(own_speed, 1)
        
        # Separations
        altitude_separation = abs(
            ownship.get('altitude', 35000) - intruder.get('altitude', 35000)
        )
        
        # Simplified horizontal separation (would need lat/lon calculation in real system)
        horizontal_separation = 10.0  # Placeholder
        
        # Time to CPA (simplified)
        time_to_cpa = context.lookahead_minutes / 2
        
        # Conflict type based on approach angle
        if approach_angle < 30:
            conflict_type = 'head_on'
        elif approach_angle > 150:
            conflict_type = 'overtaking'
        elif 60 <= approach_angle <= 120:
            conflict_type = 'crossing'
        else:
            conflict_type = 'vertical' if altitude_separation < 2000 else 'crossing'
            
        # Urgency level
        if time_to_cpa < 2:
            urgency_level = 'critical'
        elif time_to_cpa < 5:
            urgency_level = 'high'
        elif time_to_cpa < 10:
            urgency_level = 'medium'
        else:
            urgency_level = 'low'
            
        return ConflictGeometry(
            approach_angle=approach_angle,
            speed_ratio=speed_ratio,
            altitude_separation=altitude_separation,
            horizontal_separation=horizontal_separation,
            time_to_cpa=time_to_cpa,
            conflict_type=conflict_type,
            urgency_level=urgency_level
        )
    
    def _generate_context_embedding(self, context: Dict[str, Any], task_type: str) -> np.ndarray:
        """
        Generate context embedding using sentence transformers or manual features.
        
        Args:
            context: ConflictContext dictionary
            task_type: Type of task ('conflict_detection', 'resolution', 'verification')
            
        Returns:
            Normalized embedding vector
        """
        if self.embedding_model is not None:
            # Use sentence transformer for rich semantic embeddings
            text_description = self._context_to_text(context, task_type)
            embedding = self.embedding_model.encode([text_description])
            return embedding[0].astype(np.float32)
        else:
            # Fallback to manual feature engineering
            return self._generate_manual_features(context, task_type)
    
    def _context_to_text(self, context: Dict[str, Any], task_type: str) -> str:
        """Convert conflict context to text description for embedding"""
        ownship = context.get('ownship_state', {})
        intruders = context.get('intruders', [])
        
        # Extract key features
        altitude = ownship.get('altitude', 0) / 100  # Convert to flight level
        heading = ownship.get('heading', 0)
        speed = ownship.get('speed', 0)
        
        # Build text description
        text_parts = [
            f"Task: {task_type}",
            f"Aircraft at FL{int(altitude)} heading {heading} degrees at {speed} knots",
        ]
        
        if intruders:
            text_parts.append(f"Conflicts with {len(intruders)} aircraft:")
            for i, intruder in enumerate(intruders[:3]):  # Limit to 3 for text length
                int_alt = intruder.get('altitude', 0) / 100
                int_hdg = intruder.get('heading', 0)
                int_spd = intruder.get('speed', 0)
                text_parts.append(
                    f"  Intruder {i+1}: FL{int(int_alt)} heading {int_hdg} at {int_spd} knots"
                )
        
        return " ".join(text_parts)
    
    def _generate_manual_features(self, context: Dict[str, Any], task_type: str) -> np.ndarray:
        """Generate manual feature vector when sentence transformers unavailable"""
        features = []
        
        # Task type encoding (one-hot)
        task_types = ['conflict_detection', 'resolution', 'verification']
        task_encoding = [1.0 if task_type == t else 0.0 for t in task_types]
        features.extend(task_encoding)
        
        # Ownship features
        ownship = context.get('ownship_state', {})
        features.extend([
            ownship.get('altitude', 35000) / 50000,  # Normalized altitude
            ownship.get('heading', 0) / 360,  # Normalized heading
            ownship.get('speed', 400) / 600,  # Normalized speed
            ownship.get('latitude', 0) / 90,  # Normalized latitude
            ownship.get('longitude', 0) / 180,  # Normalized longitude
        ])
        
        # Intruder features (up to 3 intruders)
        intruders = context.get('intruders', [])
        for i in range(3):
            if i < len(intruders):
                intruder = intruders[i]
                features.extend([
                    intruder.get('altitude', 35000) / 50000,
                    intruder.get('heading', 0) / 360,
                    intruder.get('speed', 400) / 600,
                    1.0,  # Intruder present
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # No intruder
        
        # Scenario features
        features.extend([
            context.get('lookahead_minutes', 10) / 30,  # Normalized lookahead
            context.get('scenario_time', 0) / 3600,  # Normalized time
            len(intruders) / 10,  # Normalized intruder count
        ])
        
        # Geometric features
        try:
            conflict_context = ConflictContext(**context)
            geometry = self._extract_geometric_features(conflict_context)
            features.extend([
                geometry.approach_angle / 180,
                geometry.speed_ratio,
                geometry.altitude_separation / 10000,
                geometry.horizontal_separation / 50,
                geometry.time_to_cpa / 30,
            ])
        except:
            # Fallback if context conversion fails
            features.extend([0.0, 1.0, 1.0, 1.0, 0.5])
        
        # Pad or truncate to match embedding dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        features = features[:self.embedding_dim]
        
        return np.array(features, dtype=np.float32)
    
    def store_experience(self, record: MemoryRecord):
        """
        Store experience record with efficient indexing and duplicate detection.
        
        Args:
            record: MemoryRecord to store
        """
        try:
            # Check for duplicates using content hash
            record_hash = self._compute_record_hash(record)
            
            # Check if similar record already exists
            existing_record = self._find_duplicate_record(record, record_hash)
            if existing_record:
                # Merge with existing record
                self._merge_records(existing_record, record)
                self.logger.debug(f"Merged experience with existing record {existing_record.record_id}")
                return
            
            # Generate embedding
            embedding = self._generate_context_embedding(
                record.conflict_context, record.task_type
            )
            
            # Add to FAISS index
            if len(self.record_ids) >= self.max_records:
                # Remove oldest record to make space
                self._remove_oldest_record()
            
            # Train index if needed (for IVF indices)
            if (self.index_type == "IVFFlat" and 
                not self.index.is_trained and 
                len(self.record_ids) >= 10):  # Reduced minimum training size
                # Collect training data
                training_embeddings = []
                for r in list(self.records.values())[:100]:
                    train_emb = self._generate_context_embedding(r.conflict_context, r.task_type)
                    training_embeddings.append(train_emb)
                
                if training_embeddings:
                    training_data = np.array(training_embeddings)
                    self.index.train(training_data)
                    self.logger.info("Trained FAISS IVF index")
                else:
                    # Fallback to Flat index if training fails
                    self.logger.warning("Switching to Flat index due to training issues")
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    self.index_type = "Flat"
            
            # Add to index (only if trained for IVF indices)
            if (self.index_type != "IVFFlat" or self.index.is_trained):
                self.index.add(embedding.reshape(1, -1))
                self.record_ids.append(record.record_id)
                self.records[record.record_id] = record
            else:
                # Store record but not in index yet (will be added when trained)
                self.records[record.record_id] = record
                self.record_ids.append(record.record_id)
                self.logger.debug(f"Stored record {record.record_id} (index not trained yet)")
            
            # Periodic save
            if len(self.records) % 100 == 0:
                self._save_memory_state()
                
            self.logger.debug(f"Stored experience record {record.record_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store experience record: {e}")
    
    def _compute_record_hash(self, record: MemoryRecord) -> str:
        """Compute hash of record content for duplicate detection"""
        # Create hash from key scenario features
        content = {
            'task_type': record.task_type,
            'ownship_altitude': record.conflict_context.get('ownship_state', {}).get('altitude'),
            'ownship_heading': record.conflict_context.get('ownship_state', {}).get('heading'),
            'intruder_count': len(record.conflict_context.get('intruders', [])),
            'resolution_type': record.resolution_taken.get('resolution_type'),
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _find_duplicate_record(self, record: MemoryRecord, record_hash: str) -> Optional[MemoryRecord]:
        """Find existing record that might be a duplicate"""
        # Simple duplicate detection - could be enhanced
        for existing_record in self.records.values():
            if (existing_record.task_type == record.task_type and
                abs((existing_record.timestamp - record.timestamp).total_seconds()) < 300):  # 5 minutes
                return existing_record
        return None
    
    def _merge_records(self, existing: MemoryRecord, new: MemoryRecord):
        """Merge new record with existing record"""
        # Update success score with weighted average
        weight = 0.3  # Weight for new experience
        existing.success_score = (
            (1 - weight) * existing.success_score + 
            weight * new.success_score
        )
        
        # Update outcome metrics
        for key, value in new.outcome_metrics.items():
            if key in existing.outcome_metrics:
                existing.outcome_metrics[key] = (
                    (1 - weight) * existing.outcome_metrics[key] + 
                    weight * value
                )
            else:
                existing.outcome_metrics[key] = value
        
        # Update timestamp to most recent
        existing.timestamp = max(existing.timestamp, new.timestamp)
    
    def _remove_oldest_record(self):
        """Remove oldest record to make space for new ones"""
        if not self.records:
            return
            
        # Find oldest record
        oldest_id = min(self.records.keys(), 
                       key=lambda x: self.records[x].timestamp)
        
        # Remove from data structures
        del self.records[oldest_id]
        
        # Remove from FAISS index (rebuild index)
        old_index = self.record_ids.index(oldest_id)
        self.record_ids.remove(oldest_id)
        
        # Rebuild FAISS index without the removed record
        self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index after record removal"""
        if not self.records:
            return
            
        # Reinitialize index
        self._initialize_faiss_index()
        
        # Re-add all embeddings
        embeddings = []
        for record_id in self.record_ids:
            record = self.records[record_id]
            embedding = self._generate_context_embedding(
                record.conflict_context, record.task_type
            )
            embeddings.append(embedding)
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            
            # Train if needed
            if (self.index_type == "IVFFlat" and 
                not self.index.is_trained and 
                len(embeddings) >= 100):
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
    
    def query_similar(self, context: ConflictContext, top_k: int = 5) -> List[MemoryRecord]:
        """
        Query similar experiences with sophisticated ranking.
        
        Args:
            context: Current conflict context
            top_k: Number of similar records to return
            
        Returns:
            List of similar MemoryRecord objects, ranked by relevance
        """
        self.query_count += 1
        
        if not self.records:
            return []
        
        try:
            # Convert context to dictionary for embedding
            context_dict = {
                'ownship_callsign': context.ownship_callsign,
                'ownship_state': context.ownship_state,
                'intruders': context.intruders,
                'scenario_time': context.scenario_time,
                'lookahead_minutes': context.lookahead_minutes,
                'constraints': context.constraints,
            }
            
            # Generate query embedding
            query_embedding = self._generate_context_embedding(
                context_dict, 'resolution'  # Default task type for queries
            )
            
            # Search FAISS index
            search_k = min(top_k * 3, len(self.records))  # Get more candidates for reranking
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1), search_k
            )
            
            # Retrieve candidate records
            candidates = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.record_ids):
                    record_id = self.record_ids[idx]
                    if record_id in self.records:
                        record = self.records[record_id]
                        distance = distances[0][i]
                        candidates.append((record, distance))
            
            # Advanced reranking with multiple factors
            ranked_records = self._rerank_candidates(candidates, context)
            
            # Update hit count if we found relevant records
            if ranked_records:
                self.hit_count += 1
            
            return ranked_records[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to query similar experiences: {e}")
            return []
    
    def _rerank_candidates(self, candidates: List[Tuple[MemoryRecord, float]], 
                          query_context: ConflictContext) -> List[MemoryRecord]:
        """
        Advanced reranking of candidate records using multiple factors.
        
        Args:
            candidates: List of (record, distance) tuples from FAISS search
            query_context: Current conflict context
            
        Returns:
            Reranked list of MemoryRecord objects
        """
        if not candidates:
            return []
        
        scored_candidates = []
        current_time = datetime.now()
        
        for record, faiss_distance in candidates:
            # 1. Similarity score (inverse of FAISS distance)
            similarity_score = 1.0 / (1.0 + faiss_distance)
            
            # 2. Temporal relevance (more recent = better)
            time_diff = (current_time - record.timestamp).total_seconds()
            temporal_score = np.exp(-time_diff / (30 * 24 * 3600))  # 30-day decay
            
            # 3. Success rate weighting
            success_score = record.success_score
            
            # 4. Context similarity (geometric features)
            context_score = self._compute_context_similarity(record, query_context)
            
            # 5. Task relevance
            task_score = 1.0 if record.task_type == 'resolution' else 0.7
            
            # Combined score with weights
            combined_score = (
                0.4 * similarity_score +
                0.2 * temporal_score +
                0.2 * success_score +
                0.15 * context_score +
                0.05 * task_score
            )
            
            scored_candidates.append((record, combined_score))
        
        # Sort by combined score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [record for record, score in scored_candidates]
    
    def _compute_context_similarity(self, record: MemoryRecord, 
                                   query_context: ConflictContext) -> float:
        """Compute similarity between stored record and query context"""
        try:
            # Extract geometric features for both contexts
            stored_context = ConflictContext(**record.conflict_context)
            stored_geometry = self._extract_geometric_features(stored_context)
            query_geometry = self._extract_geometric_features(query_context)
            
            # Compare geometric features
            angle_similarity = 1 - abs(stored_geometry.approach_angle - query_geometry.approach_angle) / 180
            speed_similarity = 1 - abs(stored_geometry.speed_ratio - query_geometry.speed_ratio)
            altitude_similarity = 1 - abs(stored_geometry.altitude_separation - query_geometry.altitude_separation) / 10000
            
            # Type similarity
            type_similarity = 1.0 if stored_geometry.conflict_type == query_geometry.conflict_type else 0.5
            
            # Urgency similarity
            urgency_levels = ['low', 'medium', 'high', 'critical']
            stored_urgency_idx = urgency_levels.index(stored_geometry.urgency_level)
            query_urgency_idx = urgency_levels.index(query_geometry.urgency_level)
            urgency_similarity = 1 - abs(stored_urgency_idx - query_urgency_idx) / 3
            
            # Weighted average
            context_similarity = (
                0.3 * angle_similarity +
                0.2 * speed_similarity +
                0.2 * altitude_similarity +
                0.2 * type_similarity +
                0.1 * urgency_similarity
            )
            
            return max(0.0, min(1.0, context_similarity))
            
        except Exception:
            # Fallback to basic similarity
            return 0.5
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        hit_rate = self.hit_count / max(self.query_count, 1)
        
        # Calculate success rate distribution
        success_scores = [r.success_score for r in self.records.values()]
        avg_success = np.mean(success_scores) if success_scores else 0.0
        
        # Task type distribution
        task_counts = {}
        for record in self.records.values():
            task_type = record.task_type
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        return {
            'total_records': len(self.records),
            'query_count': self.query_count,
            'hit_count': self.hit_count,
            'hit_rate': hit_rate,
            'average_success_score': avg_success,
            'task_distribution': task_counts,
            'embedding_model': self.embedding_model_name,
            'index_type': self.index_type,
            'index_trained': getattr(self.index, 'is_trained', True),
        }
    
    def cleanup_old_records(self, max_age_days: int = 90):
        """Remove records older than specified age"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        old_records = [
            record_id for record_id, record in self.records.items()
            if record.timestamp < cutoff_time
        ]
        
        for record_id in old_records:
            del self.records[record_id]
            if record_id in self.record_ids:
                self.record_ids.remove(record_id)
        
        if old_records:
            self._rebuild_faiss_index()
            self.logger.info(f"Cleaned up {len(old_records)} old records")
    
    def export_experiences(self, output_path: Path, format: str = 'json'):
        """Export experiences for analysis or backup"""
        try:
            if format == 'json':
                export_data = []
                for record in self.records.values():
                    try:
                        record_dict = asdict(record)
                        record_dict['timestamp'] = record.timestamp.isoformat()
                        export_data.append(record_dict)
                    except Exception as e:
                        self.logger.warning(f"Failed to export record {record.record_id}: {e}")
                        continue
                
                with open(str(output_path), 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(self.records)} records to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export experiences: {e}")
            raise
    
    def __del__(self):
        """Cleanup and save state on destruction"""
        try:
            self._save_memory_state()
        except Exception:
            pass  # Ignore errors during cleanup


def create_memory_record(conflict_context: ConflictContext,
                        resolution: Dict[str, Any],
                        outcome_metrics: Dict[str, Any],
                        task_type: str = 'resolution') -> MemoryRecord:
    """
    Helper function to create a MemoryRecord from conflict resolution data.
    
    Args:
        conflict_context: The conflict context
        resolution: Resolution action taken
        outcome_metrics: Metrics about the outcome
        task_type: Type of task performed
        
    Returns:
        MemoryRecord instance
    """
    # Generate unique record ID
    timestamp = datetime.now()
    record_id = f"{task_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(conflict_context.ownship_callsign)) % 10000:04d}"
    
    # Convert ConflictContext to dictionary
    context_dict = {
        'ownship_callsign': conflict_context.ownship_callsign,
        'ownship_state': conflict_context.ownship_state,
        'intruders': conflict_context.intruders,
        'scenario_time': conflict_context.scenario_time,
        'lookahead_minutes': conflict_context.lookahead_minutes,
        'constraints': conflict_context.constraints,
        'nearby_traffic': conflict_context.nearby_traffic,
    }
    
    # Extract scenario features
    memory = ExperienceMemory()  # Temporary instance for feature extraction
    geometry = memory._extract_geometric_features(conflict_context)
    scenario_features = asdict(geometry)
    
    # Calculate success score
    success_score = outcome_metrics.get('success_rate', 0.5)
    if 'safety_margin' in outcome_metrics:
        safety_bonus = min(0.3, outcome_metrics['safety_margin'] / 10)  # Up to 0.3 bonus
        success_score = min(1.0, success_score + safety_bonus)
    
    return MemoryRecord(
        record_id=record_id,
        timestamp=timestamp,
        conflict_context=context_dict,
        resolution_taken=resolution,
        outcome_metrics=outcome_metrics,
        scenario_features=scenario_features,
        task_type=task_type,
        success_score=success_score,
        metadata={
            'version': '1.0',
            'created_by': 'LLM_ATC7'
        }
    )
