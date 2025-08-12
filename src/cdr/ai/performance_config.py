"""
Performance optimization configuration for LLM_ATC7
Includes settings for FlashAttention, CUDA Graphs, and Ollama optimizations
"""

import os
import logging


class PerformanceConfig:
    """Centralized performance optimization settings"""
    
    # FlashAttention 2 settings
    ENABLE_FLASH_ATTENTION = True
    FLASH_ATTENTION_BACKEND = "flash_attention_2"
    
    # CUDA optimization settings
    ENABLE_CUDA_GRAPHS = True
    ENABLE_TORCH_COMPILE = True
    TORCH_COMPILE_MODE = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"
    ENABLE_CUDNN_BENCHMARK = True
    ENABLE_TF32 = True
    
    # Ollama performance settings
    OLLAMA_OPTIMIZATIONS = {
        "numa": False,
        "num_ctx": 4096,
        "num_thread": -1,
        "num_gpu": 1,
        "gpu_layers": -1,
        "use_mlock": True,
        "use_mmap": True,
        "low_vram": False,
        "f16_kv": True,
        "vocab_only": False,
        "logits_all": False,
        "embedding": False,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "penalize_newline": False
    }
    
    # Request optimization settings
    CONNECTION_POOL_SIZE = 10
    KEEP_ALIVE_TIMEOUT = "5m"
    REQUEST_TIMEOUT_WARMUP = 90
    REQUEST_TIMEOUT_NORMAL = 45
    
    # Memory optimization settings
    EMBEDDING_BATCH_SIZE = 32
    FAISS_OMP_NUM_THREADS = 4
    
    @classmethod
    def apply_environment_optimizations(cls):
        """Apply environment-level performance optimizations"""
        logger = logging.getLogger(__name__)
        
        # Disable tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # OpenMP settings for FAISS
        os.environ["OMP_NUM_THREADS"] = str(cls.FAISS_OMP_NUM_THREADS)
        
        # CUDA optimizations
        if cls.ENABLE_TF32:
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        
        # Memory allocation optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        logger.info("✅ Applied environment-level performance optimizations")
    
    @classmethod
    def get_ollama_options(cls):
        """Get optimized Ollama options"""
        return cls.OLLAMA_OPTIMIZATIONS.copy()
    
    @classmethod
    def get_torch_compile_config(cls):
        """Get torch.compile configuration"""
        return {
            "mode": cls.TORCH_COMPILE_MODE,
            "fullgraph": True,
            "dynamic": False
        }


def apply_global_optimizations():
    """Apply all global performance optimizations"""
    PerformanceConfig.apply_environment_optimizations()


# Auto-apply optimizations when module is imported
apply_global_optimizations()
