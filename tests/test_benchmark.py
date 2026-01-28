"""
Benchmark tests for polars-sgt
"""

import polars as pl
import pytest
from polars_sgt import sgt_transform

@pytest.fixture
def small_dataset():
    """Small dataset: 100 sequences, avg length 10"""
    n_sequences = 100
    avg_length = 10
    return pl.DataFrame({
        "seq_id": [i // avg_length for i in range(n_sequences * avg_length)],
        "state": [f"state_{i % 5}" for i in range(n_sequences * avg_length)],
        "time": [float(i) for i in range(n_sequences * avg_length)],
    })


@pytest.fixture
def medium_dataset():
    """Medium dataset: 1000 sequences, avg length 20"""
    n_sequences = 1000
    avg_length = 20
    return pl.DataFrame({
        "seq_id": [i // avg_length for i in range(n_sequences * avg_length)],
        "state": [f"state_{i % 10}" for i in range(n_sequences * avg_length)],
        "time": [float(i) for i in range(n_sequences * avg_length)],
    })


@pytest.fixture
def large_dataset():
    """Large dataset: 10000 sequences, avg length 50"""
    n_sequences = 10000
    avg_length = 50
    return pl.DataFrame({
        "seq_id": [i // avg_length for i in range(n_sequences * avg_length)],
        "state": [f"state_{i % 20}" for i in range(n_sequences * avg_length)],
        "time": [float(i) for i in range(n_sequences * avg_length)],
    })


class TestPerformance:
    """Performance benchmarks"""
    
    def test_unigrams_small(self, benchmark, small_dataset):
        """Benchmark unigrams on small dataset"""
        def run():
            return small_dataset.select(
                sgt_transform("seq_id", "state", kappa=1)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 100
    
    def test_bigrams_small(self, benchmark, small_dataset):
        """Benchmark bigrams on small dataset"""
        def run():
            return small_dataset.select(
                sgt_transform("seq_id", "state", kappa=2)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 100
    
    def test_trigrams_small(self, benchmark, small_dataset):
        """Benchmark trigrams on small dataset"""
        def run():
            return small_dataset.select(
                sgt_transform("seq_id", "state", kappa=3)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 100
    
    def test_with_time_small(self, benchmark, small_dataset):
        """Benchmark with time decay on small dataset"""
        def run():
            return small_dataset.select(
                sgt_transform(
                    "seq_id", "state",
                    time_col="time",
                    kappa=2,
                    time_penalty="exponential",
                )
            )
        
        result = benchmark(run)
        assert result.shape[0] == 100
    
    def test_unigrams_medium(self, benchmark, medium_dataset):
        """Benchmark unigrams on medium dataset"""
        def run():
            return medium_dataset.select(
                sgt_transform("seq_id", "state", kappa=1)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 1000
    
    def test_bigrams_medium(self, benchmark, medium_dataset):
        """Benchmark bigrams on medium dataset"""
        def run():
            return medium_dataset.select(
                sgt_transform("seq_id", "state", kappa=2)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 1000
    
    def test_lazy_execution(self, benchmark, medium_dataset):
        """Benchmark lazy execution"""
        def run():
            return (
                medium_dataset.lazy()
                .select(sgt_transform("seq_id", "state", kappa=2))
                .collect()
            )
        
        result = benchmark(run)
        assert result.shape[0] == 1000


class TestScalability:
    """Scalability tests"""
    
    def test_large_dataset(self, large_dataset):
        """Test on large dataset"""
        result = large_dataset.select(
            sgt_transform("seq_id", "state", kappa=2)
        )
        
        assert result.shape[0] == 10000
    
    def test_high_kappa(self, medium_dataset):
        """Test with high kappa value"""
        result = medium_dataset.select(
            sgt_transform("seq_id", "state", kappa=5)
        )
        
        assert result.shape[0] == 1000
    
    def test_streaming(self, large_dataset):
        """Test streaming execution"""
        # Note: Streaming is more about memory efficiency than speed
        result = (
            large_dataset.lazy()
            .select(sgt_transform("seq_id", "state", kappa=2))
            .collect(streaming=True)
        )
        
        assert result.shape[0] == 10000


class TestMemoryEfficiency:
    """Memory efficiency tests"""
    
    def test_memory_small(self, benchmark, small_dataset):
        """Monitor memory usage on small dataset"""
        def run():
            return small_dataset.select(
                sgt_transform("seq_id", "state", kappa=3)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 100
    
    def test_memory_medium(self, benchmark, medium_dataset):
        """Monitor memory usage on medium dataset"""
        def run():
            return medium_dataset.select(
                sgt_transform("seq_id", "state", kappa=3)
            )
        
        result = benchmark(run)
        assert result.shape[0] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])