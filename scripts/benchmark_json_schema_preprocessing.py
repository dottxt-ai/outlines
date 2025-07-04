#!/usr/bin/env python3
"""
Comprehensive benchmarking script for JSON schema preprocessing performance.

This script tests performance under various conditions:
- Different schema sizes and complexities
- Concurrent access patterns
- Cache efficiency
- Memory usage patterns
"""

import json
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import sys
import os

# Add the outlines directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly to avoid dependency issues
exec(open('outlines/types/json_schema_utils.py').read())


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for JSON schema preprocessing."""
    
    def __init__(self):
        self.results = {}
        
    def generate_simple_schema(self, field_count: int = 10) -> Dict[str, Any]:
        """Generate a simple schema with specified number of fields."""
        properties = {}
        for i in range(field_count):
            properties[f"field_{i}"] = {"type": ["string", "null"]}
        
        return {
            "type": "object",
            "properties": properties
        }
    
    def generate_nested_schema(self, depth: int = 5, width: int = 3) -> Dict[str, Any]:
        """Generate a nested schema with specified depth and width."""
        def create_level(current_depth: int) -> Dict[str, Any]:
            if current_depth <= 0:
                return {"type": ["string", "null"]}
            
            properties = {}
            for i in range(width):
                properties[f"level_{current_depth}_field_{i}"] = create_level(current_depth - 1)
            
            return {
                "type": "object",
                "properties": properties
            }
        
        return create_level(depth)
    
    def generate_complex_schema(self, complexity: int = 50) -> Dict[str, Any]:
        """Generate a complex schema with mixed types and constraints."""
        properties = {}
        
        for i in range(complexity):
            if i % 4 == 0:
                # String with constraints
                properties[f"string_field_{i}"] = {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-zA-Z0-9]*$"
                }
            elif i % 4 == 1:
                # Number with constraints
                properties[f"number_field_{i}"] = {
                    "type": ["number", "integer", "null"],
                    "minimum": 0,
                    "maximum": 1000,
                    "multipleOf": 0.1
                }
            elif i % 4 == 2:
                # Array with union item types
                properties[f"array_field_{i}"] = {
                    "type": "array",
                    "items": {"type": ["string", "number", "null"]},
                    "minItems": 0,
                    "maxItems": 10
                }
            else:
                # Nested object
                properties[f"object_field_{i}"] = {
                    "type": ["object", "null"],
                    "properties": {
                        "nested_string": {"type": ["string", "null"]},
                        "nested_number": {"type": ["integer", "null"]}
                    }
                }
        
        return {
            "type": "object",
            "properties": properties,
            "required": [f"string_field_{i}" for i in range(0, complexity, 8)]
        }
    
    def benchmark_schema_sizes(self) -> Dict[str, Any]:
        """Benchmark performance across different schema sizes."""
        print("Benchmarking schema sizes...")
        
        sizes = [1, 5, 10, 25, 50, 100, 200, 500]
        results = {}
        
        # Configure for consistent testing
        configure_preprocessing(enable_metrics=True, enable_compression=True)
        
        for size in sizes:
            print(f"  Testing size {size}...")
            
            # Generate schema
            schema = self.generate_simple_schema(size)
            
            # Clear cache and reset metrics
            clear_schema_cache()
            reset_metrics()
            
            # Time the preprocessing
            times = []
            for _ in range(5):  # Average over 5 runs
                start = time.time()
                preprocess_schema_for_union_types(schema)
                end = time.time()
                times.append(end - start)
            
            # Get final stats
            stats = get_cache_stats()
            
            results[size] = {
                "avg_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times),
                "memory_bytes": stats.get("total_cache_memory_bytes", 0),
                "schema_size_bytes": len(json.dumps(schema).encode('utf-8'))
            }
        
        return results
    
    def benchmark_nested_complexity(self) -> Dict[str, Any]:
        """Benchmark performance with increasing nesting depth."""
        print("Benchmarking nested complexity...")
        
        depths = [1, 3, 5, 8, 10, 15, 20]
        results = {}
        
        for depth in depths:
            print(f"  Testing depth {depth}...")
            
            schema = self.generate_nested_schema(depth, width=2)
            
            clear_schema_cache()
            reset_metrics()
            
            times = []
            for _ in range(3):  # Fewer runs for deep schemas
                start = time.time()
                preprocess_schema_for_union_types(schema)
                end = time.time()
                times.append(end - start)
            
            stats = get_cache_stats()
            
            results[depth] = {
                "avg_time": statistics.mean(times),
                "memory_bytes": stats.get("total_cache_memory_bytes", 0),
                "schema_size_bytes": len(json.dumps(schema).encode('utf-8'))
            }
        
        return results
    
    def benchmark_concurrent_access(self) -> Dict[str, Any]:
        """Benchmark performance under concurrent access."""
        print("Benchmarking concurrent access...")
        
        thread_counts = [1, 2, 4, 8, 16]
        operations_per_thread = 50
        results = {}
        
        # Create a variety of schemas to test with
        schemas = [
            self.generate_simple_schema(10),
            self.generate_simple_schema(25),
            self.generate_nested_schema(3, 3),
            self.generate_complex_schema(20)
        ]
        
        for thread_count in thread_counts:
            print(f"  Testing {thread_count} threads...")
            
            clear_schema_cache()
            reset_metrics()
            
            def worker(thread_id: int) -> Dict[str, Any]:
                """Worker function for concurrent testing."""
                local_times = []
                errors = 0
                
                for i in range(operations_per_thread):
                    schema = schemas[i % len(schemas)]
                    try:
                        start = time.time()
                        preprocess_schema_for_union_types(schema)
                        end = time.time()
                        local_times.append(end - start)
                    except Exception:
                        errors += 1
                
                return {
                    "times": local_times,
                    "errors": errors,
                    "thread_id": thread_id
                }
            
            # Run concurrent benchmark
            start_total = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(worker, i) for i in range(thread_count)]
                thread_results = [f.result() for f in as_completed(futures)]
            
            end_total = time.time()
            
            # Aggregate results
            all_times = []
            total_errors = 0
            
            for thread_result in thread_results:
                all_times.extend(thread_result["times"])
                total_errors += thread_result["errors"]
            
            stats = get_cache_stats()
            
            results[thread_count] = {
                "total_time": end_total - start_total,
                "total_operations": thread_count * operations_per_thread,
                "operations_per_second": (thread_count * operations_per_thread) / (end_total - start_total),
                "avg_operation_time": statistics.mean(all_times) if all_times else 0,
                "total_errors": total_errors,
                "cache_hit_rate": stats.get("cache_hit_rate", 0),
                "final_cache_size": stats.get("cache_size", 0)
            }
        
        return results
    
    def benchmark_cache_efficiency(self) -> Dict[str, Any]:
        """Benchmark cache efficiency with different access patterns."""
        print("Benchmarking cache efficiency...")
        
        # Configure different cache sizes
        cache_sizes = [10, 50, 100, 500]
        results = {}
        
        # Create a set of schemas for testing
        test_schemas = [self.generate_simple_schema(i) for i in range(1, 101)]
        
        for cache_size in cache_sizes:
            print(f"  Testing cache size {cache_size}...")
            
            configure_preprocessing(max_cache_size=cache_size)
            clear_schema_cache()
            reset_metrics()
            
            # Sequential access pattern (should have good cache performance)
            start = time.time()
            for _ in range(3):  # Multiple passes
                for schema in test_schemas[:cache_size//2]:  # Use subset that fits in cache
                    preprocess_schema_for_union_types(schema)
            sequential_time = time.time() - start
            
            sequential_stats = get_cache_stats()
            
            # Random access pattern (should have worse cache performance)
            clear_schema_cache()
            reset_metrics()
            
            import random
            random_schemas = random.choices(test_schemas, k=cache_size * 3)  # More schemas than cache
            
            start = time.time()
            for schema in random_schemas:
                preprocess_schema_for_union_types(schema)
            random_time = time.time() - start
            
            random_stats = get_cache_stats()
            
            results[cache_size] = {
                "sequential": {
                    "time": sequential_time,
                    "hit_rate": sequential_stats.get("cache_hit_rate", 0),
                    "operations": sequential_stats.get("cache_hits", 0) + sequential_stats.get("cache_misses", 0)
                },
                "random": {
                    "time": random_time,
                    "hit_rate": random_stats.get("cache_hit_rate", 0),
                    "operations": random_stats.get("cache_hits", 0) + random_stats.get("cache_misses", 0)
                }
            }
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("Starting comprehensive JSON schema preprocessing benchmark...")
        print("=" * 60)
        
        # Store original configuration
        original_config = get_preprocessing_config()
        
        try:
            results = {
                "schema_sizes": self.benchmark_schema_sizes(),
                "nested_complexity": self.benchmark_nested_complexity(), 
                "concurrent_access": self.benchmark_concurrent_access(),
                "cache_efficiency": self.benchmark_cache_efficiency()
            }
            
            return results
            
        finally:
            # Restore original configuration
            configure_preprocessing(**original_config)
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        # Schema sizes
        print("\n1. SCHEMA SIZE PERFORMANCE")
        print("-" * 30)
        print(f"{'Size':<8} {'Avg Time(ms)':<12} {'Memory(KB)':<12} {'Size(KB)':<10}")
        for size, data in results["schema_sizes"].items():
            avg_ms = data["avg_time"] * 1000
            memory_kb = data["memory_bytes"] / 1024
            size_kb = data["schema_size_bytes"] / 1024
            print(f"{size:<8} {avg_ms:<12.2f} {memory_kb:<12.2f} {size_kb:<10.2f}")
        
        # Concurrent access
        print("\n2. CONCURRENT ACCESS PERFORMANCE")
        print("-" * 35)
        print(f"{'Threads':<8} {'Ops/sec':<10} {'Hit Rate':<10} {'Errors':<8}")
        for threads, data in results["concurrent_access"].items():
            ops_per_sec = data["operations_per_second"]
            hit_rate = data["cache_hit_rate"]
            errors = data["total_errors"]
            print(f"{threads:<8} {ops_per_sec:<10.1f} {hit_rate:<10.2f} {errors:<8}")
        
        # Cache efficiency
        print("\n3. CACHE EFFICIENCY")
        print("-" * 25)
        print(f"{'Cache Size':<12} {'Sequential Hit %':<18} {'Random Hit %':<15}")
        for cache_size, data in results["cache_efficiency"].items():
            seq_hit = data["sequential"]["hit_rate"] * 100
            rand_hit = data["random"]["hit_rate"] * 100
            print(f"{cache_size:<12} {seq_hit:<18.1f} {rand_hit:<15.1f}")


def main():
    """Main function to run the benchmark."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()
    benchmark.print_results(results)
    
    # Save detailed results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to benchmark_results.json")


if __name__ == "__main__":
    main()