from functools import wraps
from datetime import datetime
import time
import json
from typing import Callable, Any
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'latency': defaultdict(list),
            'throughput': defaultdict(int),
            'error_rates': defaultdict(int),
            'accuracy': defaultdict(list)
        }
    
    def track(self, operation_name: str):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    latency = time.perf_counter() - start_time
                    
                    # Record latency
                    self.metrics['latency'][operation_name].append(latency)
                    
                    # Record throughput
                    self.metrics['throughput'][operation_name] += 1
                    
                    # If processing documents, record accuracy metrics
                    if operation_name == 'retrieve':
                        expected = kwargs.get('expected_result')
                        if expected and result:
                            accuracy = self.calculate_accuracy(result, expected)
                            self.metrics['accuracy'][operation_name].append(accuracy)
                    
                    return result
                except Exception as e:
                    self.metrics['error_rates'][operation_name] += 1
                    raise
            return wrapper
        return decorator
    
    def calculate_accuracy(self, actual, expected):
        # Simple content overlap accuracy metric
        actual_content = actual.page_content if hasattr(actual, 'page_content') else str(actual)
        expected_content = expected.page_content if hasattr(expected, 'page_content') else str(expected)
        common_tokens = set(actual_content.split()) & set(expected_content.split())
        return len(common_tokens) / max(len(expected_content.split()), 1)
    
    def get_metrics(self):
        return {
            'timestamp': datetime.now().isoformat(),
            **self.metrics
        }

monitor = PerformanceMonitor()
