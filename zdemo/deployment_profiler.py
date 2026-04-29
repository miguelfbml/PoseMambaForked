"""
Real-time 3D Pose Estimation - Deployment Feasibility Profiler

This module provides comprehensive latency profiling and bottleneck analysis
for the 3D pose estimation pipeline. Use this to measure deployment feasibility.

Usage:
    from deployment_profiler import LatencyTracker, PipelineAnalyzer, DeploymentChecker
    
    tracker = LatencyTracker(window_size=30)
    analyzer = PipelineAnalyzer()
    checker = DeploymentChecker()
    
    # In your main loop:
    frame_id = frame_count
    tracker.start_frame(frame_id)
    
    # Mark each pipeline stage
    tracker.mark_event(frame_id, 'yolo_start')
    # ... YOLO inference ...
    tracker.mark_event(frame_id, 'yolo_end')
    tracker.mark_event(frame_id, 'buffer_full')
    tracker.mark_event(frame_id, 'posemamba_start')
    # ... PoseMamba inference ...
    tracker.mark_event(frame_id, 'posemamba_end')
    tracker.mark_event(frame_id, 'viz_start')
    # ... Visualization ...
    tracker.mark_event(frame_id, 'viz_end')
    
    tracker.end_frame(frame_id)
    
    # Get stats periodically
    if frame_count % 30 == 0:
        stats = tracker.get_stats()
        analyzer.analyze(stats)
        checker.check_feasibility(stats)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class LatencyStats:
    """Container for latency statistics."""
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    std_ms: float
    sample_count: int


class LatencyTracker:
    """
    Tracks end-to-end latency and per-stage timing.
    
    Each frame is tracked from capture to display. Intermediate events
    (YOLO start/end, buffer full, PoseMamba start/end, etc.) are recorded.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to keep in rolling window for statistics.
        """
        self.window_size = window_size
        self.stages: Dict[int, Dict] = {}
        self.latencies = deque(maxlen=window_size)  # End-to-end latencies
        self.stage_timings = {}  # Per-stage timing breakdowns
    
    def start_frame(self, frame_id: int) -> None:
        """Mark the start of frame processing (typically camera capture time)."""
        self.stages[frame_id] = {
            'start_time': time.time(),
            'events': {},
        }
    
    def mark_event(self, frame_id: int, event_name: str) -> None:
        """
        Mark a named event in the pipeline for this frame.
        
        Common events:
        - 'yolo_start': Before YOLO inference
        - 'yolo_end': After YOLO inference
        - 'buffer_full': When buffer reaches n_frames
        - 'posemamba_start': Before 3D prediction
        - 'posemamba_end': After 3D prediction
        - 'viz_start': Before visualization
        - 'viz_end': After visualization (display updated)
        """
        if frame_id in self.stages:
            self.stages[frame_id]['events'][event_name] = time.time()
    
    def end_frame(self, frame_id: int) -> Optional[float]:
        """
        Mark the end of frame processing and compute latency.
        
        Returns:
            End-to-end latency in seconds, or None if frame_id not found.
        """
        if frame_id not in self.stages:
            return None
        
        start_time = self.stages[frame_id]['start_time']
        end_time = time.time()
        latency = end_time - start_time
        
        self.latencies.append(latency)
        
        # Clean up old stages to avoid memory bloat
        del self.stages[frame_id]
        
        return latency
    
    def get_stage_timing(self, frame_id: int, start_event: str, end_event: str) -> Optional[float]:
        """
        Get timing between two events for a frame.
        
        Args:
            frame_id: Frame identifier
            start_event: Name of start event
            end_event: Name of end event
            
        Returns:
            Time elapsed in seconds between events, or None if not found.
        """
        if frame_id not in self.stages:
            return None
        
        events = self.stages[frame_id]['events']
        if start_event not in events or end_event not in events:
            return None
        
        return events[end_event] - events[start_event]
    
    def get_stats(self) -> LatencyStats:
        """
        Compute latency statistics over the current window.
        
        Returns:
            LatencyStats with percentiles and summary statistics.
        """
        if not self.latencies:
            return LatencyStats(
                mean_ms=0, p50_ms=0, p95_ms=0, p99_ms=0, max_ms=0, min_ms=0, std_ms=0, sample_count=0
            )
        
        latencies_ms = np.array(self.latencies) * 1000
        
        return LatencyStats(
            mean_ms=float(np.mean(latencies_ms)),
            p50_ms=float(np.percentile(latencies_ms, 50)),
            p95_ms=float(np.percentile(latencies_ms, 95)),
            p99_ms=float(np.percentile(latencies_ms, 99)),
            max_ms=float(np.max(latencies_ms)),
            min_ms=float(np.min(latencies_ms)),
            std_ms=float(np.std(latencies_ms)),
            sample_count=len(self.latencies),
        )
    
    def get_all_latencies_ms(self) -> List[float]:
        """Get all latencies in the current window as milliseconds."""
        return [lat * 1000 for lat in self.latencies]
    
    def reset(self) -> None:
        """Clear all tracked data."""
        self.latencies.clear()
        self.stages.clear()
        self.stage_timings.clear()


class PipelineAnalyzer:
    """
    Analyzes pipeline bottlenecks and identifies which stages are slow.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.component_timings: Dict[str, deque] = {
            'yolo': deque(maxlen=30),
            'buffer': deque(maxlen=30),
            'posemamba': deque(maxlen=30),
            'visualization': deque(maxlen=30),
        }
    
    def record_component_time(self, component: str, time_ms: float) -> None:
        """
        Record execution time for a pipeline component.
        
        Args:
            component: One of 'yolo', 'buffer', 'posemamba', 'visualization'
            time_ms: Execution time in milliseconds
        """
        if component in self.component_timings:
            self.component_timings[component].append(time_ms)
    
    def get_component_stats(self, component: str) -> Dict[str, float]:
        """
        Get statistics for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with mean, p95, max timing in milliseconds.
        """
        if component not in self.component_timings:
            return {}
        
        timings = list(self.component_timings[component])
        if not timings:
            return {}
        
        timings_array = np.array(timings)
        return {
            'mean_ms': float(np.mean(timings_array)),
            'p95_ms': float(np.percentile(timings_array, 95)),
            'max_ms': float(np.max(timings_array)),
        }
    
    def identify_bottleneck(self) -> Optional[str]:
        """
        Identify which component is the slowest bottleneck.
        
        Returns:
            Component name with highest mean execution time, or None if no data.
        """
        means = {}
        for component, timings in self.component_timings.items():
            if timings:
                means[component] = np.mean(list(timings))
        
        if not means:
            return None
        
        return max(means, key=means.get)
    
    def print_analysis(self) -> None:
        """Print a human-readable analysis of pipeline performance."""
        print("\n" + "=" * 80)
        print("PIPELINE BOTTLENECK ANALYSIS")
        print("=" * 80)
        
        bottleneck = self.identify_bottleneck()
        
        for component in ['yolo', 'buffer', 'posemamba', 'visualization']:
            stats = self.get_component_stats(component)
            if stats:
                is_bottleneck = " ⚠️  BOTTLENECK" if component == bottleneck else ""
                print(
                    f"{component:15} | mean: {stats['mean_ms']:7.2f}ms | "
                    f"p95: {stats['p95_ms']:7.2f}ms | max: {stats['max_ms']:7.2f}ms{is_bottleneck}"
                )
        
        print("=" * 80 + "\n")


@dataclass
class DeploymentRequirements:
    """Container for deployment feasibility requirements."""
    max_p95_latency_ms: float = 200.0
    min_throughput_fps: float = 10.0
    max_jitter_ms: float = 50.0
    max_frame_drop_pct: float = 5.0
    note: str = ""


class DeploymentChecker:
    """
    Checks whether the pipeline meets deployment feasibility requirements.
    """
    
    def __init__(self, requirements: Optional[DeploymentRequirements] = None):
        """
        Args:
            requirements: DeploymentRequirements object. Defaults to conservative values.
        """
        self.requirements = requirements or DeploymentRequirements()
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passes: List[str] = []
    
    def check_feasibility(
        self,
        stats: LatencyStats,
        frame_drop_pct: float = 0.0,
        target_fps: float = 30.0,
    ) -> bool:
        """
        Check if pipeline meets deployment requirements.
        
        Args:
            stats: LatencyStats from LatencyTracker
            frame_drop_pct: Percentage of frames dropped (0-100)
            target_fps: Camera FPS (used to compute expected throughput)
            
        Returns:
            True if all requirements met, False otherwise.
        """
        self.issues.clear()
        self.warnings.clear()
        self.passes.clear()
        
        # Check P95 latency
        if stats.p95_ms > self.requirements.max_p95_latency_ms:
            self.issues.append(
                f"❌ P95 Latency: {stats.p95_ms:.1f}ms > {self.requirements.max_p95_latency_ms}ms"
            )
        else:
            self.passes.append(
                f"✓ P95 Latency: {stats.p95_ms:.1f}ms ≤ {self.requirements.max_p95_latency_ms}ms"
            )
        
        # Check jitter (std deviation)
        if stats.std_ms > self.requirements.max_jitter_ms:
            self.warnings.append(
                f"⚠️  High Jitter: {stats.std_ms:.1f}ms > {self.requirements.max_jitter_ms}ms"
            )
        else:
            self.passes.append(
                f"✓ Jitter acceptable: {stats.std_ms:.1f}ms"
            )
        
        # Check frame drop rate
        if frame_drop_pct > self.requirements.max_frame_drop_pct:
            self.issues.append(
                f"❌ Frame Drop Rate: {frame_drop_pct:.2f}% > {self.requirements.max_frame_drop_pct}%"
            )
        else:
            self.passes.append(
                f"✓ Frame Drop Rate: {frame_drop_pct:.2f}%"
            )
        
        # Check throughput
        # Throughput = predictions per second = 1000 / mean_latency_ms
        actual_throughput_fps = 1000.0 / stats.mean_ms if stats.mean_ms > 0 else 0
        if actual_throughput_fps < self.requirements.min_throughput_fps:
            self.issues.append(
                f"❌ Throughput: {actual_throughput_fps:.1f}fps < {self.requirements.min_throughput_fps}fps"
            )
        else:
            self.passes.append(
                f"✓ Throughput: {actual_throughput_fps:.1f}fps"
            )
        
        # Print summary
        self._print_summary(stats, frame_drop_pct, actual_throughput_fps)
        
        return len(self.issues) == 0
    
    def _print_summary(
        self,
        stats: LatencyStats,
        frame_drop_pct: float,
        actual_throughput_fps: float,
    ) -> None:
        """Print deployment feasibility check summary."""
        print("\n" + "=" * 80)
        print("DEPLOYMENT FEASIBILITY CHECK")
        print("=" * 80)
        
        print(f"\nLatency Statistics (n={stats.sample_count}):")
        print(f"  Mean:   {stats.mean_ms:7.2f}ms")
        print(f"  P50:    {stats.p50_ms:7.2f}ms")
        print(f"  P95:    {stats.p95_ms:7.2f}ms (requirement: ≤{self.requirements.max_p95_latency_ms}ms)")
        print(f"  P99:    {stats.p99_ms:7.2f}ms")
        print(f"  Max:    {stats.max_ms:7.2f}ms")
        print(f"  Std:    {stats.std_ms:7.2f}ms (jitter, requirement: ≤{self.requirements.max_jitter_ms}ms)")
        
        print(f"\nThroughput:")
        print(f"  Actual:     {actual_throughput_fps:6.2f} FPS")
        print(f"  Required:   {self.requirements.min_throughput_fps:6.2f} FPS")
        
        print(f"\nRobustness:")
        print(f"  Frame Drop: {frame_drop_pct:.2f}% (requirement: ≤{self.requirements.max_frame_drop_pct}%)")
        
        print(f"\nRequirements Check:")
        for msg in self.passes:
            print(f"  {msg}")
        
        if self.warnings:
            print(f"\nWarnings:")
            for msg in self.warnings:
                print(f"  {msg}")
        
        if self.issues:
            print(f"\nCritical Issues:")
            for msg in self.issues:
                print(f"  {msg}")
            print(f"\n❌ NOT FEASIBLE FOR DEPLOYMENT")
        else:
            print(f"\n✅ DEPLOYMENT FEASIBLE!")
        
        if self.requirements.note:
            print(f"\nNote: {self.requirements.note}")
        
        print("=" * 80 + "\n")
    
    def set_requirements(self, requirements: DeploymentRequirements) -> None:
        """Update deployment requirements."""
        self.requirements = requirements


class TemporalConfigComparison:
    """
    Compares different temporal window configurations (5, 9, 27 frames).
    Helps you choose the best trade-off for your use case.
    """
    
    CONFIGS = {
        5: {
            'name': 'PoseMamba-S (5 frames)',
            'latency_ms': 167,  # 5 frames @ 30fps
            'accuracy_relative': 1.0,
            'expected_throughput_fps': 6,
        },
        9: {
            'name': 'PoseMamba-S (9 frames)',
            'latency_ms': 300,  # 9 frames @ 30fps
            'accuracy_relative': 1.03,
            'expected_throughput_fps': 3,
        },
        27: {
            'name': 'PoseMamba-S (27 frames)',
            'latency_ms': 900,  # 27 frames @ 30fps
            'accuracy_relative': 1.05,
            'expected_throughput_fps': 1,
        },
    }
    
    @staticmethod
    def print_comparison() -> None:
        """Print comparison of temporal window configs."""
        print("\n" + "=" * 100)
        print("TEMPORAL WINDOW CONFIGURATION COMPARISON")
        print("=" * 100)
        print(f"{'Config':<30} {'Latency':<15} {'Accuracy':<15} {'Throughput':<15}")
        print(f"{'-' * 100}")
        
        for n_frames, config in TemporalConfigComparison.CONFIGS.items():
            accuracy_str = f"+{(config['accuracy_relative'] - 1) * 100:.1f}%"
            print(
                f"{config['name']:<30} "
                f"{config['latency_ms']:>6.0f}ms{'':<7} "
                f"{accuracy_str:>6} baseline{'':<5} "
                f"{config['expected_throughput_fps']:>6.1f} FPS"
            )
        
        print("=" * 100)
        print("Recommendation: For real-time applications, use 5-frame window (best latency-accuracy tradeoff)\n")
    
    @staticmethod
    def recommend_config(max_latency_ms: float) -> int:
        """
        Recommend a temporal window size based on latency budget.
        
        Args:
            max_latency_ms: Maximum acceptable latency in milliseconds
            
        Returns:
            Number of frames (5, 9, or 27) that fits the budget.
        """
        for n_frames in sorted(TemporalConfigComparison.CONFIGS.keys()):
            config = TemporalConfigComparison.CONFIGS[n_frames]
            if config['latency_ms'] <= max_latency_ms:
                recommended = n_frames
        
        print(
            f"For max latency {max_latency_ms}ms: "
            f"Recommend {recommended}-frame config "
            f"({TemporalConfigComparison.CONFIGS[recommended]['name']})"
        )
        return recommended


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("DEPLOYMENT PROFILER - EXAMPLE USAGE")
    print("=" * 80)
    
    # Create tracker and analyzer
    tracker = LatencyTracker(window_size=30)
    analyzer = PipelineAnalyzer()
    
    # Simulate some frame processing with realistic timings
    print("\nSimulating 30 frames of processing...")
    
    np.random.seed(42)
    for frame_id in range(30):
        tracker.start_frame(frame_id)
        
        # Simulate YOLO (30-35ms)
        yolo_time = np.random.normal(32, 3)
        tracker.mark_event(frame_id, 'yolo_start')
        time.sleep(yolo_time / 1000.0)  # Simulate work
        tracker.mark_event(frame_id, 'yolo_end')
        analyzer.record_component_time('yolo', yolo_time)
        
        # Simulate buffer accumulation (only on certain frames)
        if (frame_id + 1) % 5 == 0:
            buffer_time = 0  # Instantaneous
            tracker.mark_event(frame_id, 'buffer_full')
            
            # Simulate PoseMamba (50-100ms) - only when buffer is full
            posemamba_time = np.random.normal(75, 15)
            tracker.mark_event(frame_id, 'posemamba_start')
            time.sleep(posemamba_time / 1000.0)  # Simulate work
            tracker.mark_event(frame_id, 'posemamba_end')
            analyzer.record_component_time('posemamba', posemamba_time)
            
            # Simulate visualization (15-25ms)
            viz_time = np.random.normal(20, 5)
            tracker.mark_event(frame_id, 'viz_start')
            time.sleep(viz_time / 1000.0)  # Simulate work
            tracker.mark_event(frame_id, 'viz_end')
            analyzer.record_component_time('visualization', viz_time)
        
        tracker.end_frame(frame_id)
    
    # Get and display statistics
    stats = tracker.get_stats()
    
    print(f"\nLatency Statistics:")
    print(f"  Mean:  {stats.mean_ms:.2f}ms")
    print(f"  P95:   {stats.p95_ms:.2f}ms")
    print(f"  P99:   {stats.p99_ms:.2f}ms")
    print(f"  Max:   {stats.max_ms:.2f}ms")
    print(f"  Jitter (std): {stats.std_ms:.2f}ms")
    
    # Analyze bottlenecks
    analyzer.print_analysis()
    
    # Check deployment feasibility
    checker = DeploymentChecker(
        DeploymentRequirements(
            max_p95_latency_ms=200,
            min_throughput_fps=5,
            max_jitter_ms=50,
            note="Conservative requirements for AR/VR applications",
        )
    )
    checker.check_feasibility(stats, frame_drop_pct=0.5)
    
    # Compare temporal configurations
    TemporalConfigComparison.print_comparison()
    TemporalConfigComparison.recommend_config(max_latency_ms=300)
    
    print("\n✅ Profiler module ready for integration!\n")
