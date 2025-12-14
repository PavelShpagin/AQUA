#!/usr/bin/env python3
"""
High-performance LLM processing scale test.

Target: Process 4K rows in 2-3 minutes using maximum optimization.
- Session pooling and connection reuse
- Parallel processing with optimal concurrency
- Fast timeouts and fail-fast behavior  
- Direct LLM calls bypassing ensemble overhead
- Real-time throughput monitoring
"""

import os
import sys
import time
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from utils.llm.backends import call_model
from utils.judge import detect_language_from_text, parse_tpfp_label

# Global spaCy model cache to avoid per-thread loading
_global_model_cache = {}
_model_lock = Lock()

def get_fast_alignment(src: str, tgt: str) -> str:
    """Ultra-fast alignment generation without heavy spaCy loading per thread."""
    # Simple diff-based alignment for speed
    if src == tgt:
        return src
    
    # Basic word-level diff
    src_words = src.split()
    tgt_words = tgt.split()
    
    if len(src_words) == len(tgt_words):
        changes = []
        for i, (s_word, t_word) in enumerate(zip(src_words, tgt_words)):
            if s_word != t_word:
                changes.append(f"{s_word}->{t_word}")
        if changes:
            return f"{src} [{', '.join(changes)}]"
    
    return f"{src} -> {tgt}"


@dataclass
class ProcessingStats:
    """Real-time processing statistics."""
    start_time: float
    completed: int = 0
    successful: int = 0
    failed: int = 0
    total_tokens: int = 0
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []

    def add_result(self, success: bool, tokens: int, response_time: float):
        """Thread-safe result tracking."""
        self.completed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
        self.total_tokens += tokens
        self.response_times.append(response_time)
    
    def get_throughput(self) -> float:
        """Current requests per second."""
        elapsed = time.time() - self.start_time
        return self.completed / elapsed if elapsed > 0 else 0
    
    def get_eta(self, total_rows: int) -> float:
        """Estimated time to completion in seconds."""
        throughput = self.get_throughput()
        if throughput > 0:
            remaining = total_rows - self.completed
            return remaining / throughput
        return 0
    
    def get_summary(self, total_rows: int) -> str:
        """Get current processing summary."""
        elapsed = time.time() - self.start_time
        throughput = self.get_throughput()
        eta = self.get_eta(total_rows)
        
        avg_response = statistics.mean(self.response_times) if self.response_times else 0
        med_response = statistics.median(self.response_times) if self.response_times else 0
        
        return f"""
📊 PROCESSING STATS:
   Completed: {self.completed:,}/{total_rows:,} ({100*self.completed/total_rows:.1f}%)
   Success Rate: {100*self.successful/self.completed:.1f}% ({self.successful}/{self.completed})
   Throughput: {throughput:.1f} req/sec
   Elapsed: {elapsed/60:.1f}min
   ETA: {eta/60:.1f}min remaining
   
📈 RESPONSE TIMES:
   Average: {avg_response:.2f}s
   Median: {med_response:.2f}s
   Total Tokens: {self.total_tokens:,}
"""


def build_gec_prompt(src: str, tgt: str, language: str = "Spanish") -> str:
    """Build optimized GEC classification prompt."""
    # Ultra-fast alignment generation
    aligned = get_fast_alignment(src, tgt)
    
    # Minimal prompt for speed
    prompt = f"""Classify this {language} grammar correction:

Original: {src}
Corrected: {tgt}
Changes: {aligned}

Answer: TP (valid correction), FP (unnecessary), TN (no change needed), or FN (error missed)"""
    
    return prompt


def process_single_row(row_data: Tuple[int, pd.Series], backend: str, api_token: str, stats: ProcessingStats, stats_lock: Lock) -> Dict:
    """Process a single row with optimal performance."""
    idx, row = row_data
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    if not src or not tgt:
        with stats_lock:
            stats.add_result(False, 0, 0)
        return {'idx': idx, 'src': src, 'tgt': tgt, 'prediction': 'Error', 'reasoning': 'Empty input'}
    
    start_time = time.time()
    
    try:
        # Build prompt  
        prompt = build_gec_prompt(src, tgt, "Spanish")
        
        # Direct LLM call with maximum speed optimizations
        success, content, token_usage = call_model(
            prompt=prompt,
            backend=backend,
            api_token=api_token,
            moderation=False,
            no_temperature=False,
            temperature_override=0.0  # Zero temperature for maximum speed
        )
        
        response_time = time.time() - start_time
        tokens_used = token_usage.get('total_tokens', 0)
        
        # Parse result
        if success and content:
            prediction = parse_tpfp_label(content)
            if prediction not in ['TP', 'FP', 'TN', 'FN']:
                prediction = 'TP' if 'TP' in content.upper() else 'FP'  # Fallback parsing
        else:
            prediction = 'Error'
        
        # Update stats thread-safely
        with stats_lock:
            stats.add_result(success, tokens_used, response_time)
        
        return {
            'idx': idx,
            'src': src,
            'tgt': tgt,
            'prediction': prediction,
            'reasoning': content[:100] if content else 'No response',
            'response_time': response_time,
            'tokens': tokens_used,
            'success': success
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        with stats_lock:
            stats.add_result(False, 0, response_time)
        
        return {
            'idx': idx,
            'src': src,
            'tgt': tgt,
            'prediction': 'Error',
            'reasoning': str(e)[:100],
            'response_time': response_time,
            'tokens': 0,
            'success': False
        }


def run_scale_test(
    input_file: str = "data/eval/es_eval.csv",
    backend: str = "gas_gemini20_flash_lite",  # Fastest for bulk processing
    max_workers: int = 150,  # Maximum concurrency for 2-3 min target
    max_rows: Optional[int] = None,
    report_interval: int = 100
):
    """Run optimized scale test targeting 2-3 minutes for 4K rows."""
    
    print("🚀 HIGH-PERFORMANCE GEC SCALE TEST")
    print("=" * 60)
    
    # Load data
    print(f"📂 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if max_rows:
        df = df.head(max_rows)
    
    total_rows = len(df)
    print(f"📊 Processing {total_rows:,} rows")
    print(f"🔧 Backend: {backend}")
    print(f"⚡ Concurrency: {max_workers} workers")
    
    # Get API token
    api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
    if not api_token:
        print("❌ ERROR: No API token found. Set API_TOKEN or OPENAI_API_KEY")
        return
    
    print(f"🔑 API Token: {'Found' if api_token else 'Missing'}")
    print()
    
    # Initialize stats
    stats = ProcessingStats(start_time=time.time())
    stats_lock = Lock()
    results = []
    
    # Target metrics - aggressive 2-3 min target
    target_time_min = 2.5 if total_rows >= 3000 else 3.0  # More aggressive for large datasets
    target_throughput = total_rows / (target_time_min * 60)  # req/sec needed
    print(f"🎯 TARGET: {total_rows:,} rows in {target_time_min:.1f} min")
    print(f"🎯 REQUIRED THROUGHPUT: {target_throughput:.1f} req/sec")
    
    # Warmup - prime the connection pool
    print(f"🔥 Warming up connection pool...")
    warmup_start = time.time()
    warmup_prompt = "Classify: 'Hola' -> 'Hello'. Answer: TP/FP/TN/FN"
    success, _, _ = call_model(warmup_prompt, backend, api_token, temperature_override=0.0)
    warmup_time = time.time() - warmup_start
    print(f"🔥 Warmup completed in {warmup_time:.1f}s - {'Success' if success else 'Failed'}")
    print()
    
    # Process with optimal concurrency
    print("⚡ Starting processing...")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(process_single_row, (idx, row), backend, api_token, stats, stats_lock): idx
                for idx, row in df.iterrows()
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Report progress periodically
                    if i % report_interval == 0 or i == total_rows:
                        print(stats.get_summary(total_rows))
                        
                        # Check if we're on track
                        current_throughput = stats.get_throughput()
                        if current_throughput < target_throughput * 0.8:  # 80% of target
                            print(f"⚠️  WARNING: Throughput {current_throughput:.1f} req/sec below target {target_throughput:.1f}")
                        
                except Exception as e:
                    print(f"❌ Future failed: {e}")
                    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        
    # Final results
    end_time = time.time()
    total_time = end_time - stats.start_time
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)
    
    print(f"⏱️  TIMING:")
    print(f"   Total Time: {total_time/60:.2f} minutes ({total_time:.1f}s)")
    print(f"   Target Time: {target_time_min:.1f} minutes")
    print(f"   {'✅ SUCCESS' if total_time <= target_time_min*60 else '❌ MISSED TARGET'}")
    
    print(f"\n📊 THROUGHPUT:")
    print(f"   Final Throughput: {stats.get_throughput():.1f} req/sec")
    print(f"   Target Throughput: {target_throughput:.1f} req/sec")
    print(f"   Efficiency: {100*stats.get_throughput()/target_throughput:.1f}%")
    
    print(f"\n✅ SUCCESS RATE:")
    print(f"   Successful: {stats.successful:,}/{stats.completed:,} ({100*stats.successful/stats.completed:.1f}%)")
    print(f"   Failed: {stats.failed:,}")
    
    if stats.response_times:
        print(f"\n⚡ RESPONSE TIMES:")
        print(f"   Average: {statistics.mean(stats.response_times):.2f}s")
        print(f"   Median: {statistics.median(stats.response_times):.2f}s")
        print(f"   Min: {min(stats.response_times):.2f}s")
        print(f"   Max: {max(stats.response_times):.2f}s")
    
    print(f"\n🔢 TOKENS:")
    print(f"   Total: {stats.total_tokens:,}")
    print(f"   Avg per request: {stats.total_tokens/stats.completed:.1f}" if stats.completed > 0 else "   Avg per request: 0")
    
    # Extrapolation
    if total_rows < 4000:
        scale_factor = 4000 / total_rows
        projected_time = total_time * scale_factor
        print(f"\n🔮 4K EXTRAPOLATION:")
        print(f"   Projected time for 4K rows: {projected_time/60:.2f} minutes")
        print(f"   {'✅ Would meet target' if projected_time <= target_time_min*60 else '❌ Would miss target'}")
    
    # Save results
    if results:
        output_file = f"test/scale_results_{int(time.time())}.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    return {
        'total_time_minutes': total_time / 60,
        'throughput_req_sec': stats.get_throughput(),
        'success_rate': stats.successful / stats.completed if stats.completed > 0 else 0,
        'total_tokens': stats.total_tokens,
        'target_met': total_time <= target_time_min * 60
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-performance LLM scale test - Target: 4K rows in 2-3 min")
    parser.add_argument('--input', default='data/eval/es_eval.csv', help='Input CSV file')
    parser.add_argument('--backend', default='gas_gemini20_flash_lite', help='LLM backend (gas_gemini20_flash_lite fastest)')
    parser.add_argument('--workers', type=int, default=150, help='Number of concurrent workers')
    parser.add_argument('--max-rows', type=int, help='Limit number of rows to process')
    parser.add_argument('--report-interval', type=int, default=100, help='Progress report interval')
    
    args = parser.parse_args()
    
    results = run_scale_test(
        input_file=args.input,
        backend=args.backend,
        max_workers=args.workers,
        max_rows=args.max_rows,
        report_interval=args.report_interval
    )
    
    print(f"\n🏁 Test completed with {'SUCCESS' if results['target_met'] else 'TARGET MISSED'}")
