"""Speed Benchmark for NILM Transformer"""
import torch
import time
import sys
sys.path.insert(0, 'transformer')

from model import HybridCNNTransformer

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = HybridCNNTransformer(
        n_features=7, 
        appliances=['test'],
        use_stationarization=True,
        d_model=128,
        n_layers=4
    ).to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Benchmark
    batch_sizes = [8, 32, 64]
    seq_lens = [256, 512, 1024]
    
    for bs in batch_sizes:
        for seq in seq_lens:
            x = torch.randn(bs, seq, 7).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            t0 = time.perf_counter()
            n_iters = 10
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            ms = (t1 - t0) / n_iters * 1000
            throughput = bs * n_iters / (t1 - t0)
            print(f"Batch {bs:2d} x Seq {seq:4d}: {ms:6.1f} ms  |  {throughput:7.0f} samples/sec")

if __name__ == '__main__':
    benchmark()
