"""
Quick Test Script for Hybrid CNN-Transformer NILM
==================================================
Run this to verify the model architecture works correctly.

Usage:
    cd transformer
    python test_model.py
"""
import sys
import os

# Disable CUDA for testing if there are driver issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def test_model():
    """Test model instantiation and forward pass."""
    print("Testing Hybrid CNN-Transformer NILM Model...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from config import Config
        cfg = Config()
        print(f"✓ Config loaded")
        print(f"  - Appliances: {cfg.appliances[:3]}...")
        print(f"  - Window size: {cfg.window_size}")
        print(f"  - d_model: {cfg.d_model}")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from model import HybridCNNTransformer
        
        # Create model with test config
        model = HybridCNNTransformer(
            n_features=7,
            d_model=64,  # Smaller for quick test
            n_heads=4,
            n_layers=2,
            d_ff=128,
            appliances=['test_appliance'],
            cnn_channels=[16, 32, 64],
            use_rope=True,
            seq2point=True
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created")
        print(f"  - Parameters: {n_params:,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Test forward pass
        batch_size = 4
        seq_len = 256  # Smaller for quick test
        n_features = 7
        
        x = torch.randn(batch_size, seq_len, n_features)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output appliances: {list(output.keys())}")
        
        for name, out in output.items():
            print(f"  - {name}: power={out['power'].shape}, state={out['state'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from losses import NILMLoss
        
        criterion = NILMLoss()
        
        # Create dummy targets
        targets = {
            'test_appliance': {
                'power': torch.randn(batch_size, 1),
                'state': torch.randint(0, 2, (batch_size, 1)).float()
            }
        }
        
        losses = criterion(output, targets)
        print(f"✓ Loss computation successful")
        print(f"  - Total loss: {losses['total'].item():.4f}")
        print(f"  - MSE: {losses['mse'].item():.4f}")
        print(f"  - BCE: {losses['bce'].item():.4f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from dataset import NILMDataset
        import numpy as np
        
        # Create dummy data
        n_samples = 1000
        aggregate = np.random.randn(n_samples).astype(np.float32)
        targets = {'test': np.random.randn(n_samples).astype(np.float32)}
        temporal = np.random.randn(n_samples, 6).astype(np.float32)
        
        dataset = NILMDataset(
            aggregate=aggregate,
            targets=targets,
            temporal=temporal,
            window_size=256,
            stride=64
        )
        
        print(f"✓ Dataset creation successful")
        print(f"  - Dataset length: {len(dataset)}")
        
        # Test getitem
        x, y = dataset[0]
        print(f"  - Sample X shape: {x.shape}")
        print(f"  - Sample Y keys: {list(y.keys())}")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*60)
    print("Testing NILMFormer Model (Single-Appliance Wrapper)...")
    try:
        from model import create_model
        
        # Test NILMFormer creation
        cfg.model_type = 'nilmformer'
        cfg.d_model = 96 # Standard NILMFormer size
        cfg.n_layers = 2
        cfg.n_features = 9 # Match config (1 agg + 8 temporal)
        
        model_nf = create_model(cfg)
        
        n_params_nf = sum(p.numel() for p in model_nf.parameters())
        print(f"✓ NILMFormer Wrapper created")
        print(f"  - Parameters: {n_params_nf:,}")
        
        # Test forward pass
        n_features_nf = 9
        x = torch.randn(batch_size, seq_len, n_features_nf).to(cfg.device)
        model_nf.to(cfg.device)
        
        with torch.no_grad():
            output_nf = model_nf(x)
            
        print(f"✓ NILMFormer Forward pass successful")
        for name, out in output_nf.items():
             print(f"  - {name}: power={out['power'].shape}, state={out['state'].shape}")
             # Logits check
             print(f"  - {name} state range: [{out['state'].min():.2f}, {out['state'].max():.2f}] (Logits expected)")

        # Verify differentiability (Requires Grad Check)
        x.requires_grad = True
        model_nf.train()
        out_grad = model_nf(x)
        # Check first appliance output
        first_key = list(out_grad.keys())[0]
        state_grad = out_grad[first_key]['state']
        
        if state_grad.grad_fn is not None:
             print(f"✓ State output is differentiable (grad_fn={state_grad.grad_fn.__class__.__name__})")
        else:
             print(f"✗ State output is NOT differentiable!")
             raise ValueError("State must be differentiable logits")

    except Exception as e:
        print(f"✗ NILMFormer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
