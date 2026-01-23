"""
NILM Loss Functions
===================
Multi-objective loss for NILM training:
1. MSE/MAE for power regression
2. BCE for ON/OFF classification
3. Soft Dice Loss for temporal segmentation
4. WeightedNILMLoss: Specialized loss with False Negative penalty (from notebook)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# =============================================================================
# WEIGHTED NILM LOSS (From high-performing notebook)
# =============================================================================
class WeightedNILMLoss(nn.Module):
    """
    Specialized NILM Loss with class weights and False Negative penalty.
    
    This loss was identified as the KEY to the notebook's superior performance.
    It heavily penalizes missing ON events (False Negatives), forcing the model
    to actually detect activations rather than predicting zeros.
    
    Components:
    1. Hybrid MSE/MAE: (1-mix)*MSE + mix*MAE for robust regression
    2. Class Weights: Higher weight for ON samples (on_weight >> off_weight)
    3. FN Penalty: Extra penalty when True=ON but Pred=OFF
    
    Args:
        on_weight: Weight for ON samples (default: 30.0)
        off_weight: Weight for OFF samples (default: 5.0)  
        fn_weight: Extra penalty for False Negatives (default: 15.0)
        mix: MAE ratio in hybrid loss (0=pure MSE, 1=pure MAE, default: 0.5)
        on_threshold: Threshold to determine ON state (default: 0.001 normalized)
    """
    
    def __init__(
        self,
        on_weight: float = 30.0,
        off_weight: float = 5.0,
        fn_weight: float = 15.0,
        mix: float = 0.5,
        on_threshold: float = 0.001
    ):
        super().__init__()
        self.on_weight = on_weight
        self.off_weight = off_weight
        self.fn_weight = fn_weight
        self.mix = mix
        self.on_threshold = on_threshold
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss with False Negative penalty.
        
        Args:
            pred: Predicted power [batch, 1] (normalized)
            target: Ground truth power [batch, 1] (normalized)
            
        Returns:
            Scalar loss value
        """
        # Ensure same shape
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Determine ON/OFF states based on target
        is_on = (target.abs() > self.on_threshold).float()
        is_off = 1.0 - is_on
        
        # Class weights: ON samples weighted much more than OFF
        weights = is_on * self.on_weight + is_off * self.off_weight
        
        # Hybrid MSE/MAE loss
        mse = (pred - target) ** 2
        mae = torch.abs(pred - target)
        base_loss = (1 - self.mix) * mse + self.mix * mae
        
        # Weighted base loss
        weighted_loss = base_loss * weights
        
        # False Negative penalty: target is ON but prediction is low
        # This is the KEY innovation - heavily penalize missing activations
        # We classify a "miss" as target > threshold AND pred < target/2 (heuristic)
        # Or simply pred < target as in the notebook
        fn_mask = is_on * (pred < target).float()  # ON and under-predicting
        fn_penalty = fn_mask * self.fn_weight * mse
        
        # Total loss
        total = weighted_loss + fn_penalty
        
        return total.mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for temporal segmentation.
    
    Helps with class imbalance (appliances mostly OFF).
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [batch, *]
            target: Ground truth binary [batch, *]
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1.0 - dice


class WeightedMSE(nn.Module):
    """
    MSE with higher weight for ON states.
    
    Addresses class imbalance where most timesteps have near-zero consumption.
    """
    
    def __init__(self, on_weight: float = 5.0, threshold: float = 0.01):
        super().__init__()
        self.on_weight = on_weight
        self.threshold = threshold
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted power [batch, 1]
            target: Ground truth power [batch, 1]
        """
        # Calculate squared error
        sq_error = (pred - target) ** 2
        
        # Weight ON states more heavily
        weights = torch.where(
            target.abs() > self.threshold,
            torch.full_like(target, self.on_weight),
            torch.ones_like(target)
        )
        
        return (sq_error * weights).mean()


class FocalMSE(nn.Module):
    """
    Focal MSE Loss - focuses on hard examples (Improved V3 Implementation).
    
    Uses tanh-based scaling to heavily penalize errors on active appliances.
    focal = 1 + gamma * tanh(mse / (target + epsilon))
    """
    
    def __init__(self, gamma: float = 2.0, on_weight: float = 5.0, threshold: float = 0.02):
        super().__init__()
        self.gamma = gamma
        self.on_weight = on_weight
        self.threshold = threshold
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_error = (pred - target) ** 2
        
        with torch.no_grad():
            # V3 Logic: Scale focal weight by relative error magnitude
            # This makes the model focus on "hard" examples (large errors relative to signal)
            focal_weight = 1 + self.gamma * torch.tanh(sq_error / (target.abs() + 0.01))
            
            # Class weighting
            state_weight = torch.where(
                target.abs() > self.threshold, 
                torch.tensor(self.on_weight, device=target.device),
                torch.tensor(1.0, device=target.device)
            )
            
        return (sq_error * focal_weight * state_weight).mean()


# =============================================================================
# ENERGY CONSISTENCY LOSS (Confidence-Gated)
# =============================================================================
class EnergyConsistencyLoss(nn.Module):
    """
    Confidence-gated energy conservation constraint.
    
    Enforces that the sum of predicted appliance powers approximates 
    the aggregate power, BUT only for appliances with high confidence.
    
    This prevents the model from incorrectly "filling" missing power
    onto appliances that should be OFF.
    
    Applied only to the midpoint for causal consistency.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__()
        self.confidence_threshold = confidence_threshold
    
    def forward(self, predictions: dict, aggregate_midpoint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: {appliance: {'power': [B, 1], 'state': [B, 1]}}
            aggregate_midpoint: [B, 1] aggregate power at midpoint
            
        Returns:
            L1 loss between sum of confident predictions and aggregate
        """
        total_pred = torch.zeros_like(aggregate_midpoint)
        n_confident = 0
        
        for name, pred in predictions.items():
            power = pred['power']
            state_logit = pred['state']
            
            # Confidence = probability from sigmoid
            confidence = torch.sigmoid(state_logit)
            
            # Create mask for high-confidence predictions
            high_conf_mask = confidence > self.confidence_threshold
            
            # Add power weighted by confidence (soft gating)
            # When ON with high confidence: full power
            # When OFF with high confidence: power → 0 anyway
            # When low confidence: reduced contribution
            weighted_power = power * confidence
            total_pred = total_pred + weighted_power
            
            if high_conf_mask.any():
                n_confident += 1
        
        # Only apply loss if we have confident predictions
        if n_confident == 0:
            return torch.tensor(0.0, device=aggregate_midpoint.device)
        
        # L1 loss (robust to outliers)
        return F.l1_loss(total_pred, aggregate_midpoint)



class NILMLoss(nn.Module):
    """
    Combined Multi-Objective NILM Loss with Energy Consistency.
    
    Components:
    1. Regression Loss: MSE / FocalMSE / WeightedNILMLoss (New!)
    2. BCE Loss: ON/OFF classification stability
    3. Dice Loss: Temporal segmentation quality
    4. Energy Loss: Physical constraint (sum ≈ aggregate)
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.1,
        energy_weight: float = 0.1,
        on_weight: float = 5.0,
        loss_type: str = 'focal', # 'focal', 'mse', or 'weighted_nilm'
        focal_gamma: float = 2.0,
        energy_warmup_steps: int = 1000,
        fn_weight: float = 15.0 # For WeightedNILMLoss
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.energy_weight = energy_weight
        self.energy_warmup_steps = energy_warmup_steps
        self.current_step = 0
        
        # Regression loss selection
        if loss_type == 'weighted_nilm':
            # The powerful loss from the notebook
            self.reg_loss = WeightedNILMLoss(
                on_weight=30.0, # Stronger ON weight as in notebook
                off_weight=5.0,
                fn_weight=fn_weight,
                mix=0.5
            )
        elif loss_type == 'focal':
            self.reg_loss = FocalMSE(gamma=focal_gamma, on_weight=on_weight)
        else:
            self.reg_loss = WeightedMSE(on_weight=on_weight)
        
        # Classification loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Segmentation loss
        self.dice_loss = SoftDiceLoss()
        
        # Energy consistency loss
        self.energy_loss = EnergyConsistencyLoss()
        
    def get_energy_weight(self) -> float:
        """Get current energy weight with warmup scheduling."""
        if self.energy_warmup_steps <= 0:
            return self.energy_weight
        
        warmup_factor = min(1.0, self.current_step / self.energy_warmup_steps)
        return self.energy_weight * warmup_factor
        
    def step(self):
        """Increment step counter for warmup scheduling."""
        self.current_step += 1
        
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, Dict[str, torch.Tensor]],
        aggregate_midpoint: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for all appliances.
        
        Args:
            predictions: {appliance: {'power': [B,1], 'state': [B,1]}}
            targets: {appliance: {'power': [B,1], 'state': [B,1]}}
            aggregate_midpoint: Optional [B, 1] aggregate power for energy loss
            
        Returns:
            dict with 'total', 'mse', 'bce', 'dice', 'energy' and per-appliance losses
        """
        total_reg = 0.0
        total_bce = 0.0
        total_dice = 0.0
        n_appliances = 0
        
        losses = {}
        
        for name in predictions.keys():
            if name not in targets:
                continue
                
            pred = predictions[name]
            tgt = targets[name]
            
            # Regression loss for power
            reg = self.reg_loss(pred['power'], tgt['power'])
            
            # BCE loss for state
            bce = self.bce_loss(pred['state'], tgt['state'])
            
            # Dice loss for state
            dice = self.dice_loss(pred['state'], tgt['state'])
            
            # Store per-appliance
            losses[f'{name}_mse'] = reg # We call it mse for logging consistency
            losses[f'{name}_bce'] = bce
            
            total_reg += reg
            total_bce += bce
            total_dice += dice
            n_appliances += 1
        
        if n_appliances == 0:
            return {'total': torch.tensor(0.0, requires_grad=True)}
        
        # Average across appliances
        avg_reg = total_reg / n_appliances
        avg_bce = total_bce / n_appliances
        avg_dice = total_dice / n_appliances
        
        # Combined loss
        total = (
            self.mse_weight * avg_reg +
            self.bce_weight * avg_bce +
            self.dice_weight * avg_dice
        )
        
        # Energy consistency loss (with warmup)
        energy = torch.tensor(0.0, device=avg_reg.device)
        if aggregate_midpoint is not None and self.energy_weight > 0:
            energy = self.energy_loss(predictions, aggregate_midpoint)
            effective_energy_weight = self.get_energy_weight()
            total = total + effective_energy_weight * energy
        
        losses.update({
            'total': total,
            'mse': avg_reg,
            'bce': avg_bce,
            'dice': avg_dice,
            'energy': energy
        })
        
        return losses


class SingleApplianceLoss(nn.Module):
    """
    Simplified loss for single-appliance training.
    """
    
    def __init__(self, on_weight: float = 5.0):
        super().__init__()
        self.mse = WeightedMSE(on_weight=on_weight)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


def create_loss(config) -> NILMLoss:
    """Create loss function from config."""
    # Check if we should use the new WeightedNILMLoss
    loss_type = getattr(config, 'loss_type', 'focal')
    
    # Optional overrides
    fn_weight = getattr(config, 'fn_weight', 15.0)
    
    return NILMLoss(
        mse_weight=config.loss_mse_weight,
        bce_weight=config.loss_bce_weight,
        dice_weight=config.loss_sdl_weight,
        energy_weight=getattr(config, 'energy_loss_weight', 0.1),
        on_weight=5.0,
        loss_type=loss_type,
        focal_gamma=2.0,
        energy_warmup_steps=getattr(config, 'energy_warmup_steps', 1000),
        fn_weight=fn_weight
    )
