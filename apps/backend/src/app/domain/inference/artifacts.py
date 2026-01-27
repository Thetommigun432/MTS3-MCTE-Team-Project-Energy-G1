"""
Model Artifact Management
=========================

Handles verification and downloading of model artifacts (safetensors).
Ensures that the required model files are present and match their SHA256 checksums.
"""

import hashlib
import shutil
import urllib.request
import urllib.error
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import ErrorCode, ModelError
from app.core.logging import get_logger
from app.domain.inference.registry import ModelEntry

logger = get_logger(__name__)


def calculate_sha256(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 64k chunks
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ensure_artifact_present(entry: ModelEntry, models_dir: Path | None = None) -> Path:
    """
    Ensure the model artifact exists and matches checksum.
    
    If missing and MODEL_ARTIFACT_BASE_URL is configured, attempts to download it.
    
    Args:
        entry: Model registry entry
        models_dir: Optional override for models directory (default: from settings)
        
    Returns:
        Path to the validated artifact
        
    Raises:
        ModelError: If artifact is missing/corrupt and cannot be recovered
    """
    settings = get_settings()
    
    # Resolve path
    if models_dir:
        base_dir = models_dir
    else:
        base_dir = Path(settings.models_dir)
        
    relative_path = Path(entry.artifact_path)
    artifact_path = base_dir / relative_path
    
    # 1. Check if file exists
    if artifact_path.exists():
        # If no SHA256 in registry, trust the file exists (local models)
        if not entry.artifact_sha256:
            logger.debug(f"Artifact exists, no SHA check for {entry.model_id}")
            return artifact_path
            
        current_hash = calculate_sha256(artifact_path)
        if current_hash == entry.artifact_sha256:
            # Match!
            return artifact_path
        
        logger.warning(
            f"Artifact checksum mismatch for {entry.model_id}",
            extra={
                "expected": entry.artifact_sha256,
                "actual": current_hash,
                "path": str(artifact_path),
            },
        )
        # Delete corrupt file
        artifact_path.unlink()
    
    # 2. File missing or deleted - Attempt download
    if not settings.model_artifact_base_url:
        raise ModelError(
            code=ErrorCode.MODEL_ARTIFACT_MISSING,
            message=f"Artifact missing for {entry.model_id} and no MODEL_ARTIFACT_BASE_URL configured",
            details={"path": str(artifact_path)},
        )
        
    # Construct download URL
    base_url = settings.model_artifact_base_url.rstrip("/")
    # Join with forward slashes regardless of OS
    url_path = "/".join(relative_path.parts)
    download_url = f"{base_url}/{url_path}"
    
    logger.info(f"Downloading artifact for {entry.model_id} from {download_url}")
    
    try:
        # Ensure parent directory exists
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with timeout and size limit
        # urllib doesn't support easy "max size" without custom reading, but we can check Content-Length
        with urllib.request.urlopen(download_url, timeout=settings.model_artifact_timeout_sec) as response:
            # Check content length if available
            content_length = response.getheader("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > settings.model_artifact_max_mb:
                    raise ModelError(
                        code=ErrorCode.MODEL_ARTIFACT_INVALID,
                        message=f"Artifact too large: {size_mb:.1f}MB > {settings.model_artifact_max_mb}MB",
                    )
            
            # Stream to file
            with open(artifact_path, "wb") as f:
                shutil.copyfileobj(response, f)
                
        # 3. Verify downloaded file (only if SHA is specified)
        if entry.artifact_sha256:
            new_hash = calculate_sha256(artifact_path)
            if new_hash != entry.artifact_sha256:
                artifact_path.unlink()  # Delete bad file
                raise ModelError(
                    code=ErrorCode.MODEL_ARTIFACT_INVALID,
                    message="Downloaded artifact checksum mismatch",
                    details={"expected": entry.artifact_sha256, "actual": new_hash},
                )
            
        logger.info(f"Successfully downloaded and verified {entry.model_id}")
        return artifact_path
        
    except urllib.error.URLError as e:
        # Clean up partial download
        if artifact_path.exists():
            artifact_path.unlink()
        
        raise ModelError(
            code=ErrorCode.MODEL_ARTIFACT_MISSING,
            message=f"Failed to download artifact: {e}",
            details={"url": download_url},
        )
    except Exception as e:
        if artifact_path.exists():
            artifact_path.unlink()
        raise ModelError(
            code=ErrorCode.MODEL_ARTIFACT_MISSING,
            message=f"Error handling artifact: {e}",
        )
