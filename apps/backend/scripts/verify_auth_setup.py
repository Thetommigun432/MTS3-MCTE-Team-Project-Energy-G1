"""
Script to verify Supabase backend configuration.
Run from repository root:
    python -m apps.backend.scripts.verify_auth_setup
"""
import sys
import os

# Add apps/backend to path
sys.path.append(os.path.join(os.getcwd(), 'apps/backend'))

from app.core.config import get_settings
from app.infra.supabase.client import init_supabase_client, get_supabase_client

def verify():
    print("Locked & Loaded: Verifying Supabase Config...")
    settings = get_settings()
    
    print("-" * 40)
    print(f"URL: {settings.supabase_url}")
    print(f"Publishable Key Configured: {'YES' if settings.supabase_publishable_key else 'NO'}")
    print(f"Anon Key Configured: {'YES' if settings.supabase_anon_key else 'NO'}")
    print(f"JWT Secret Configured: {'YES' if settings.supabase_jwt_secret else 'NO'}")
    print("-" * 40)

    if not settings.supabase_url:
        print("❌ ERROR: SUPABASE_URL is missing")
        return

    key = settings.supabase_publishable_key or settings.supabase_anon_key
    if not key:
        print("❌ ERROR: No Supabase key found (Publishable or Anon)")
        return
        
    print("Attempting connection...")
    try:
        init_supabase_client()
        client = get_supabase_client()
        if client.client:
            print("✅ Supabase Client initialized successfully")
        else:
            print("❌ Supabase Client failed to initialize")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    verify()
