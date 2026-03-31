#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

try:
    from engine import SentimentEngine
    print("✓ Engine module imported successfully")
    
    engine = SentimentEngine()
    print("✓ Engine initialized successfully")
    
    has_client = hasattr(engine, 'client')
    print(f"✓ Engine has 'client' attribute: {has_client}")
    
    if not has_client:
        print("✗ ERROR: client attribute is missing!")
        
except Exception as e:
    print(f"✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
