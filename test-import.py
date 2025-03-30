# Test new imports
try:
    from pydantic_settings import BaseSettings

    print("Successfully imported BaseSettings from pydantic_settings")
except ImportError as e:
    print(f"Error: {e}")
