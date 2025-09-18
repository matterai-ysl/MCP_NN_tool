# BASE_URL = "http://localhost:8090"
BASE_URL = "http://47.99.180.80/nn"
DOWNLOAD_URL = "./trained_models"
MCP_PORT = 8090
from pathlib import Path


def get_download_url(path: str):
    """Get download URL for a file path.

    Args:
        path: File path (can be absolute or relative)

    Returns:
        Download URL for the file
    """
    path_obj = Path(path)

    # Try to make relative to DOWNLOAD_URL first
    try:
        relative_path = path_obj.relative_to(DOWNLOAD_URL)
        return f"{BASE_URL}/download/file/{relative_path.as_posix()}"
    except ValueError:
        # If path is not under DOWNLOAD_URL, try to find a common base
        # This handles user-specific directories like trained_models/user1/
        try:
            # Try to find the first occurrence of a models directory
            parts = path_obj.parts
            if 'trained_models' in parts or 'trained_model' in parts:
                # Find the index of the models directory
                models_idx = None
                for i, part in enumerate(parts):
                    if part in ['trained_models', 'trained_model']:
                        models_idx = i
                        break

                if models_idx is not None:
                    # Create relative path from the models directory
                    relative_parts = parts[models_idx:]
                    relative_path = Path(*relative_parts)
                    return f"{BASE_URL}/download/file/{relative_path.as_posix()}"

            # Fallback: use the path as-is if it's already relative
            if not path_obj.is_absolute():
                return f"{BASE_URL}/download/file/{path_obj.as_posix()}"
            else:
                # Last resort: use just the filename
                return f"{BASE_URL}/download/file/{path_obj.name}"

        except Exception:
            # Ultimate fallback
            return f"{BASE_URL}/download/file/{path_obj.name}"

def get_static_url(path: str):
    """Get static URL for a file path.

    Args:
        path: File path (can be absolute or relative)

    Returns:
        Static URL for the file
    """
    path_obj = Path(path)

    # Try to make relative to DOWNLOAD_URL first
    try:
        relative_path = path_obj.relative_to(DOWNLOAD_URL)
        return f"{BASE_URL}/static/{relative_path.as_posix()}"
    except ValueError:
        # If path is not under DOWNLOAD_URL, try to find a common base
        try:
            # Try to find the first occurrence of a models directory
            parts = path_obj.parts
            if 'trained_models' in parts or 'trained_model' in parts:
                # Find the index of the models directory
                models_idx = None
                for i, part in enumerate(parts):
                    if part in ['trained_models', 'trained_model']:
                        models_idx = i
                        break

                if models_idx is not None:
                    # Create relative path from the models directory
                    relative_parts = parts[models_idx:]
                    relative_path = Path(*relative_parts)
                    return f"{BASE_URL}/static/{relative_path.as_posix()}"

            # Fallback: use the path as-is if it's already relative
            if not path_obj.is_absolute():
                return f"{BASE_URL}/static/{path_obj.as_posix()}"
            else:
                # Last resort: use just the filename
                return f"{BASE_URL}/static/{path_obj.name}"

        except Exception:
            # Ultimate fallback
            return f"{BASE_URL}/static/{path_obj.name}"