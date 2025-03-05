import json
from pathlib import Path

class SharpnessCache:
    def __init__(self, cache_file=".claritas_cache.json"):
        """
        Initialize sharpness cache.
        
        Args:
            cache_file: Cache file name
        """
        self.cache_file = cache_file
        self.cache = {}
        self.load()
        
    def load(self):
        """Load cache from file."""
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}
            
    def save(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
            
    def get(self, image_path):
        """Get cached sharpness value."""
        key = self._get_cache_key(image_path)
        entry = self.cache.get(key)
        
        if entry:
            path_stat = Path(image_path).stat()
            if (entry['mtime'] == path_stat.st_mtime and 
                entry['size'] == path_stat.st_size):
                return entry['score']
        return None
        
    def set(self, image_path, score):
        """Set sharpness value in cache."""
        key = self._get_cache_key(image_path)
        path_stat = Path(image_path).stat()
        
        self.cache[key] = {
            'score': score,
            'mtime': path_stat.st_mtime,
            'size': path_stat.st_size
        }
        
    def _get_cache_key(self, image_path):
        """Generate cache key for image."""
        return str(Path(image_path).absolute())
