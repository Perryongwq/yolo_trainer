import os
import yaml

class YAMLManager:
    @staticmethod
    def load_yaml(filepath):
        """Load and parse a YAML file"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load YAML file: {str(e)}")
    
    @staticmethod
    def save_yaml(filepath, content):
        """Save content to a YAML file"""
        try:
            with open(filepath, 'w') as f:
                yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise Exception(f"Failed to save YAML file: {str(e)}")
    
    @staticmethod
    def create_default_yaml(filepath, path=None, classes=None):
        """Create a default YAML file with the specified path and classes"""
        if path is None:
            path = "./datasets/custom"
        
        if classes is None:
            classes = ["class0"]
        
        content = {
            'path': path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': classes
        }
        
        YAMLManager.save_yaml(filepath, content)
        return content