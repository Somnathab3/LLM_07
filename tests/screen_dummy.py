"""
ScreenDummy implementation for BlueSky simulation
Based on bluesky-gym but without requiring the full package
"""

class ScreenDummy:
    """
    Dummy class for the screen. Reimplements echo method so that console messages are ignored.
    Minimal implementation to work with BlueSky headless mode.
    """
    
    def __init__(self):
        pass
    
    def echo(self, text='', flags=0):
        """Ignore all console output"""
        pass
    
    def update(self):
        """No-op for screen updates"""
        pass
    
    def reset(self):
        """No-op for screen reset"""
        pass
