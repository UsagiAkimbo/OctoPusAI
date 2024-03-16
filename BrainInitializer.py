from models import get_model
from communication import communication_bus

class BrainInitializer:
    def __init__(self):
        communication_bus.register_listener("initialize_brain", self.initialize_brain)

    def initialize_brain(self, data, brain_type, **kwargs):
        """
        Dynamically create a new AI "mini-brain" based on the specified type and parameters.
        
        :param brain_type: The type of the brain to initialize (e.g., 'CNN', 'LSTM').
        :param kwargs: Additional parameters for brain initialization.
        :return: An instance of the initialized model.
        """
        try:
            # Use get_model for dynamic model creation based on the specified brain type.
            model = get_model(brain_type, **kwargs)
            return model
        except ValueError as e:
            print(f"Error initializing brain: {e}")
            return None

