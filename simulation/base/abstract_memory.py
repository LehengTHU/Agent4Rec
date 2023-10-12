class abstract_memory:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.memory_size = 0

    def add_memory(self):
        """
        Add one new memory to the memory bank
        """
        raise NotImplementedError
    
    def time_weighting(self):
        """
        Weighting the memory according to the time
        """
        raise NotImplementedError
    
    def importance_weighting(self):
        """
        Weighting the importance of memory according to 
        the results of recommendation and the personal taste
        """
        raise NotImplementedError
    
    def reflect(self):
        """
        Generate a high level understanding of previous memories
        """
        raise NotImplementedError