class IterationCounter():
    def __init__(self, filename="iterations.txt"):
        self.filename = filename
        self.iteration = self._get_iterations()
        
    def _get_iterations(self) -> int:
        try:
            with open(self.filename, 'r') as file:
                iteration = int(file.read().strip())
        except Exception as e :
            iteration = 0
            print (e)
        return iteration
    
    def _save(self) -> None:
        with open(self.filename, "w") as file:
            file.write(str(self.iteration))
    
    def increment_save(self) -> int:
        self.iteration += 1
        self._save()
        return self.iteration
    
    def get(self) -> int:
        return self.iteration
    
    