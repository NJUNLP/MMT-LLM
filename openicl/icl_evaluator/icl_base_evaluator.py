'''Base Evaluator'''
from typing import List
class IclBaseEvaluator:
    def __init__(self,
                 references: List
    ) -> None:
        self.references = references
    
    
    def score(self):
        raise NotImplementedError("Method hasn't been implemented yet")
    