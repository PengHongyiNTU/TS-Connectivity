from abc import ABC, abstractmethod
import random
class Selection(ABC):
    @abstractmethod
    def __call__(self, 
                 clients_list: list):
        pass

class RandomSelection(Selection):
    def __init__(self):
        pass
    def __call__(self, 
                 clients_list: list, 
                 clients_per_round: int):
        if clients_per_round > len(clients_list):
            raise ValueError('Number of clients to select is larger than the number of clients')
        selected_clients = random.sample(clients_list, clients_per_round)
        return selected_clients
        