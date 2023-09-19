from abc import ABC, abstractmethod
import torch
class Aggregation(ABC):
    @abstractmethod
    def __call__(self, 
                 params: list):
        pass
    
class FedAVG(Aggregation):
    def __call__(self, params: list, clients_weights: list):
        with torch.no_grad():
        # Initialize a new dict to store the aggregated parameters
            assert len(params) == len(clients_weights)
            aggregated_params = {k: 0 for k in params[0].keys()}
            # Total weight for normalization (if weights are not normalized)
            total_weight = sum(clients_weights)
            # For each client's parameters...
            for w, p in zip(clients_weights, params):
                # For each parameter in the state_dict...
                for k, v in p.items():
                    # Add this client's weighted parameters to the aggregated parameters
                    aggregated_params[k] += v * w / total_weight

            return aggregated_params
        

if __name__ == "__main__":
    import torch.nn as nn
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(3, 16, 3, 1)
            self.bn = nn.BatchNorm2d(16)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return x
    num_clients = 3
    models = [SimpleModel() for _ in range(num_clients)]
    clients_weights = [1.0, 2.0, 3.0]  # Example weights for clients
    params = [model.state_dict() for model in models]
    aggregator = FedAVG()
    aggregated_params = aggregator(params, clients_weights)
    print(aggregated_params['bn.weight'])
    print(aggregated_params['bn.bias'])