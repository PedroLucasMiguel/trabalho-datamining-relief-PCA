from torch.utils.data import DataLoader
from .arff import ArffLib
import torch.nn as nn
import torch.cuda as cuda

def fe(model:nn.Module = None, 
        data_loader:DataLoader = None,
        n_features:int = 100,
        arff_file_name:str = 'resnet50',
        arff_file_path:str = '.') -> None:
    
    device = "cuda" if cuda.is_available() else "cpu"

    activations_output = {}

    def hook(model, input, output):
        aux_array = output.detach().cpu().numpy()
        aux_array = aux_array.reshape(aux_array.shape[1] * aux_array.shape[2], 
                                      aux_array.shape[3])
        activations_output['avg_pool'] = aux_array.flatten()

    model.to(device)

    model.baseline.avgpool.register_forward_hook(hook)

    arff = ArffLib(arff_file_name, 
                    arff_file_path, 
                    n_features)
        
    arff.create_file()

    print(f"Iniciando extração do modelo {model.__class__.__name__}...")

    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        model(data)

        arff.append_to_file(activations_output['avg_pool'], target[0])

    print("Extração finalizada...")

    arff.close_file()