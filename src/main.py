from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.resnet import ResNet50
from models.densenet import DenseNet201
from models.efficientnet import EfficientNetB0
from torch.utils.data import DataLoader
from extractor.feature_extractor import *
import torch.optim as optim
import torch
import json
import os

from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from torch.utils.data import random_split

# Hiper parâmetros
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001

RESNET_N_FEATURES = 1280
DENSENET_N_FEATURES = 1920
EFFICIENTNET_N_FEATURES = 1280

def extract_features(dataset_path:str = '../dataset') -> None:
    # Criando o dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = ImageFolder(dataset_path, transform=transform)
    data_loader = DataLoader(data, batch_size=1, num_workers=0)

    resnet_model = ResNet50(2)
    densenet_model = DenseNet201(2)
    efficientnet_model = EfficientNetB0(2)

    fe(model=resnet_model,
        data_loader=data_loader,
        n_features=RESNET_N_FEATURES,
        arff_file_name="resnet50_avgpool",
        arff_file_path=os.path.join("..", "output"))

    fe(model=densenet_model,
        data_loader=data_loader,
        n_features=DENSENET_N_FEATURES,
        arff_file_name="densenet201_avgpool",
        arff_file_path=os.path.join("..", "output"))
    
    fe(model=efficientnet_model,
        data_loader=data_loader,
        n_features=EFFICIENTNET_N_FEATURES,
        arff_file_name="efficientnetb0_avgpool",
        arff_file_path=os.path.join("..", "output"))

def train_validade_test_model(model_name:str, dataset_path:str = '../dataset') -> None:
    
    # JSON com as métricas finais
    final_json = {}

    preprocess = transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carregando o dataset a partir da pasta
    dataset = ImageFolder(dataset_path, preprocess)

    # Criando o dataset com split 80/20 (Perfeitamente balanceado)
    train_split, val_split = random_split(dataset, [0.8, 0.2])

    # Criando os "loaders" para o nosso conjunto de treino e validação
    train_loader = DataLoader(train_split, 
                             batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_split,
                            batch_size=BATCH_SIZE)


    # Verificando se a máquina possui suporte para treinamento em GPU
    # (Suporte apenas para placas NVIDIA)
    device = f"cuda" if cuda.is_available() else "cpu"

    match model_name:
        case "ResNet50":
            model = ResNet50(2).to(device)
        case "DenseNet201":
            model = DenseNet201(2).to(device)
        case "EfficientNetB0":
            model = EfficientNetB0(2).to(device)

    print(f"Treinando {model.__class__.__name__} utilizando: {device}")

    # Definindo o otimizador e a loss-functions
    optimizer = optim.Adamax(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss().to(device)

    val_metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average='weighted'),
        "recall": Recall(average='weighted'),
        "f1": (Precision(average='weighted') * Recall(average='weighted') * 2 / (Precision(average='weighted') + Recall(average='weighted'))),
        "loss": Loss(criterion)
    }

    # Definindo os trainers para treinamento e validação
    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    val_evaluator = create_supervised_evaluator(model, val_metrics, device)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    train_bar = ProgressBar(desc="Treinando...")
    val_bar = ProgressBar(desc="Validando...")
    train_bar.attach(trainer)
    val_bar.attach(val_evaluator)

    # Função que é executada ao fim de toda epoch de treinamento
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics

        final_json[trainer.state.epoch] = metrics

        print(f"Resultados da Validação - Epoch[{trainer.state.epoch}] {final_json[trainer.state.epoch]}")

    # Definição da métrica para realizar o "checkpoint" do treinamento
    # nesse caso será utilizada a métrica F1
    def score_function(engine):
        return engine.state.metrics["f1"]
    
    # Definindo o processo de checkpoint do modelo
    model_checkpoint = ModelCheckpoint(
        os.path.join("..", "output"),
        require_empty=False,
        n_saved=1,
        filename_prefix=f"{model.__class__.__name__}_train",
        score_function=score_function,
        score_name="f1",
        global_step_transform=global_step_from_engine(trainer),
    )
        
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    print(f"\nTreinando o modelo {model.__class__.__name__}...")

    trainer.run(train_loader, max_epochs=EPOCHS)

    print(f"\nTrain finished for model {model.__class__.__name__}")

    # Salvando as métricas em um arquivo .json
    with open(os.path.join("..", "output", f"{model.__class__.__name__}_training_results.json"), "w") as f:
        json.dump(final_json, f)

if __name__ == "__main__":

    # Seed para manter reprodutibilidade
    torch.manual_seed(0)

    try:
        os.mkdir(os.path.join("..", "output"))
    except OSError as _:
        pass
    
    terminate_execution = False
    
    while not terminate_execution:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Escolha uma das opções abaixo:")
        print("[1] - Extrair as características dos modelos")
        print("[2] - Treinar a FC de cada modelo")
        print("[3] - Finalizar programa")

        op = int(input("Resposta: "))

        match op:
            case 1:
                extract_features(os.path.join("..", "dataset"))

                input("\nPressione ENTER para voltar ao menu")
            case 2:
                train_validade_test_model('DenseNet201')
                input("\nPressione ENTER para voltar ao menu")
            case _:
                terminate_execution = True

    