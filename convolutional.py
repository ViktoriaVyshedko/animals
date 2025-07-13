import torch
from datasets import get_loaders
from model import SimpleCNN
from trainer import train_model
from utils import count_parameters
#from pytorch_lightning.callbacks import EarlyStopping


device = torch.device('cpu')

train_loader, test_loader = get_loaders(batch_size=64)

simple_cnn = SimpleCNN(num_classes=10).to(device)

print(f"Simple CNN parameters: {count_parameters(simple_cnn)}")

print("Training Simple CNN...")
simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=10, device=str(device))


'''
early_stopping = EarlyStopping(
    monitor="val_loss",   # Мониторим validation loss
    patience=5,           # Сколько эпох ждать улучшения
    mode="min",           # Минимизируем loss (для accuracy — "max")
    verbose=True,
)

trainer = Trainer(
    callbacks=[early_stopping, checkpoint],
    max_epochs=100,
)
trainer.fit(model, datamodule)'''