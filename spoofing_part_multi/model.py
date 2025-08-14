import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import shutil

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0, dropout=0.2, freeze_layers=None):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [
            nn.Conv2d(1, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False)
        )
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        if freeze_layers is not None:
            self.apply_freeze_strategy(freeze_layers)
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def apply_freeze_strategy(self, freeze_strategy):
        print(f"apply freezing strategy: {freeze_strategy}")
        if freeze_strategy == 'features':
            self._freeze_module(self.features)
        elif freeze_strategy == 'early_features':
            total_layers = len(self.features)
            freeze_count = total_layers // 2
            self._freeze_layers_by_count(freeze_count)
        elif freeze_strategy == 'backbone':
            total_layers = len(self.features)
            freeze_count = int(total_layers * 0.7)
            self._freeze_layers_by_count(freeze_count)
        elif isinstance(freeze_strategy, int):
            self._freeze_layers_by_count(freeze_strategy)
        elif isinstance(freeze_strategy, list):
            self._freeze_layers_by_indices(freeze_strategy)
        self._print_trainable_params()
    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
    def _freeze_layers_by_count(self, freeze_count):
        for i, layer in enumerate(self.features):
            if i < freeze_count:
                self._freeze_module(layer)
    def _freeze_layers_by_indices(self, indices):
        for i in indices:
            if i < len(self.features):
                self._freeze_module(self.features[i])
    def _print_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"parameters statistics:")
        print(f"    total parameters: {total_params:,}")
        print(f"    trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"    frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        self._print_trainable_params()
    def freeze_features_gradually(self, epoch, total_epochs, strategy='linear'):
        if strategy == 'linear':
            unfreeze_ratio = epoch / total_epochs
            total_layers = len(self.features)
            unfreeze_count = int(total_layers * unfreeze_ratio)
            self._freeze_module(self.features)
            for i in range(unfreeze_count):
                for param in self.features[i].parameters():
                    param.requires_grad = True

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, num_epochs=20, lr=0.00007, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        
    def train(self):
        model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        best_epoch = 0
        
        # Ensure snapshot directory exists
        os.makedirs('./snapshot', exist_ok=True)
        
        # Initialize training history file
        history_file = './snapshot/training_history.txt'
        with open(history_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,val_accuracy\n")
        
        print(f"\nStarting training, using device: {self.device}")
        print(f"Training epochs: {self.num_epochs}, learning rate: {self.lr}")
        print(f"Training history will be saved to: {history_file}")
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Training]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Validation]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Record training history
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            
            # Save training history to file
            with open(history_file, 'a') as f:
                f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{val_acc:.4f}\n")
            
            # Save model for each epoch
            epoch_model_path = f'./snapshot/model_epoch_{epoch+1:03d}.pth'
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Saved epoch {epoch+1} model: {epoch_model_path}")
            
            # Update learning rate scheduler
            scheduler.step(val_acc)
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                print(f"Found better model! Validation accuracy: {val_acc:.2f}% (Epoch {best_epoch})")
            
            # Print training results for this epoch
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'  Training - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
            print(f'  Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)
        
        # After training completes, save the best model as best_model.pth
        best_model_path = f'./snapshot/model_epoch_{best_epoch:03d}.pth'
        best_model_final_path = './snapshot/best_model.pth'
        
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, best_model_final_path)
            print(f"\nTraining completed!")
            print(f"Best model from epoch {best_epoch}, validation accuracy: {best_val_acc:.2f}%")
            print(f"Best model saved as: {best_model_final_path}")
            print(f"Training history saved as: {history_file}")
        else:
            print(f"\nWarning: Best model file not found {best_model_path}")
        
        return model, train_losses, val_losses, val_accuracies
    
    def save_test_results(self, model, test_loader, device='cuda'):
        """
        Save test results to file for visualization purposes
        
        Args:
            model: trained model
            test_loader: test data loader
            device: computing device
        """
        model.to(device)
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []  # 新增：保存预测概率
        
        print("\nEvaluating model on test set for confusion matrix...")
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 获取预测概率
                probabilities = torch.softmax(output, dim=1)
                
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Save test results to file (including probabilities)
        test_results_file = './snapshot/test_results.txt'
        
        # Get number of classes from first probability vector
        num_classes = len(all_probabilities[0]) if all_probabilities else 2
        
        with open(test_results_file, 'w') as f:
            # Create header with proper number of probability columns
            if num_classes == 6:
                header = "true_label,predicted_label,prob_clean,prob_ds1,prob_ds2,prob_ds3,prob_ds4,prob_ds7\n"
            elif num_classes == 2:
                header = "true_label,predicted_label,prob_real,prob_spoof\n"
            else:
                # General case for any number of classes
                prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
                header = "true_label,predicted_label," + ",".join(prob_cols) + "\n"
            
            f.write(header)
            
            for true_label, pred_label, prob in zip(all_labels, all_predictions, all_probabilities):
                prob_str = ",".join([f"{p:.6f}" for p in prob])
                f.write(f"{true_label},{pred_label},{prob_str}\n")
        
        print(f"Test results saved to: {test_results_file} (with {num_classes} class probabilities)")
        
        return all_predictions, all_labels, all_probabilities 