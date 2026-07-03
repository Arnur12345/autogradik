import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import make_dataloaders
from model import model_vit  # Подключаем наш ViT вместо старого build_model
from loss import DINOLoss
from eval import knn_evaluate
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------------------- hyperparameters ------------------------------- #
ARCH = "resnet18"             # Используется как ключ конфигурации в CONFIGS внутри ViT
PROJ_DIM = 100                # Задаем размерность проекции (NUM_CLASSES)
TEMPERATURE = 0.5             
BATCH_SIZE = 32
EPOCHS = 200
WARMUP_EPOCHS = 10
BASE_LR = 0.15                 # SGD, линейное масштабирование 0.3 * BATCH_SIZE / 256
MIN_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
KNN_K = 200
KNN_T = 0.1
EVAL_EVERY = 1              # Запуск kNN монитора каждые N эпох
NUM_CLASSES = 100
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _lr_at(epoch, epochs, warmup, base_lr, min_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup                  # linear warmup
    p = (epoch - warmup) / max(1, epochs - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))   # cosine decay


def main(arch=ARCH, proj_dim=PROJ_DIM, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
         epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, base_lr=BASE_LR, min_lr=MIN_LR,
         momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, num_workers=NUM_WORKERS,
         knn_k=KNN_K, knn_t=KNN_T, eval_every=EVAL_EVERY, num_classes=NUM_CLASSES, seed=SEED):
    
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # 1. Собираем даталоадеры (ncrops=4: 2 глобальных + 2 локальных)
    train_loader, memory_loader, test_loader = make_dataloaders(num_workers, batch_size=batch_size)

    # 2. Собираем ViT для Студента и Учителя
    print(f"Building ViT ({arch}) with output_dim={proj_dim}...")
    student_model = model_vit(size=arch, num_classes=proj_dim).to(DEVICE)
    teacher_model = model_vit(size=arch, num_classes=proj_dim).to(DEVICE)
    
    # Изначально веса Учителя полностью идентичны Студенту
    teacher_model.load_state_dict(student_model.state_dict())
    
    # Отключаем градиенты для Учителя (он обновляется только через EMA)
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 3. Настраиваем лосс и оптимизатор (Строго SGD для nesterov/momentum)
    criterion = DINOLoss(out_dim=proj_dim, ncrops=4, nepochs=epochs).to(DEVICE)
    
    optimizer = torch.optim.SGD(student_model.parameters(), lr=base_lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    
    print(f"DINO pretrain: ViT-{arch} on CIFAR-100 32x32 | batch {batch_size} | "
          f"{epochs} epochs | device {DEVICE}")

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval() # Учитель всегда в режиме eval
        
        lr = _lr_at(epoch, epochs, warmup_epochs, base_lr, min_lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        t0 = time.time()
        loss_sum, n_steps = 0.0, 0

        for x_g1, x_g2, x_locals in train_loader:
            x_g1 = x_g1.to(DEVICE, non_blocking=True)
            x_g2 = x_g2.to(DEVICE, non_blocking=True)
            # Оставляем локальные кропы списком на GPU
            x_locals = [x.to(DEVICE, non_blocking=True) for x in x_locals]
            
            # Глобальные кропы склеиваем — их всего 2, они пролезут
            x_student_global = torch.cat([x_g1, x_g2], dim=0)
            x_teacher_global = x_student_global # Учитель видит то же самое
            
            # 1. Forward Учителя 
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    teacher_output = teacher_model(x_teacher_global)
            
            # 2. Forward Студента на глобальных кропах
            with torch.amp.autocast('cuda', dtype=torch.float16):
                student_global_output = student_model(x_student_global)
            
            # 3. ПОСЛЕДОВАТЕЛЬНЫЙ Forward Студента на локальных кропах (Спасает от OOM)
            student_local_outputs = []
            for x_loc in x_locals:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out_loc = student_model(x_loc)
                student_local_outputs.append(out_loc)
            
            # Теперь склеиваем уже посчитанные эмбеддинги (они маленькие, размера [B, PROJ_DIM])
            student_local_output = torch.cat(student_local_outputs, dim=0)
                
            # 4. Вычисление лосса
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = criterion(
                    student_global_view_output=student_global_output,
                    student_local_view_output=student_local_output,
                    teacher_global_view_output=teacher_output,
                    epoch=epoch
                )
            
            # 5. Backward & Optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=3.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 6. EMA обновление весов Учителя
            with torch.no_grad():
                m = 0.996 
                for param_q, param_k in zip(student_model.parameters(), teacher_model.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            loss_sum += loss.item()
            n_steps += 1
            
        epoch_time = time.time() - t0
        avg_loss = loss_sum / n_steps
        print(f"Epoch {epoch:03d} | LR: {lr:.4f} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Периодическая kNN валидация замороженного бэкбона
        if (epoch + 1) % eval_every == 0:
            knn_acc = knn_evaluate(student_model, memory_loader, test_loader, DEVICE, k=knn_k, temperature=knn_t)
            print(f"--- kNN Evaluation Acc: {knn_acc:.2f}% ---")


if __name__ == "__main__":
    main()