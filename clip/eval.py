import torch
import torch.nn.functional as F

@torch.no_grad()
def zeroshot_evaluate(model, test_loader, num_classes, device):
    model.eval()
    
    class_inputs = torch.arange(num_classes, device=device)
    class_emb = model.class_encoder(class_inputs)
    class_emb = F.normalize(class_emb, p=2, dim=-1)
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        image_emb = model.vision_encoder(images)
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        
        similarity = torch.matmul(image_emb, class_emb.t())
        predictions = similarity.argmax(dim=-1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    accuracy = (correct / total) * 100.0
    return accuracy