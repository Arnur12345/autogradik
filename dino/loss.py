import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, nepochs,
                 center_momentum=0.9, student_temp=0.1,
                 teacher_temp=0.07):
        super().__init__()
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1, out_dim))

    def H(self, t, s):
        t = t.detach()
        s = F.log_softmax(s / self.student_temp, dim=1)
        t = F.softmax((t - self.center) / self.teacher_temp, dim=1)
        return -(t * s).sum(dim=1).mean()

    def forward(self, student_global_view_output, student_local_view_output, teacher_global_view_output, epoch):
        t_g1, t_g2 = teacher_global_view_output.chunk(2)
        s_g1, s_g2 = student_global_view_output.chunk(2)

        s_locals = student_local_view_output.chunk(self.ncrops - 2)

        total_loss = 0
        n_loss = 0

        total_loss += self.H(t_g1, s_g2)
        n_loss += 1

        for local in s_locals:
            total_loss += self.H(t_g1, local) + self.H(t_g2, local)
            n_loss += 2

        total_loss += self.H(t_g2, s_g1)
        n_loss += 1

        loss = total_loss / n_loss

        self.update_center(teacher_global_view_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_mean = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_mean * (1.0 - self.center_momentum)
