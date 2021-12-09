import torch 

DEVICE = 'cuda'

class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__()
    self.arc = smp.UnetPlusPlus(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        #encoder_depth = 5,
        decoder_channels = [1025, 512, 256, 128, 64],
        in_channels = 1,
        classes = 1,
        activation = None
    )
  
  def forward(self, images, masks=None):
    logits = self.arc(images)

    if masks != None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      return logits, loss1+loss2
    return logits

model = SegmentationModel()
model.to(DEVICE)

idx = 5

model.load_state_dict(torch.load("./best_model.pt"))

image, mask = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W) minibatch
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask >0.8) * 1.0
helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))