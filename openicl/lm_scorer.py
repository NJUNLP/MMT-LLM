import torch
import tqdm

class Scorer:
    def __init__(self,
                 model,
                 tokenizer,
                 dataloader
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader


    def nll_loss(self, entry, output):
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = entry.input_ids[..., 1:].contiguous()
        pad_token_id = self.tokenizer.pad_token_id
        # entry.labels is already padded with pad_token_id, we further pad it to full length
        pad_mask = torch.nn.functional.pad(entry.labels,
                                           (shift_labels.shape[-1] - entry.labels.shape[-1], 0),
                                           value=pad_token_id)
        shift_labels.masked_fill_(pad_mask == pad_token_id, pad_token_id)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        answer_lens = (entry.labels != pad_token_id).sum(-1)
        loss = loss.sum(-1) / answer_lens
        loss = loss.cpu().detach().numpy().tolist()
        return loss