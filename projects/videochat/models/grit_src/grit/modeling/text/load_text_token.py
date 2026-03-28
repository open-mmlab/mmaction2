import torch


class LoadTextTokens(object):

    def __init__(self, tokenizer, max_text_len=40, padding='do_not_pad'):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.padding = padding

    def descriptions_to_text_tokens(self, target, begin_token):
        target_encoding = self.tokenizer(
            target,
            padding=self.padding,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len)

        need_predict = [1] * len(target_encoding['input_ids'])
        payload = target_encoding['input_ids']
        if len(payload) > self.max_text_len - 2:
            payload = payload[-(self.max_text_len - 2):]
            need_predict = payload[-(self.max_text_len - 2):]

        input_ids = [begin_token] + payload + [self.tokenizer.sep_token_id]

        need_predict = [0] + need_predict + [1]
        data = {
            'text_tokens': torch.tensor(input_ids),
            'text_lengths': len(input_ids),
            'need_predict': torch.tensor(need_predict),
        }

        return data

    def __call__(self, object_descriptions, box_features, begin_token):
        text_tokens = []
        text_lengths = []
        need_predict = []
        for description in object_descriptions:
            tokens = self.descriptions_to_text_tokens(description, begin_token)
            text_tokens.append(tokens['text_tokens'])
            text_lengths.append(tokens['text_lengths'])
            need_predict.append(tokens['need_predict'])

        text_tokens = torch.cat(
            self.collate(text_tokens), dim=0).to(box_features.device)
        text_lengths = torch.tensor(text_lengths).to(box_features.device)
        need_predict = torch.cat(
            self.collate(need_predict), dim=0).to(box_features.device)

        assert text_tokens.dim() == 2 and need_predict.dim() == 2
        data = {
            'text_tokens': text_tokens,
            'text_lengths': text_lengths,
            'need_predict': need_predict
        }

        return data

    def collate(self, batch):
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                assert all(
                    len(b.shape) == len(batch[0].shape) for b in batch[1:])
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if any(c < m for c, m in zip(b.shape, max_shape)):
                        b2 = torch.zeros(
                            max_shape, dtype=b.dtype, device=b.device)
                        if b.dim() == 1:
                            b2[:b.shape[0]] = b
                        elif b.dim() == 2:
                            b2[:b.shape[0], :b.shape[1]] = b
                        elif b.dim() == 3:
                            b2[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                        else:
                            raise NotImplementedError
                        b = b2
                    batch2.append(b[None, ...])
            else:
                batch2 = []
                for b in batch:
                    batch2.append(b[None, ...])
            return batch2
        else:
            raise NotImplementedError
