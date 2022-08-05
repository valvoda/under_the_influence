import re
import torch
import random
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class TestData:

    def __init__(self, batch_size, log):
        self.batch_size = batch_size
        self.log = log

    def make_loader(self, input, mask, labels, claims, train=True):
        labels = torch.tensor(labels)
        claims = torch.tensor(claims)
        data = TensorDataset(input, mask, labels, claims)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        return dataloader


    def text_preprocessing(self, text):
        """
        - Remove entity mentions (eg. '@united')
        - Correct errors (eg. '&amp;' to '&')
        @param    text (str): a string to be processed.
        @return   text (Str): the processed string.
        """
        # Remove '@name'
        text = re.sub(r'(@.*?)[\s]', ' ', text)

        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)

        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    def preprocessing_for_bert(self, data, tokenizer, max=512):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """

        # For every sentence...
        input_ids = []
        attention_masks = []

        for sent in tqdm(data):
            sent = " ".join(sent)
            sent = sent[:500000] # Speeds the process up for documents with a lot of precedent we would truncate anyway.
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = tokenizer.encode_plus(
                text=self.text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True,  # Return attention mask
                truncation=True,
            )

            # Add the outputs to the lists
            input_ids.append([encoded_sent.get('input_ids')])
            attention_masks.append([encoded_sent.get('attention_mask')])

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def load(self):

        data_size = 100
        dataset_dog = ['this is a dog' for _ in range(data_size // 2)]
        dataset_cat = ['this is a cat' for _ in range(data_size // 2)]
        label_dog = [[0, 1] for _ in range(data_size // 2)]
        label_cat = [[1, 0] for _ in range(data_size // 2)]
        test_ids = [i for i in range(data_size)][:10]

        dataset = dataset_dog + dataset_cat
        labels = label_dog + label_cat

        c = list(zip(dataset, labels))
        random.seed(10)
        random.shuffle(c)

        dataset, labels = zip(*c)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids, masks = self.preprocessing_for_bert(dataset, tokenizer, max=20)

        train_dataloader = self.make_loader(input_ids[:80], masks[:80], labels[:80], labels[:80], train=True)
        val_dataloader = self.make_loader(input_ids[80:90], masks[80:90], labels[80:90], labels[80:90], train=False)
        test_dataloader = self.make_loader(input_ids[90:], masks[90:], labels[90:], labels[90:], train=False)

        if self.log != None:
            self.log.ids = test_ids
            self.log.precedent = test_ids

        return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    batch_size = 2
    data = TestData(batch_size, None)
    train_dataloader, val_dataloader, test_dataloader = data.load()
    print('test')