import torch
import torch.nn as nn
import math

class LongformerClassifier(nn.Module):
    """
        Bert Model for Classification Tasks.
        """

    def __init__(self, model, out_dim=2, dropout=0.2, n_hidden=50, device='cpu', longformer=False, use_claims=False,
                 architecture='classifier'):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, n_hidden, out_dim

        self.model = model

        self.device = device
        self.longformer = False
        if model.name_or_path == 'allenai/longformer-base-4096':
            self.longformer = True

        self.mtl = False
        self.discriminate = False
        self.use_claims = use_claims

        if architecture == 'mtl':
            self.mtl = True
        elif architecture == 'latent':
            self.discriminate = True

        # for claim embeddings
        vocab_size = 19
        self.embedding = nn.Embedding(vocab_size, D_in)

        # Instantiate an one-layer feed-forward classifier for main task
        if architecture == 'baseline_all':
            self.classifier_positive = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(H, D_out * 3)
            )
        else:
            self.classifier_positive = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(H, D_out)
            )

        # Instantiate an one-layer feed-forward classifier for auxilary task
        self.classifier_aux = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, D_out)
        )

    def process_claims(self, claims, outputs):
        # Introduce claims
        # claims = BATCH_N x LABEL_N

        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]
        embedded = self.embedding(claims).to(self.device)
        # last_hidden_state_cls concatenated with claim embeddings

        all_batches = torch.zeros(embedded.size(0), embedded.size(2))
        for i in range(embedded.size(0)):
            all_batches[i, :] = torch.mean(
                torch.stack([last_hidden_state_cls[i, :], embedded[i][claims[0] != 0].mean(0)]), dim=0)

        last_hidden_state_cls = all_batches.to(self.device)

        return last_hidden_state_cls

    def forward(self, input_ids, attention_mask, global_attention, claims):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        if self.longformer:
            outputs = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Introduce claims as input
        if self.use_claims:
            last_hidden_state_cls = self.process_claims(claims, outputs)

        # Feed input to classifier to compute logits of pos_precedent
        logits = self.classifier_positive(last_hidden_state_cls)

        if self.mtl:
            # Feed input to classifier to compute logits of neg_precedent
            logits_aux = self.classifier_aux(last_hidden_state_cls)
            logits = torch.cat((logits, logits_aux), dim=1)
        elif self.discriminate:
            # Feed input to classifier to compute logits of claims
            logits_aux = self.classifier_aux(last_hidden_state_cls)
            return logits, logits_aux

        return logits

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, model, args, device):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, args.n_hidden, args.out_dim
        self.model = model
        self.device = device


        self.mlp = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Feed input to classifier to compute logits of pos_precedent
        logits = self.mlp(last_hidden_state_cls)

        return logits

class JointClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, model, args, device):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(JointClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, args.n_hidden, args.out_dim
        self.model = model
        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(H, D_out * 3)
        )

    def forward(self, input_ids, attention_mask, global_attention):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        # Feed input to BERT
        outputs = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Feed input to classifier to compute logits of pos_precedent
        logits = self.mlp(last_hidden_state_cls)
        # logits = logits.reshape(-1, 3)

        return logits


class PositiveLinear(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        A = torch.Tensor(size_out, size_in)
        self.A = nn.Parameter(A)

        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        A_mul = torch.mm(self.A.t(), self.A)
        a_times_x = torch.mm(x, A_mul)
        w_times_a = torch.mm(a_times_x, self.weights.t())
        # return torch.add(w_times_a, self.bias)  # w times x + b
        return w_times_a

class TestClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, model, args, device):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(TestClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, args.n_hidden, args.out_dim
        self.model = model
        self.device = device

        self.linear_layer = nn.Linear(D_in, D_out)

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Feed input to classifier to compute logits of pos_precedent
        #logits = self.mlp(last_hidden_state_cls)
        logits = self.linear_layer(last_hidden_state_cls)

        return logits