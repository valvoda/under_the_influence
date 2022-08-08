import sys
sys.path.append("../../")

import torch
from src.preprocess.data_loader import DataPrep
import argparse

import pickle
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--gpu", type=int, default=1, required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--test", dest='test', action='store_true')
    parser.add_argument("--input", type=str, default="facts", required=False)  # arguments
    parser.add_argument("--model", type=str, default="bert", required=False)
    parser.add_argument("--start", type=int, default=0, required=False)
    parser.add_argument("--end", type=int, default=10000, required=False)
    parser.add_argument("--path", type=str, default="b5c78e5c1a3c46c3bc725bee281cae51", required=False)
    args = parser.parse_args()

    print(args)

    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'

    sys.path.insert(0, '../train')
    # find the last trained model
    model_path = '../train/trained_models/precedent/' + args.model + '/' + args.input + '/' + args.path
    # model_path = find_best(model_path, args.input, args.model)
    # print(f'Best test F1: {get_score(model_path, "test")[1]}, val F1: {get_score(model_path, "val")[1]}')
    # Toy data model test:
    # model_path = '../train/trained_models/precedent/bert/both/e621d7fd7fcb4535a6b48207e1c03dfd'
    print('loaded:', model_path)
    model = torch.load(model_path + '/model.pt', map_location=torch.device(device))

    tokenized_dir = "../datasets/" + 'precedent' + "/" + args.model
    # tokenizer_dir, test, log, max_len, batch_size
    loader = DataPrep(tokenized_dir, args.test, None, args.max_len, args.batch_size, args.input)
    # loader = TestData(args.batch_size, None)
    train_dataloader, val_dataloader, test_dataloader = loader.load(args.start, args.end)
    model.eval()

    all_embs = {}

    all_embs['W'] = model.linear_layer.weight

    with torch.no_grad():
        for loader, name in zip([train_dataloader, val_dataloader, test_dataloader], ["train_dataloader", "val_dataloader", "test_dataloader"]):
            loader_embs = []
            for batch in tqdm(loader):
                b_input_ids, b_attn_mask, b_labels, b_claims = tuple(t.to(device) for t in batch)
                b_input_ids = b_input_ids.squeeze(1)
                b_attn_mask = b_attn_mask.squeeze(1)
                outputs = model.model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                emb = outputs.last_hidden_state[:, 0, :]
                loader_embs.append(emb)
            all_embs[name] = torch.cat(loader_embs, dim=0)

    with open(model_path + "/embeddings.pkl", "wb") as f:
        pickle.dump(all_embs, f, protocol=4)

    print('Done')

    # with open(model_path + "/embeddings.pkl", "rb") as f:
    #     emb_dic = pickle.load(f)
