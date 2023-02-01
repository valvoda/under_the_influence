import csv
import uuid
import torch
from pathlib import Path

class Logger:

    def __init__(self, args):
        self.ids = None
        self.precedent = None
        self.args = args
        self.id = uuid.uuid4().hex
        self.path = "trained_models/" + args.dataset + "/" + args.arch + "/" + args.model + "/" + args.input + "/" + self.id + "/"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def save_model(self, model):
        torch.save(model, self.path + "model.pt")

    def load_model(self):
        model = torch.load(self.path + "model.pt")
        return model

    def save_results(self, results, outputs):

        with open(self.path + "outputs.csv", 'w') as loss_file:
            writer = csv.writer(loss_file)
            for pred, truth, id, prec in zip(outputs['preds'], outputs['truths'], self.ids, self.precedent):
                writer.writerow([id, pred, truth, prec])

        csv_columns = [i[0] for i in self.args._get_kwargs()]
        csv_columns += list(results.keys())
        csv_data = {**dict(self.args._get_kwargs()), **results}
        csv_file = self.path + "results.csv"

        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(csv_data)

