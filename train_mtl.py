import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.metrics import AccMetric
from LibMTL.loss import CELoss

from hanziconv import HanziConv

class load_dataset(torch.utils.data.Dataset):
    def __init__(self, my_dataset, max_length):
        self.labels, self.data = self.load_data(my_dataset)#把数据和标签取出
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoded_input = self.bert_tokenizer.encode_plus(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
        label = torch.tensor(label)
        return inputs, label

    def __len__(self):
        return len(self.labels)

    def load_data(self, my_dataset,data_size=None):
        items = [item.strip().split("_!_") for item in open(my_dataset).readlines()]
        labels = [int(item[1]) for item in items]
        sents = [HanziConv.toSimplified(item[2]) for item in items]
        return labels, sents

# define encoder and decoders
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', add_pooling_layer=True)
    
    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        token_type_ids=inputs['token_type_ids'])
        return outputs[1]
decoders = nn.ModuleDict({"vulgar_text": nn.Sequential(nn.Dropout(p=0.1, inplace=False), nn.Linear(768, 2)) })

from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.weighting import EW
from LibMTL.architecture import HPS

def parse_args(parser):
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    return parser.parse_args()

if __name__ == "__main__":

    params = parse_args(LibMTL_args)
    set_random_seed(params.seed)

    kwargs, optim_param, scheduler_param = prepare_args(params)


    # define tasks
    task_name = "vulgar_text"
    task_dict = {task_name: {'metrics': 'Acc', 'metrics_fn': AccMetric(), 'loss_fn': CELoss, 'weight': 1} }

    trainer = Trainer(task_dict=task_dict,
                        weighting=EW,
                        architecture=HPS,
                        encoder_class=Encoder,
                        decoders=decoders,
                        rep_grad=params.rep_grad,
                        multi_input=True,
                        optim_param=optim_param,
                        scheduler_param=scheduler_param,
                        save_path="checkpoints/",
                        **kwargs)

    train_set = load_dataset("data/train.txt", 128)
    train_dataloaders = {task_name: torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)}
    test_set = load_dataset("data/test_v2.txt", 128)
    test_dataloaders = {task_name: torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)}
    
    trainer.train(train_dataloaders=train_dataloaders, 
                  test_dataloaders=test_dataloaders, 
                  epochs=params.epochs)


