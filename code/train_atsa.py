import torch
import argparse
import os
from tqdm import tqdm, trange
import numpy as np
import pdb


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from transformers import \
        BertForSequenceClassification, BertTokenizer, \
        RobertaForSequenceClassification, RobertaTokenizer, \
        AdamW

from data_util import DatasetReader

BERT_name='roberta-large'

if BERT_name=='bert-base-uncased':
## may test other BERT family model ##
    BERT_Model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=3, output_hidden_states=True)
elif BERT_name=='roberta-large':
    BERT_Model = RobertaForSequenceClassification.from_pretrained(
            'roberta-large', num_labels=3, output_hidden_states=True)



# save model for predict
def save_model(net, path):
    torch.save(net.state_dict(), path)

def load_model(net, path, device):
    net.load_state_dict(torch.load(path))#,map_location=torch.device(device)))
    net.cuda()
    return net

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train a classifier model for IF')
    parser.add_argument("--data_dir", default="./data/MAMS-ATSA/", type=str, required=False, help="The input data_dir")
    parser.add_argument("--train_data_name", default="model_5_train_w_qa.json", type=str, required=False, help="The input training data name")
    parser.add_argument("--dev_data_name", default="model_5_val_w_qa.json", type=str, required=False, help="The input dev data name")
    parser.add_argument("--test_data_name", default='model_5_test_w_qa.json', type=str, required=False, help="The input testing data name")
    parser.add_argument("--model_output", default='model_5_train_w_qa.out-otc', type=str, required=False, help="The output directory where the model checkpoints saved")
    parser.add_argument("--max_seq_length", default=128, type=int, help="")
    parser.add_argument("--batch_size", default=512, type=int, help="Total batch size for training and eval.")
    parser.add_argument("--epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--num_labels", default=3, type=int, help="labels to predict")


    args = parser.parse_args()

    print(args)

    use_gpu, device_name = False, 'cpu'
    if torch.cuda.is_available():
        device_name, use_gpu ='cuda', True
    device = torch.device(device_name)

    # for retrain
    data_reader = DatasetReader(args.max_seq_length, BERT_name)
    train_dids, train_tids, train_inputs, train_masks, train_labels = data_reader.read_data(
        os.path.join(args.data_dir,args.train_data_name),
        filter=False,
        aspect_only_by_rule=False #args.aspect_only
    )
    # data_reader = DatasetReader(args.max_seq_length, BERT_name)
    # train_dids, train_tids, train_inputs, train_masks, train_labels = data_reader.read_data(
    #     os.path.join(args.data_dir,args.train_data_name),
    #     aspect_only_by_rule=False #args.aspect_only
    # )

    dev_dids, dev_tids, dev_inputs, dev_masks, dev_labels = data_reader.read_data(
        os.path.join(args.data_dir,args.dev_data_name),
        aspect_only_by_rule=False #args.aspect_only
    )

    batch_size = args.batch_size


    train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    dev_data = TensorDataset(torch.tensor(dev_inputs), torch.tensor(dev_masks), torch.tensor(dev_labels))
    dev_sampler = SequentialSampler(dev_data) #RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    print("Data prepare Finished")

    ### ##

    if use_gpu:
        BERT_Model.cuda()

    '''
    param_optimizer = list(BERT_Model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {
            'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate':
            0.01
        },
        {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate':
            0.0
        }
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)
    '''

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in BERT_Model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in BERT_Model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    #optimizer = AdamW(BERT_Model.parameters(), lr=1e-5)

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    train_loss_set = []
    print("Starting training")

#    Freeze BERT ENCODER's Parameter!!!
#encoder.layer.11.output.LayerNorm.bias
#pooler.dense.weight
#pooler.dense.bias
# fine tune only last layer and output layer.
    for name, param in BERT_Model.base_model.named_parameters():
        param.requires_grad = False
        if 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'pooler.dense' in name:
            param.requires_grad = True
    for epoch in trange(args.epochs, desc="Epoch"):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        BERT_Model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print()
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            print("\rEpoch:", epoch, "step:", step, end='')
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = BERT_Model(
                b_input_ids,
#                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)
            train_loss_set.append(float(outputs.loss))
            # Backward pass
            outputs.loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += float(outputs.loss)
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps / nb_tr_examples))

        ##### Validation #####
        # Put model in evaluation mode to evaluate loss on the validation set
        BERT_Model.eval()
        # Tracking variables
        global_eval_accuracy = 0
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = BERT_Model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        if eval_accuracy > global_eval_accuracy:
            global_eval_accuracy = eval_accuracy
            save_model(BERT_Model, BERT_name + '-' + args.model_output)
