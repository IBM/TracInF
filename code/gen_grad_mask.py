#! /usr/bin/env python3

import torch
import pif as ptif
# from train_sentihood import load_model, BERT_Model, BERT_name
from train import load_model, BERT_Model, BERT_name
from data_util import DatasetReader
import argparse
import pdb
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

'''
This is for generating masked version of data and s_test. 
'''

def load_data(
        Utraindids, Utraintids, # unmasked
        Utrain_inputs, Utrain_masks, Utrain_labels,
        traindids, traintids, # masked
        train_inputs, train_masks, train_labels,
        testdids, testtids, #masked
        test_inputs, test_masks, test_labels,
        batch_size=4
):

    Utrain_data = TensorDataset(torch.tensor(Utraindids),
                               torch.tensor(Utraintids),
                               torch.tensor(Utrain_inputs),
                               torch.tensor(Utrain_masks),
                               torch.tensor(Utrain_labels))

    Utrain_sampler = RandomSampler(Utrain_data)

    Utrain_dataloader = DataLoader(Utrain_data,
                                  sampler=Utrain_sampler,
                                  batch_size=batch_size)

    train_data = TensorDataset(torch.tensor(traindids),
                               torch.tensor(traintids),
                               torch.tensor(train_inputs),
                               torch.tensor(train_masks),
                               torch.tensor(train_labels))

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    test_data = TensorDataset(torch.tensor(testdids),
                              torch.tensor(testtids),
                              torch.tensor(test_inputs),
                              torch.tensor(test_masks),
                              torch.tensor(test_labels))

    test_sampler = RandomSampler(test_data)

    test_dataloader = DataLoader(test_data,
                                 sampler=test_sampler,
                                 batch_size=batch_size)

    return Utrain_dataloader, train_dataloader, test_dataloader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a classifier model for IF')
    parser.add_argument("--data_dir", default="./data/MAMS-ATSA/", type=str, required=False, help="The input data_dir")
    # parser.add_argument("--data_dir", default="./data/sentihood/", type=str, required=False, help="The input data_dir")
    parser.add_argument("--train_data_name", default="model_6_train_w_qa", type=str, required=True, help="The input training data name")
    parser.add_argument("--test_data_name", default='model_6_test_w_qa', type=str, required=True, help="The input testing data name")
    parser.add_argument("--bert_name", default='roberta-large', type=str, required=False, help="The input testing data name")

    parser.add_argument("--gradient_out_dir", default="./data/roberta-large-model_6_train_w_qa/", type=str, required=True, help="specifies the name of model here")
    parser.add_argument("--log_file_name", default=None, type=str, required=True, help="The log file name")

#    parser.add_argument("--test_sample_num", default=1332, type=int, required=True,help="test sample number")
    parser.add_argument("--num_classes", default=3, type=int, required=True,help="num of classes for the model")

    parser.add_argument("--test_start_pos", default=0, type=int, required=True,help="Testing start pos")
    parser.add_argument("--test_end_pos", default=0, type=int, required=True,help="Testing start pos")

    parser.add_argument("--train_start_pos", default=0, type=int, required=True,help="Testing start pos")
    parser.add_argument("--train_end_pos", default=0, type=int, required=True,help="Testing start pos")

    parser.add_argument("--test_hessian_start_pos", default=0, type=int, required=True,help="Testing start pos")
    parser.add_argument("--test_hessian_end_pos", default=0, type=int, required=True,help="Testing start pos")

    parser.add_argument("--recursion_depth", default=0, type=int, required=True, help="recursion depth of data")
    parser.add_argument("--scale", default=25, type=int, required=True, help="recursion depth of data")
    parser.add_argument("--damp", default=0.01, type=float, required=True, help="recursion depth of data")
    parser.add_argument('--calculate_if', action='store_true')

    args = parser.parse_args()

    # If two names does not equal, the init model won't match the loaded model.
    assert(BERT_name == args.bert_name)

    dr = DatasetReader(max_len=128, BERT_name=args.bert_name)
    # dr = DatasetReader(max_len=60, BERT_name=args.bert_name) # for sentihood

    breakpoint()
    traindids, traintids, trainx, trainm, trainl = dr.read_data(
        args.data_dir + args.train_data_name[:-5] + '.json', mask_version=False )

    Mtraindids, Mtraintids, Mtrainx, Mtrainm, Mtrainl = dr.read_data(
        args.data_dir + args.train_data_name[:-5] + '.json', mask_version=True )

    Mtestdids, Mtesttids, Mtestx,  Mtestm,  Mtestl  = dr.read_data(
        args.data_dir + args.test_data_name + '.json', mask_version=True )

    Utrainloader, trainloader, testloader = load_data(
            traindids, traintids, trainx, trainm, trainl,
            Mtraindids, Mtraintids, Mtrainx, Mtrainm, Mtrainl,
            Mtestdids, Mtesttids, Mtestx, Mtestm, Mtestl
            )

    config = {
            "outdir": args.gradient_out_dir, # the dir should combine dataset and method, e.g. IF-model_1_val_w_qa
            "seed": 42,
            "gpu": 0,
            "recursion_depth": args.recursion_depth, # set recursion to use entire training data
            "r_averaging": 1,
            "scale": args.scale,
            "damp": args.damp,
            "num_classes": args.num_classes,
#            "test_sample_num": args.test_sample_num,
            "test_start_index": args.test_start_pos,
            "test_end_index": args.test_end_pos,
            "test_hessian_start_index": args.test_hessian_start_pos,
            "test_hessian_end_index": args.test_hessian_end_pos,
            "train_start_index": args.train_start_pos,
            "train_end_index": args.train_end_pos,
            "log_filename": args.log_file_name,
        }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(BERT_Model, args.bert_name + '-' + args.train_data_name + '.out' , device)
    print(args.bert_name + '-' + args.train_data_name + '.out' )
    model.eval()
    # 10, 11 layers and pooler, classifer layers needs to be set to true.
    #bert.encoder.layer.11.output.LayerNorm.bias True
    #bert.pooler.dense.weight True
    #bert.pooler.dense.bias True
    #classifier.weight True
    #classifier.bias True
    '''
    for name, param in model.named_parameters():
        if 'bert.encoder.layer.10' in name or 'bert.encoder.layer.11' in name or 'bert.pooler' in name or 'classifier' in name:
            param.requires_grad=True
        else:
            param.requires_grad=False
    '''
    ptif.init_logging(config['log_filename'])

    ptif.calc_all_grad_then_test_mask(
        config, model, Utrainloader, trainloader, testloader,
        calculate_if=args.calculate_if
    )
