#! /usr/bin/env python3

import torch
import pif as ptif
from train import load_model, BERT_Model, BERT_name
# from train_sentihood import load_model, BERT_Model, BERT_name # for sentihood
from data_util import DatasetReader
import argparse
import pdb
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

'''
This is for generating masked version of data and s_test. 
'''

def create_loader(Utraindids, Utraintids, Utrain_inputs, Utrain_masks, Utrain_labels, batch_size):
    Utrain_data = TensorDataset(torch.tensor(Utraindids),
                               torch.tensor(Utraintids),
                               torch.tensor(Utrain_inputs),
                               torch.tensor(Utrain_masks),
                               torch.tensor(Utrain_labels))

    Utrain_sampler = RandomSampler(Utrain_data)

    Utrain_dataloader = DataLoader(Utrain_data,
                                  sampler=Utrain_sampler,
                                  batch_size=batch_size)
    return Utrain_dataloader

def load_data(
        Utraindids, Utraintids, Utrain_inputs, Utrain_masks, Utrain_labels,
        Utestdids, Utesttids, Utest_inputs, Utest_masks, Utest_labels,
        traindids, traintids, train_inputs, train_masks, train_labels,
        testdids, testtids, test_inputs, test_masks, test_labels,
        batch_size=4
):

    return create_loader(Utraindids, Utraintids, Utrain_inputs, Utrain_masks, Utrain_labels, batch_size), \
            create_loader(Utestdids, Utesttids, Utest_inputs, Utest_masks, Utest_labels, batch_size), \
            create_loader(traindids, traintids, train_inputs, train_masks, train_labels, batch_size), \
            create_loader(testdids, testtids, test_inputs, test_masks, test_labels, batch_size)
    

def index_with_ids( dataset):
    a,b,c,d,e = dataset
    datasetmap = {}
    for i in range(len(a)):
        did,tid, origin_sentence, term, label = a[i], b[i], c[i], d[i], e[i]
        did = int(did)
        tid = int(tid)
        if did not in datasetmap: datasetmap[did]={}
        if tid not in datasetmap[did]: datasetmap[did][tid]=(origin_sentence, term, label)
    return datasetmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a classifier model for IF')
    parser.add_argument("--data_dir", default="./data/MAMS-ATSA/", type=str, required=True, help="The input data_dir")
    parser.add_argument("--train_data_name", default=None, type=str, required=True, help="The input training data name")
    parser.add_argument("--test_data_name", default=None, type=str, required=True, help="The input testing data name")
    parser.add_argument("--bert_name", default=None, type=str, required=False, help="The input testing data name")

    parser.add_argument("--stest_path", default=None, type=str, required=False, help="The input testing data name")
    parser.add_argument("--stest_mask_path", default=None, type=str, required=False, help="The input testing data name")

    parser.add_argument("--score_out_dir", default=None, type=str, required=True, help="specifies the name of model here")
    parser.add_argument("--log_file_name", default=None, type=str, required=True, help="The log file name")

    parser.add_argument("--num_classes", default=3, type=int, required=True,help="num of classes for the model")
    parser.add_argument("--ntest_start", default=-1, type=int, required=True,help="num of classes for the model")
    parser.add_argument("--ntest_end", default=-1, type=int, required=True,help="num of classes for the model")

    parser.add_argument("--test_delta", default="True", type=str, required=True, help="multiple by delta test (True) or by test (False)")

    parser.add_argument('--mode', type=str, help='the mode of influence function: IF, IF+, TC, TC+')

    args = parser.parse_args()

    # If two names does not equal, the init model won't match the loaded model.
    assert(BERT_name == args.bert_name)

    dr = DatasetReader(max_len=128, BERT_name=args.bert_name)
    # dr = DatasetReader(max_len=60, BERT_name=args.bert_name) # for sentihood

    traindids, traintids, trainx, trainm, trainl, traindata  = dr.read_data(
        args.data_dir + args.train_data_name[:-5] + '.json', mask_version=False, return_origin=True )

    testdids, testtids, testx,  testm,  testl, testdata = dr.read_data(
        args.data_dir + args.test_data_name + '.json', mask_version=False, return_origin=True )

    Mtraindids, Mtraintids, Mtrainx, Mtrainm, Mtrainl, Mtraindata = dr.read_data(
        args.data_dir + args.train_data_name[:-5] + '.json', mask_version=True, return_origin=True )

    Mtestdids, Mtesttids, Mtestx,  Mtestm,  Mtestl, Mtestdata  = dr.read_data(
        args.data_dir + args.test_data_name + '.json', mask_version=True, return_origin=True )

    Utrainloader, Utestloader, trainloader, testloader = load_data(
            traindids, traintids, trainx, trainm, trainl,
            testdids, testtids, testx, testm, testl,
            Mtraindids, Mtraintids, Mtrainx, Mtrainm, Mtrainl,
            Mtestdids, Mtesttids, Mtestx, Mtestm, Mtestl
            )
    config = {
            "outdir": args.score_out_dir, # IF result output dir.
            'stest_path': args.stest_path,
            'stest_mask_path': args.stest_mask_path,
            "seed": 42,
            "gpu": 0,
            "recursion_depth": 1000, # set recursion to use entire training data
            "r_averaging": 1,
            "scale": 1000,
            "damp": 0.01,
            "num_classes": args.num_classes,
            "log_filename": args.log_file_name,
        }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(BERT_Model, args.bert_name + '-' + args.train_data_name + '.out' , device)
    model.eval()

    # You can not mask it here because the gradient calculation won't work!!!!!!! 
    #Instead, you should obtain the entire parameter vector, and select 11, 10, pooler and classifier layer there.

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

    traindata = index_with_ids(traindata)
    testdata = index_with_ids(testdata)
    Mtraindata = index_with_ids(Mtraindata)
    Mtestdata = index_with_ids(Mtestdata)

    if 'MASK' in args.mode:
        ptif.calc_all_grad_mask(config, model, Utrainloader, Utestloader, trainloader, testloader, args.mode,
            traindata, testdata, Mtraindata, Mtestdata,
            args.ntest_start, args.ntest_end,
            args.test_delta=='True'
            )
    else:
        ptif.calc_all_grad(config, model, Utrainloader, Utestloader, trainloader, testloader, args.mode,
            traindata, testdata, Mtraindata, Mtestdata,
            args.ntest_start, args.ntest_end
            )

    
