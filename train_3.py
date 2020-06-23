# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018


import os, sys, argparse, re, json
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
#import bert.tokenization as tokenization
#from bert.modeling import BertConfig, BertModel
from transformers import BertModel, BertConfig, BertTokenizer

from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
# from sqlova.model.nl2sql.nl2sql_models import *
from sqlnet.dbengine import DBEngine

from tqdm import tqdm
# from apex import amp
from model.dataset import *
from model.sql import *

# amp.register_float_function(torch, "sigmoid")
# amp.init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--half", default=False, action='store_true')
    parser.add_argument("--do_train", default=True, action='store_true')
    parser.add_argument('--do_infer', default=False, action='store_true')
    parser.add_argument('--infer_loop', default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')

    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=16, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=True,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=300, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-6, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=16,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='wwm-ext', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12',
                         'uB': 'bert-base-uncased',
                         'wwm-ext': 'hfl/chinese-bert-wwm-ext'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


# FIXME: use huggingface/transformers instead
#def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
#    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
#    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
#    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')
#
#    bert_config = BertConfig.from_json_file(bert_config_file)
#    tokenizer = tokenization.FullTokenizer(
#        vocab_file=vocab_file, do_lower_case=do_lower_case)
#    bert_config.print_status()
#
#    model_bert = BertModel(bert_config)
#    if no_pretraining:
#        pass
#    else:
#        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
#        print("Load pre-trained parameters.")
#    model_bert.to(device)
#
#    return model_bert, tokenizer, bert_config

# always use pretrained bert-base-uncased
def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    model_bert = BertModel.from_pretrained(bert_type, output_hidden_states=True).to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_type)
    bert_config = model_bert.config
    return model_bert, tokenizer, bert_config


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        # opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
        #                             lr=args.lr_bert, weight_decay=0)
        opt_bert = None
    else:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    # agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    # cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
    agg_ops = ['', 'avg', 'max', 'min', 'count', 'sum']
    cond_ops = ['>', '<', '==', '!=']
    cond_conn = ['', 'and', 'or']

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    # model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = SQLNet(model_bert)
    model = model.to(device)

    if trained:
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])
        # model = model.to(device)

    return model, model_bert, tokenizer, bert_config


def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train', half=False):
    model.train()
    # model_bert.train()

    ave_loss = 0
    cnt = 0  # count the # of examples
    # cnt_cc = 0
    cnt_sn = 0
    cnt_sc = 0  # count the # of correct predictions of select column
    # cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_w_conn = 0
    # cnt_wo = 0  # of where operator
    # cnt_wv = 0  # of where-value
    # cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_wv_s = 0
    cnt_wv_s_1 = 0
    cnt_wv_e = 0
    cnt_sc_r = 0
    cnt_wc_r = 0
    cnt_wv_s_r = 0
    cnt_wv_e_r = 0
    cnt_s_agg = 0
    cnt_w_op = 0
    cnt_lx = 0  # of logical form acc
    # cnt_x = 0  # of execution acc

    # Engine for SQL querying.
    # import pdb; pdb.set_trace()
    # engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    iters = 0
    for iB, t in tqdm(enumerate(train_loader)):
        # model.zero_grad()
        iters += 1
        if cnt < st_pos:
            continue
        # Get fields
        # nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # import pdb; pdb.set_trace()
        inputs, labels, col_sel, col_whr, col_match, gold = None, None, None, None, None, None
        try:
            # import pdb; pdb.set_trace()
            inputs, labels, (col_sel, col_whr, col_match), gold = generate_inputs(t, train_table, device=device)
            # import pdb; pdb.set_trace()
            cnt += len(t)
            # inputs, labels, (col_sel, col_whr) = load_inputs(t, train_table, device=device)
        except KeyboardInterrupt as e:
            sys.exit()
        except Exception as e:
            print(e)
            continue
            
        y_predict = model(inputs, labels)
        loss, (labels, (_, _)) = model.loss(y_predict, labels, iter=iters)# col_sel=col_sel, col_whr=col_whr, detail=True, records=t, table=train_table)

        sql_i = model.predict(y_predict, labels, q_length=inputs[3], h_length=inputs[4])
        detail = model.detail(y_predict, labels, col_sel=col_sel, col_whr=col_whr, col_match=col_match, q_length=inputs[3], records=t, table=train_table)

        opt.zero_grad()
        # opt_bert.zero_grad()
        loss.backward()
        opt.step()

        s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e = detail
        cnt_sn += s_num
        cnt_wn += w_num
        cnt_w_conn += w_conn_op
        cnt_sc += s_col
        cnt_s_agg += s_agg
        cnt_wc += w_col
        cnt_wv_e += w_val_e
        cnt_wv_s += w_val_s[0]
        cnt_wv_s_1 += w_val_s[1]
        cnt_w_op += w_op

        # if iters % 10000 == 0:
        #     import pdb; pdb.set_trace()

        for b, g in enumerate(gold):
            y = sql_i[b]
            sn, sca, wn, wconn, wco, wv, wvm = y
            gsn, gsca, gwn, gwconn, gwco, gwv, gwvm = g
            if sn == gsn and sca == gsca and wn == gwn and wconn == gwconn and wco == gwco and wvm == gwvm:
                cnt_lx += 1

        if iters % 500 == 0:
            # import pdb; pdb.set_trace()
            print("loss: ", loss, ave_loss / cnt)
            print(f"s_num: {cnt_sn/cnt:.3f}, s_col: {cnt_sc/cnt:.3f}, s_agg: {cnt_s_agg/cnt:.3f}\n"
                f"w_num: {cnt_wn/cnt:.3f}, w_cc: {cnt_w_conn/cnt:.3f}, w_op: {cnt_w_op/cnt:.3f}\n"
                f"w_col: {cnt_wc/cnt:.3f}, w_val_s: {cnt_wv_s/cnt:.3f}, w_val_e: {cnt_wv_e/cnt:.3f}\n"
                f"w_val_s_1: {cnt_wv_s_1/cnt}\n"
                f"lx: {cnt_lx/cnt:.3f}")

        # statistics
        ave_loss += loss.item()

    print("------TRAIN-------")
    print("loss:", ave_loss / cnt)
    print(f"s_num: {cnt_sn/cnt:.3f}, s_col: {cnt_sc/cnt:.3f}, s_agg: {cnt_s_agg/cnt:.3f}\n"
        f"w_num: {cnt_wn/cnt:.3f}, w_cc: {cnt_w_conn/cnt:.3f}, w_op: {cnt_w_op/cnt:.3f}\n"
        f"w_col: {cnt_wc/cnt:.3f}, w_val_s: {cnt_wv_s/cnt:.3f}, w_val_e: {cnt_wv_e/cnt:.3f}\n"
        f"lx: {cnt_lx/cnt:.3f}")
    return 0, 0


def report_detail(hds, nlu, g_cc,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans, pr_cc,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_cc, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_cc : {g_cc}')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_cc = {cnt_cc / cnt:.3f}\n'
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')


def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test', half=False):
    model.eval()
    
    ave_loss = 0
    cnt = 0  # count the # of examples
    # cnt_cc = 0
    cnt_sn = 0
    cnt_sc = 0  # count the # of correct predictions of select column
    # cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_w_conn = 0
    # cnt_wo = 0  # of where operator
    # cnt_wv = 0  # of where-value
    # cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_wv_s = 0
    cnt_wv_s_1 = 0
    cnt_wv_e = 0
    cnt_sc_r = 0
    cnt_wc_r = 0
    cnt_wv_s_r = 0
    cnt_wv_e_r = 0
    cnt_s_agg = 0
    cnt_w_op = 0
    cnt_lx = 0  # of logical form acc
    # cnt_x = 0  # of execution acc

    # Engine for SQL querying.
    # import pdb; pdb.set_trace()
    # engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    iters = 0
    for iB, t in tqdm(enumerate(data_loader)):
        # model.zero_grad()
        iters += 1
        if cnt < st_pos:
            continue
        inputs, labels, col_sel, col_whr, gold = None, None, None, None, None
        try:
            inputs, labels, (col_sel, col_whr, col_match), gold = generate_inputs(t, data_table, device=device)
            cnt += len(t)
            # inputs, labels, (col_sel, col_whr) = load_inputs(t, train_table, device=device)
        except KeyboardInterrupt as e:
            sys.exit()
        except Exception as e:
            print(iters, e)
            sys.exit()
            # continue
        y_predict = model(inputs, label=None, records=t)
        loss, (labels, (_, _)) = model.loss(y_predict, labels, records=t)# col_sel=col_sel, col_whr=col_whr, detail=True, records=t, table=train_table)
        # gwvse = [g[-2] for g in gold]
        sql_i = model.predict(y_predict, labels, q_length=inputs[3], h_length=inputs[4], gwvse=None)
        detail = model.detail(y_predict, labels, col_sel=col_sel, col_whr=col_whr, col_match=col_match, q_length=inputs[3], records=t, table=train_table)

        s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e = detail
        cnt_sn += s_num
        cnt_wn += w_num
        cnt_w_conn += w_conn_op
        cnt_sc += s_col
        cnt_s_agg += s_agg
        cnt_wc += w_col
        cnt_wv_e += w_val_e
        cnt_wv_s += w_val_s[0]
        cnt_wv_s_1 += w_val_s[1]
        cnt_w_op += w_op

        # if iters % 10 == 0:
        #     import pdb; pdb.set_trace()
        for b, g in enumerate(gold):
            y = sql_i[b]
            sn, sca, wn, wconn, wco, wv, wvm = y
            gsn, gsca, gwn, gwconn, gwco, gwv, gwvm = g
            if sn == gsn and sca == gsca and wn == gwn and wconn == gwconn and wco == gwco and wvm == gwvm:
                cnt_lx += 1

        if iters % 1000 == 0:
            # import pdb; pdb.set_trace()
            print("loss: ", loss, ave_loss / cnt)
            print(f"s_num: {cnt_sn/cnt:.3f}, s_col: {cnt_sc/cnt:.3f}, s_agg: {cnt_s_agg/cnt:.3f}\n"
                f"w_num: {cnt_wn/cnt:.3f}, w_cc: {cnt_w_conn/cnt:.3f}, w_op: {cnt_w_op/cnt:.3f}\n"
                f"w_col: {cnt_wc/cnt:.3f}, w_val_s: {cnt_wv_s/cnt:.3f}, w_val_e: {cnt_wv_e/cnt:.3f}\n"
                f"w_val_s_1: {cnt_wv_s_1/cnt}\n"
                f"lx: {cnt_lx/cnt:.3f}")

        # statistics
        ave_loss += loss.item()

    print("------TEST-------")
    print(f"s_num: {cnt_sn/cnt:.3f}, s_col: {cnt_sc/cnt:.3f}, s_agg: {cnt_s_agg/cnt:.3f}\n"
        f"w_num: {cnt_wn/cnt:.3f}, w_cc: {cnt_w_conn/cnt:.3f}, w_op: {cnt_w_op/cnt:.3f}\n"
        f"w_col: {cnt_wc/cnt:.3f}, w_val_s: {cnt_wv_s/cnt:.3f}, w_val_e: {cnt_wv_e/cnt:.3f}\n"
        f"w_val_s_1: {cnt_wv_s_1/cnt}\n"
        f"lx: {cnt_lx/cnt:.3f}")
    return 0, 0, 0 #acc, results, cnt_list


def tokenize_corenlp(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1):
        for tok in sentence:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def infer(nlu1,
          table_name, data_table, path_db, db_name,
          model, model_bert, bert_config, max_seq_length, num_target_layers,
          beam_size=4, show_table=False, show_answer_only=False):
    # I know it is of against the DRY principle but to minimize the risk of introducing bug w, the infer function introuced.
    model.eval()
    model_bert.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    # Get inputs
    nlu = [nlu1]
    # nlu_t1 = tokenize_corenlp(client, nlu1)
    nlu_t1 = tokenize_corenlp_direct_version(client, nlu1)
    nlu_t = [nlu_t1]

    tb1 = data_table[0]
    hds1 = tb1['header']
    tb = [tb1]
    hds = [hds1]
    hs_t = [[]]

    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, engine, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size)

    # sort and generate
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    if len(pr_sql_i) != 1:
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])
    pr_sql_q = [pr_sql_q1]

    try:
        pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
    except:
        pr_ans = ['Answer not found.']
        pr_sql_q = ['Answer not found.']

    if show_answer_only:
        print(f'Q: {nlu[0]}')
        print(f'A: {pr_ans[0]}')
        print(f'SQL: {pr_sql_q}')

    else:
        print(f'START ============================================================= ')
        print(f'{hds}')
        if show_table:
            print(engine.show_table(table_name))
        print(f'nlu: {nlu}')
        print(f'pr_sql_i : {pr_sql_i}')
        print(f'pr_sql_q : {pr_sql_q}')
        print(f'pr_ans: {pr_ans}')
        print(f'---------------------------------------------------------------------')

    return pr_sql_i, pr_ans


def print_result(epoch, acc, dname):
    ave_loss, acc_cc, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss:.3f}, acc_cc: {acc_cc:.3f}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './data_and_model'  # '/home/wonseok'
    path_wikisql = './data_and_model/nl2sql'  # os.path.join(path_h, 'data', 'wikisql_tok')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data

    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    args.trained=True
    if not args.trained:
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)
    else:
        # To start from the pre-trained models, un-comment following lines.
        path_model_bert = './data_and_model/model_bert_best.pt'
        path_model = './data_and_model/model_26.pt'
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                               path_model_bert=path_model_bert, path_model=path_model)

    ## 4.5. Use fp16 for training
    #if args.half:
    #    model = model.half()
    #    model_bert = model_bert.half()

    ## 5. Get optimizers
    if args.do_train:
        opt, opt_bert = get_opt(model, model_bert, args.fine_tune)
        
        # 5.5 Use Apex to accelerate training
        if args.half:
            model, opt = amp.initialize(model, opt)
            model_bert, opt_bert = amp.initialize(model_bert, opt_bert)

        ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1
        for epoch in range(args.tepoch):
            train_loader = torch.utils.data.DataLoader(
                batch_size=args.bS,
                dataset=train_data,
                shuffle=True,
                num_workers=4,
                collate_fn=lambda x: x  # now dictionary values are not merged!
            )
            # train
            # acc_train, aux_out_train = train(train_loader,
            #                                  train_table,
            #                                  model,
            #                                  model_bert,
            #                                  opt,
            #                                  bert_config,
            #                                  tokenizer,
            #                                  args.max_seq_length,
            #                                  args.num_target_layers,
            #                                  args.accumulate_gradients,
            #                                  opt_bert=opt_bert,
            #                                  st_pos=0,
            #                                  path_db=path_wikisql,
            #                                  dset_name='train',
            #                                  half=False)

            # check DEV
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test(dev_loader,
                                                      dev_table,
                                                      model,
                                                      model_bert,
                                                      bert_config,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.num_target_layers,
                                                      detail=False,
                                                      path_db=path_wikisql,
                                                      st_pos=0,
                                                      dset_name='dev', EG=args.EG,
                                                      half=args.half)

            # print_result(epoch, acc_train, 'train')
            # print_result(epoch, acc_dev, 'dev')

            # # # save results for the official evaluation
            # save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

            # save model
            # state = {'model': model.state_dict()}
            # torch.save(state, os.path.join('data_and_model', f'model_{epoch}.pt'))

            # state = {'model_bert': model_bert.state_dict()}
            # torch.save(state, os.path.join('.', 'model_bert_best.pt'))

            # # # save best model
            # # # Based on Dev Set logical accuracy lx
            # acc_lx_t = acc_dev[-2]
            # if acc_lx_t > acc_lx_t_best:
            #     acc_lx_t_best = acc_lx_t
            #     epoch_best = epoch
            #     # save best model
            #     state = {'model': model.state_dict()}
            #     torch.save(state, os.path.join('.', 'model_best.pt'))

            #     state = {'model_bert': model_bert.state_dict()}
            #     torch.save(state, os.path.join('.', 'model_bert_best.pt'))

            # print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")

    if args.do_infer:
        # To use recent corenlp: https://github.com/stanfordnlp/python-stanford-corenlp
        # 1. pip install stanford-corenlp
        # 2. download java crsion
        # 3. export CORENLP_HOME=/Users/wonseok/utils/stanford-corenlp-full-2018-10-05

        # from stanza.nlp.corenlp import CoreNLPClient
        # client = CoreNLPClient(server='http://localhost:9000', default_annotators='ssplit,tokenize'.split(','))

        from stanza.server import CoreNLPClient

        client = CoreNLPClient(annotators='ssplit,tokenize'.split(','))

        nlu1 = "Which company have more than 100 employees?"
        path_db = './data_and_model'
        db_name = 'ctable'
        data_table = load_jsonl('./data_and_model/ctable.tables.jsonl')
        table_name = 'ftable1'
        n_Q = 100000 if args.infer_loop else 1
        for i in range(n_Q):
            if n_Q > 1:
                nlu1 = input('Type question: ')
            pr_sql_i, pr_ans = infer(
                nlu1,
                table_name, data_table, path_db, db_name,
                model, model_bert, bert_config, max_seq_length=args.max_seq_length,
                num_target_layers=args.num_target_layers,
                beam_size=1, show_table=False, show_answer_only=False
            )
