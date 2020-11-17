import itertools
import random
import torch

from utils.utils_wikisql import load_wikisql, get_loader_wikisql



def get_data(path_wikisql, bs=16):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, toy_model=False, toy_size=10,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, bs, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader



def find_se(x, it):
    l = len(it)
    for i in range(len(x)):
        if x[i:i+l] == it:
            return i, i+l-1
    raise RuntimeError(f"Could not find '{it}' in {x}")



def generate_inputs(bert_tokenizer, records, table, padding_to=300, max_headers=30, max_selected=4, device="cuda"):
    PAD = 0
    CLS = 101
    SEP = 102
    EMPTY = 90 # [unused90]
    # import pdb; pdb.set_trace()
    # type_q
    # type_h_text
    # type_h_real
    # type_h_empty
    # token_types = 4
    header_type_map = {
        "text": 1,
        "real": 2,
        "empty": 3
    }

    TEXT = '[unused1]'
    REAL = '[unused4]'

    bs = len(records)
    
    x_input = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    attention_mask = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    token_type_ids = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    
    q_length = []
    h_length = []
    seps = []

    
    for b, rec in enumerate(records):
        tid = rec["table_id"]
        q = rec["question"]
        tb = table[tid]
        header = tb["header"]
        header_types = tb["types"]

        # + [EMPTY] col
        l_header = len(header) + 1

        q_tok = bert_tokenizer.tokenize(q, add_special_tokens=False)
        rec["q_tok"] = q_tok
        q_len = len(q_tok)

        q_length.append(q_len)
        h_length.append(l_header)

        hdr_enc = []
        for i,hdr in enumerate(header):
            he = bert_tokenizer.encode(hdr, add_special_tokens=False)
            h_type = header_type_map[header_types[i]]
            hdr_enc += [SEP] + [h_type] + he
        q_enc = bert_tokenizer.encode(q, add_special_tokens=False)
        # q_enc = [CLS] + q_enc
        x = [CLS] + q_enc + hdr_enc + [SEP, header_type_map["empty"], EMPTY, SEP]

        seq_len = len(x)
        # x += [PAD] * (padding_to - seq_len)
        x = torch.tensor(x, dtype=torch.long).to(device)
        x_input[b, :seq_len] = x
        attention_mask[b, :seq_len] = 1
        sep_i = (x == SEP).nonzero().squeeze(1).tolist()
        seps.append(sep_i)

        x_q_tok_len = sep_i[0]
        # token_type_id = [0] * x_q_tok_len + [1] * (seq_len - x_q_tok_len) + [0] * (padding_to - seq_len)
        token_type_ids[b, x_q_tok_len:seq_len] = 1

    # import pdb; pdb.set_trace()
    return (x_input, attention_mask, token_type_ids, q_length, h_length, seps)



# max_seq_len: 246
# max_headers: 24
# max_q_len: 86
def generate_samples(bert_tokenizer, records, table, padding_to=300, max_headers=30, max_selected=4, device="cuda"):
    PAD = 0
    CLS = 101
    SEP = 102
    EMPTY = 90 # [unused90]
    # import pdb; pdb.set_trace()
    # type_q
    # type_h_text
    # type_h_real
    # type_h_empty
    # token_types = 4
    header_type_map = {
        "text": 1,
        "real": 2,
        "empty": 3
    }

    TEXT = '[unused1]'
    REAL = '[unused4]'

    bs = len(records)
    
    x_input = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    # q_mask = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    # h_mask = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    attention_mask = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    token_type_ids = torch.zeros(bs, padding_to, dtype=torch.long).to(device)
    
    q_length = []
    h_length = []
    seps = []

    gs_num = []
    gs_col = []
    gs_agg = []
    gw_num = []
    gw_conn_op = []
    gw_col = []
    gw_op = []
    gw_val = []
    gw_val_match = []

    col_sel = []
    col_whr = []
    col_match = []

    gold = []

    g_cnt = 0
    
    for b, rec in enumerate(records):
        tid = rec["table_id"]
        q = rec["question"]
        sql = rec["sql"]
        agg = sql["agg"]
        sel = sql["sel"]
        if type(agg) != list:
            agg = [agg]
        if type(sel) != list:
            sel = [sel]
        conn_op = sql.get("cond_conn_op", 0)
        conds = sql["conds"]

        tb = table[tid]
        header = tb["header"]
        header_types = tb["types"]

        ####
        ls_num = len(sel)
        ls_col_agg = set([(sel[i], agg[i]) for i in range(len(sel))])
        lw_num = len(conds)
        lw_conn_op = (0 if conn_op < 2 else 1)
        lw_col = []# [cond[0] for cond in conds]
        lw_col_op = set([(cond[0], cond[1]) for cond in conds])
        lwvse = []
        lwvm = []
        ####


        # + [EMPTY] col
        l_header = len(header) + 1
        l_conds = len(conds)    # conds could be empty.
        s_num = len(sel)        # we always have selected column(s).

        # if s_num > 0:
        #     s_num -= 1
        # if l_conds > 0:
        #     l_conds -= 1
        s_num -= 1
        # l_conds -= 1

        q_tok = bert_tokenizer.tokenize(q, add_special_tokens=False)
        q_len = len(q_tok)
        rec["q_tok"] = q_tok

        cond_val_idx = set()

        w_col = [0] * l_header + [-1] * (max_headers - l_header)
        w_op = [-1] * max_headers
        w_col_val = [0] * q_len + [-1] * (150 - q_len)   # q_len == 150
        # w_col_vs = [-1] * max_headers
        # w_col_ve = [-1] * max_headers
        l_conds = len(conds)
        p_cond = (1 / l_conds if l_conds > 0 else 0)
        p_cond = 1
        w_val_not_found = False
        for i,cond in enumerate(conds):
            # if not lw_col:
            #     lw_col.append(cond[0])
            #     lw_op.append(cond[1])
            # else:
            #     if cond[0] != lw_col[-1]:
            #         lw_col.append(cond[0])
            #         lw_op.append(cond[1])
            # lw_col.append(cond[0])
            # lw_op.append((cond[0], cond[1]))
            w_col[cond[0]] = p_cond
            w_op[cond[0]] = cond[1]
            
            w_v = str(cond[2])
            # import pdb; pdb.set_trace()
            w_v_t = bert_tokenizer.tokenize(w_v)
            ws, we = find_se(['[CLS]'] + q_tok, w_v_t)

            cond.append([ws-1, we-1])
            lwvse.append((ws-1,we-1))
            
            for j in range(ws-1, we):
                w_col_val[j] = 1
            
            # w_col_vs[cond[0]] = ws-1
            # w_col_ve[cond[0]] = we-1
            
            cond_val_idx.add(ws-1)
            
            # w_val_match[cond_col_idx_map[cond[0]] * 4 + i] = 1
            # lwvs.append(ws-1)
            # lwve.append(we-1)
            # w_col_vs.append(ws-1)
            # w_col_ve.append(we-1)
        
        # cond_val_idx = list(set(cond_val_idx))
        idx_match = []
        cond_col_idx_map = dict([[cond[0], i] for i,cond in enumerate(conds)])
        cond_val_idx_map = dict([[idx, i] for i, idx in enumerate(sorted(cond_val_idx))])
        # w_val_match_len = len(cond_col_idx_map) * len(cond_val_idx_map)
        # w_val_match = [0] * w_val_match_len + [-1] * (16 - w_val_match_len) # max(w-num) == 4
        # w_val_match = [0] * (l_header * 4) + [-1] * (max_headers - l_header) * 4
        w_val_match = [-1] * max_headers * 4
        # w_val_match = [-1] * 16
        cond_c = cond_col_idx_map.keys()
        l_cond_c = len(cond_c)
        l_cond_v = len(cond_val_idx_map)
        # import pdb; pdb.set_trace()
        rv_pair = list(itertools.product(cond_c, range(l_cond_v)))
        for i,j in rv_pair:
            w_val_match[i * 4 + j] = 0
            idx_match.append([i * 4 + j, 0])

        # 构造负例
        # import pdb; pdb.set_trace()
        if l_cond_v < 2 or l_cond_c < 2:
            max_try = 4
            while max_try > 0:
                rand_col_idx = random.randint(0, l_header-1)
                rand_val_idx = random.randint(0, l_cond_v-1)
                if w_val_match[rand_col_idx * 4 + rand_val_idx] == -1:
                    w_val_match[rand_col_idx * 4 + rand_val_idx] = 0
                max_try -= 1
        for i,cond in enumerate(conds):
            ws, we = cond[-1]
            col_idx = cond[0]
            val_idx = cond_val_idx_map[ws]
            w_val_match[col_idx * 4 + val_idx] = 1
            lwvm.append(col_idx * 4 + val_idx)
            idx_match.append([col_idx * 4 + val_idx, 1])
        
        if l_conds == 0:
            w_col[l_header-1] = 1

        gw_col.append(w_col)
        # gw_val_s.append(w_col_vs)# + [0] * (max_selected - len(w_col_vs)))
        # gw_val_e.append(w_col_ve)# + [0] * (max_selected - len(w_col_ve)))
        gw_op.append(w_op)
        gw_val.append(w_col_val)
        gw_val_match.append(w_val_match)


        q_length.append(q_len)
        h_length.append(l_header)

        gs_num.append(s_num)
        gw_num.append(l_conds)
        gw_conn_op.append((0 if conn_op < 2 else 1))

        col_sel.append(sel[:])
        col_whr.append([item[0] for item in conds])
        col_match.append(idx_match)


        # s_col = [0] * l_header + [-1] * (max_headers - l_header)
        # s_col = [-1] * max_headers
        s_col = [0] * max_headers
        s_agg = [-1] * max_headers
        l_sel = len(sel)
        p_sel = (1 / l_sel if l_sel > 0 else 0)
        # p_sel = 1.
        for i,col in enumerate(sel):
            s_col[col] = p_sel
            s_agg[col] = agg[i]
        if l_sel == 0:
            s_col[l_header - 1] = 1.
        gs_col.append(s_col)
        gs_agg.append(s_agg)


        hdr_enc = []
        for i,hdr in enumerate(header):
            he = bert_tokenizer.encode(hdr, add_special_tokens=False)
            h_type = header_type_map[header_types[i]]
            hdr_enc += [SEP] + [h_type] + he
        q_enc = bert_tokenizer.encode(q, add_special_tokens=False)
        # q_enc = [CLS] + q_enc
        x = [CLS] + q_enc + hdr_enc + [SEP, header_type_map["empty"], EMPTY, SEP]
        # x = f"[XLS] {q} [SEP] {' [SEP] '.join(header)} [SEP]"
        # x = torch.tensor(bert_tokenizer.encode(x, add_special_tokens=False, max_length=padding_to), dtype=torch.long).to(device)
        # x_input[b, :x.size(-1)] = x
        # attention_mask[b, :x.size(-1)] = 1
        # sep_i = (x == SEP).nonzero().squeeze(1)
        # seps.append(sep_i)
        # q_mask[b, 1:sep_i[0]] = 1
        # h_mask[b, sep_i[0]+1:sep_i[-1]+1] = 1


        # x = bert_tokenizer.encode(x, add_special_tokens=False)
        # x += [header_type_map['empty'], EMPTY, SEP]
        seq_len = len(x)
        # x += [PAD] * (padding_to - seq_len)
        x = torch.tensor(x, dtype=torch.long).to(device)
        x_input[b, :seq_len] = x
        attention_mask[b, :seq_len] = 1
        sep_i = (x == SEP).nonzero().squeeze(1).tolist()
        seps.append(sep_i)
        # h_index = 0
        # for r,l in zip(sep_i[1:], sep_i[:-1]):
        #     h_type = (header_types[h_index] if h_index < len(header) else "empty")
        #     # h_len = r-l
        #     h_type_id = header_type_map[h_type]
        #     # token_type_ids += [head_type_map[h_type]] * h_len
        #     token_typd_ids[b, l+1:r+1] = h_type_id
        #     h_index += 1
        x_q_tok_len = sep_i[0]
        # token_type_id = [0] * x_q_tok_len + [1] * (seq_len - x_q_tok_len) + [0] * (padding_to - seq_len)
        token_type_ids[b, x_q_tok_len:seq_len] = 1
        # token_type_ids.append(token_type_id)

        lwvs = []
        lwve = []
        # ls_col_agg.sort()
        # lw_col = list(set(lw_col))
        # lw_op = list(set(lw_op))
        lwvse = set(lwvse)
        # lw_col.sort()
        # lw_op.sort()
        lwvm = set(lwvm)
        gold.append([ls_num, ls_col_agg, lw_num, lw_conn_op, lw_col_op, lwvse, lwvm])


    # import pdb; pdb.set_trace()
    return (x_input, attention_mask, token_type_ids, q_length, h_length, seps), \
        (gs_num, gs_col, gs_agg, gw_num, gw_conn_op, gw_col, gw_op, gw_val, gw_val_match), (col_sel, col_whr, col_match), gold