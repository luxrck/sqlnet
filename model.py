import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel, BertTokenizer


def isreal(t):
    try:
        float(t)
        return True
    except:
        return False


class Attention(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self, Q, K, V, seps=None, scale=0, q_mask=None, k_mask=None):
        assert scale >= 0

        if scale == 0:
            scale = math.sqrt(Q.size(-1))
        # import pdb; pdb.set_trace()
        e = Q.bmm(K.transpose(-1, -2)) / scale
        
        if not seps:
            masked = -2**14 if Q.dtype == torch.float16 else -2**31
            if k_mask is not None:
                # k_mask: [bs, k_len] -> [bs, 1, k_len]
                k_mask = k_mask.unsqueeze(-2)
                e.masked_fill_(k_mask == 0, masked)
            a = F.softmax(e, dim=-1)
            a = self.dropout(a)
            a = a.bmm(V)
            if q_mask is not None:
                # q_mask: [bs, q_len] -> [bs, .. , q_len]
                q_mask = q_mask.expand(a.shape[:-1])
                a[q_mask == 0] = 0.
            return a
        # import pdb; pdb.set_trace()

        bs = Q.size(0)
        h_max = max([len(s) for s in seps]) - 1
        # h_seq_max = max([max([a-b-1 for (a,b) in zip(sep[1:], sep[:-1])]) for sep in seps])
        y = torch.zeros(bs, h_max, V.size(-1)).to(Q.device)
        y_mask = torch.zeros(bs, h_max).to(Q.device)
        
        for b in range(bs):
            h_i = 0
            for rs_i, ls_i in zip(seps[b][1:], seps[b][:-1]):
                ls_i += 1
                h_len = rs_i - ls_i
                e1 = e[b, :, ls_i:rs_i] #Q[b,:].bmm(K[b, ls_i:rs_i, :].transpose(-1, -2))
                a = F.softmax(e1, dim=-1)
                a = self.dropout(a)
                y[b, h_i:h_i+1, :] = a.mm(V[b, ls_i:rs_i, :])
                h_i += 1
            y_mask[b, :h_i] = 1
        return y, y_mask



class LinearLayer(nn.Module):
    def __init__(self, d_hidden, d_out, fc_activation=nn.ReLU(), dropout_p=0.1):
        super().__init__()
        self.proj_a = nn.Linear(d_hidden, d_hidden)
        self.proj_b = nn.Linear(d_hidden, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        # self.out = nn.Linear(d_hidden, d_out)
        self.out = nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(p=dropout_p),
                    fc_activation,
                    nn.Linear(d_hidden, d_out))

    # a: [bs, a, dim]
    # b: [bs, b, dim]
    # return: [bs, a, b, dim] -> [bs, a, b, d_out]
    def forward(self, a, b=None, output_hidden=False, norm=True):
        if b is None:
            return self.out(a)
        
        a_len = a.size(1)
        b_len = b.size(1)

        a = self.proj_a(a)
        b = self.proj_b(b)

        x = a.unsqueeze(2).expand(-1, -1, b_len, -1) + b.unsqueeze(1).expand(-1, a_len, -1, -1)

        # x = cls_enc.expand(-1, h_max, -1) + h_enc
        if norm:
            x = self.norm(x)
        if output_hidden:
            return x
        x = self.out(x)
        return x



# Multi column / Multi Aggs
class SQLNet(nn.Module):
    COND_OP = {0: "NOP", 1: ">", 2: "<", 3: "==", 4: "!="}
    COND_CONN_OP = {0: "NOP", 1: "and", 2: "or"}
    AGG_OP = {0: "NOP", 1: "", 2: "AVG", 3: "MAX", 4: "MIN", 5: "COUNT", 6: "SUM"}

    max_q_len = 150
    max_headers = 30
    max_sel = 4
    max_whr = 4
    def __init__(self, bert, dropout_p=0.1):
        super().__init__()
        self.bert = bert

        self.d_hidden = self.bert.config.hidden_size

        self.select_num = LinearLayer(self.d_hidden, 4, dropout_p=dropout_p)
        # [not_selected, agg0, agg1, ... , agg5]
        self.select_column = LinearLayer(self.d_hidden, 1, dropout_p=dropout_p)
        self.select_agg = LinearLayer(self.d_hidden, 6, dropout_p=dropout_p)
        # self.select_agg = nn.Linear(self.bert.config.hidden_size, 1)

        # [0: and, 1: or]
        self.where_conn_op = LinearLayer(self.d_hidden, 2, dropout_p=dropout_p)
        # where clause could be none
        self.where_num = LinearLayer(self.d_hidden, 4+1, dropout_p=dropout_p)
        # [not_selected, col_on_op1, ... , col_on_op4]
        
        self.where_column = LinearLayer(self.d_hidden, 1, dropout_p=dropout_p)
        self.where_op = LinearLayer(self.d_hidden, 4, dropout_p=dropout_p)

        # per column
        self.where_val_s_rh = LinearLayer(self.d_hidden, self.d_hidden)
        self.where_val_e_rh = LinearLayer(self.d_hidden, self.d_hidden)
        self.where_val_s = LinearLayer(self.d_hidden, 1)
        self.where_val_e = LinearLayer(self.d_hidden, 1)

        # POS Tag: 0:not val.  1: val.
        self.where_val = LinearLayer(self.d_hidden, 2, dropout_p=dropout_p)
        # self.where_val = 
        # self.where_val = nn.Linear(self.d_hidden, 2)

        self.where_match = LinearLayer(self.d_hidden, 1, dropout_p=dropout_p)#, fc_activation=nn.Tanh())
        # self.where_match = Matcher(self.d_hidden)

        self.attention = Attention(dropout_p=dropout_p)

        # self.init_weights()

    # (x_input, q_mask, h_mask, attention_mask, seps)
    # x: [batch_size, max_seq_len]
    def forward(self, inputs, gw_val=None):
        # import pdb; pdb.set_trace()
        x, attention_mask, token_type_ids, q_length, h_length, seps = inputs

        # gs_num, gs_col, gs_agg, gw_num, gw_conn_op, gw_col, gw_op, gw_val_s, gw_val_e = (None,) * 9
        # if label:
        #     gs_num, gs_col, gs_agg, gw_num, gw_conn_op, gw_col, gw_op, gw_val_s, gw_val_e = label

        bs = x.size(0)
        q_len = max(q_length)
        h_cnt = max(h_length)
        dim = self.d_hidden
        
        emb, pooled_out, all_emb = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)

        device = emb.device
        
        cls_emb = emb[:, 0, :].unsqueeze(1)
        # q_emb: [bs, q_len, dim]
        # h_emb: [bs, h_cnt, dim]
        # TODO: calc q_emb/h_emb
        # q_emb = emb.index_select(1, torch.arange(seps[0] + 1))
        # h_emb = emb.index_select(1, torch.arange(seps[0] + 1, seps[-1] + 1))
        
        # q_enc: [bs, q_len, dim]
        # h_enc: [bs, h_cnt, dim]
        # h_enc_mask: [bs, h_cnt]
        # h_enc_q: [bs, h_cnt, dim]
        # q_enc_h: [bs, q_len, dim]
        # import pdb; pdb.set_trace()
        
        masked = -2**14 if emb.dtype == torch.float16 else -2**31
        # q_mask = (torch.arange(q_len).expand(bs, q_len) < torch.tensor(q_length).view(bs, -1).expand(bs, q_len)).type(torch.long).to(emb.device)
        q_enc = emb[:, 1:q_len+1, :]#.detach().clone()
        # q_enc[q_mask == 0] = 0.
        
        h_enc, h_enc_mask = self.attention(cls_emb, emb, emb, seps=seps)
        # h_enc_q = self.attention(h_enc, q_enc, q_enc, seps=None, q_mask=None, k_mask=q_mask)
        # q_enc_h = self.attention(q_enc, h_enc, h_enc, q_mask=None, k_mask=h_enc_mask)

        # [bs, 1, dim]
        # cls_enc = self.attention(cls_emb, h_enc, h_enc, q_mask=None, k_mask=h_enc_mask)
        # cls_enc_emb = torch.cat([cls_enc, cls_emb], dim=2).squeeze(1)

        # s_num = self.select_num(cls_enc_emb)
        # [bs, h_cnt, dim] -> [bs, h_cnt, 1, dim]
        # [bs, 1, 4] -> [bs, 4]
        s_num = self.select_num(cls_emb, None).squeeze(1)
        w_num = self.where_num(cls_emb, None).squeeze(1)
        # [bs, 1, 2] -> [bs, 2]
        w_conn_op = self.where_conn_op(cls_emb, None).squeeze(1)
        # w_conn_op = self.where_conn_op(cls_enc_emb)
        
        # cls_enc_emb = cls_enc_emb.unsqueeze(1)

        # [bs, h_cnt, dim * 3]
        # import pdb; pdb.set_trace()
        # h_enc_q_cls = torch.cat([h_enc, h_enc_q, cls_enc_emb.detach().clone().expand(bs, h_cnt, cls_enc_emb.shape[-1])], dim=2)    # [bs, h_cnt, dim*4]
        # h_enc_q_cls = torch.cat([h_enc, h_enc_q, cls_emb.expand(-1, h_enc.shape[1], -1)], dim=2)    # [bs, h_cnt, dim*2]

        # [bs, h_cnt, 1, 1] -> [bs, h_cnt, 1]
        s_col = self.select_column(h_enc, cls_emb).squeeze(-1)
        w_col = self.where_column(h_enc, cls_emb).squeeze(-1)

        # s_test = s_col[0][0].item()
        # w_test = w_col[0][0].item()
        # if math.isnan(w_test) or math.isnan(s_test):
        #     import pdb; pdb.set_trace()

        # s_col.masked_fill_(h_enc_mask == 0, masked)
        # w_col.masked_fill_(h_enc_mask == 0, masked)
        s_col[h_enc_mask == 0] = masked
        w_col[h_enc_mask == 0] = masked

        # [bs, h_cnt, 1, 6] -> [bs, h_cnt, 6]
        s_agg = self.select_agg(h_enc, cls_emb).squeeze(-2)
        # [bs, h_cnt, 1, 4] -> [bs, h_cnt, 4]
        w_op = self.where_op(h_enc, cls_emb).squeeze(-2)

        # s_agg.masked_fill_(h_enc_mask == 0, masked)
        # w_op.masked_fill_(h_enc_mask == 0, masked)
        # s_agg[h_enc_mask == 0] = masked
        # w_op[h_enc_mask == 0] = masked

        # [bs, dim]
        # q_enc_h = self.attention(q_enc.unsqueeze(1), h_enc, h_enc, seps=None, mask=h_enc_mask.unsqueeze(1)).squeeze(1)
        # h_enc = torch.cat([h_enc, h_enc_q], dim=2)    # [bs, q_len, dim*2]

        # [bs, q_len, dim * 4]
        # q_enc_h_cls = torch.cat([q_enc, h_enc_q, cls_enc_emb.detach().clone().expand(bs, q_len, cls_enc_emb.shape[-1])], dim=2)
        
        # [bs, w_col(4), q_len, dim * 3]
        # q_enc_h_enc_q = torch.zeros(bs, 4, q_len, dim * 3).to(emb.device)
        # for b in range(bs):
        #     gw_col_b = gw_col[b]
        #     gw_cnt_b = 0
        #     # import pdb; pdb.set_trace()
        #     for j, selected in enumerate(gw_col_b):
        #         if selected == 0:
        #             continue
        #         q_enc_h_enc_q[b, gw_cnt_b, :, :] = torch.cat([q_enc[b, :, :], h_enc[b, j:j+1, :].expand(q_enc.shape[1], -1), h_enc_q[b, j:j+1, :].expand(q_enc.shape[1], -1)], dim=-1)
        #         gw_cnt_b += 1
        #         if gw_cnt_b >= gw_num[b] + 1:
        #             break
                # q_enc_h_enc_q_b.append(q_enc_h_enc_q_b_j)
        # import pdb; pdb.set_trace()

        # [bs, h_cnt, 1, dim] -> [bs, h_cnt, dim]
        # not pass `out` layer
        # wvs_rh = self.where_val_s_rh(h_enc, cls_emb, output_hidden=True).squeeze(-2)
        # wve_rh = self.where_val_e_rh(h_enc, cls_emb, output_hidden=True).squeeze(-2)
        # # [bs, h_cnt, q_len, dim] -> [bs, h_cnt, q_len, 1]
        # w_val_s = self.where_val_s(wvs_rh, q_enc)
        # w_val_e = self.where_val_e(wve_rh, q_enc)

        # w_val_s.masked_fill_(h_enc_mask == 0, masked)
        # w_val_e.masked_fill_(h_enc_mask == 0, masked)

        # [bs, q_len, 1, dim] -> [bs, q_len, 1, 2]
        # w_val = self.where_val(q_enc, cls_emb, norm=False).squeeze(-2)
        w_val = self.where_val(q_enc, cls_emb).squeeze(-2)

        # gw_val = label[-2]
        # gw_col = label[5]
        # k = 4 if bs > 4 else bs
        # import pdb; pdb.set_trace()
        # _, w_col_idx_ = (w_col.squeeze(-1).topk(k) if label is None else torch.tensor(label[5]).to(device)[:, :h_cnt].topk(k))
        
        # w_col_ = []
        # w_col_idx_, _ = w_col_idx_.sort()
        # for b in range(bs):
        #     col = h_enc[b, w_col_idx_[b]]
        #     w_col_.append(col)
        # w_col_ = torch.stack(w_col_, dim=0)

        # w_col_ = (w_col.squeeze(-1) if label is None else torch.tensor(label[5]).to(device)[:, :h_cnt])
        
        w_val_ = (w_val.argmax(-1) if gw_val is None else torch.tensor(gw_val).to(device)[:, :q_len])
        v_enc, w_val_se = self.w_val(q_enc, w_val_, q_length=q_length)
        # import pdb; pdb.set_trace()
        # [bs, h_cnt, v_cnt, dim] -> [bs, h_cnt, v_cnt]
        w_val_match = self.where_match(h_enc, v_enc).squeeze(-1)

        # s_num = None
        # w_conn_op = None
        w_val_s = None
        w_val_e = None
        # w_val_match = None
        # w_val_se = None
        return s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e, w_val, w_val_match, w_val_se
    
    def w_val(self, q_enc, w_val, q_length):
        bs = q_enc.size(0)
        dim = q_enc.size(2)
        v_enc = torch.zeros(bs, 4, dim).to(q_enc.device)
        # v_cnt = []
        w_val_se = []
        # if is_train:
        #     w_val = w_val.argmax(dim=-1)    # [bs, q_len]
        for b, val in enumerate(w_val):
            start = False
            start_idx = -1
            prev_flag = False
            v_idx = 0
            v_s_ = []
            for j, flag in enumerate(val):
                if v_idx > 3:
                    break
                if j >= q_length[b] or flag == -1:
                    j -= 1
                    break
                if flag == 1:
                    if not start:
                        start = True
                        start_idx = j
                        # v_s_.append([start_idx, j-1])
                elif prev_flag == 1:
                    v_s_.append((start_idx, j-1))
                    v_enc[b, v_idx] = q_enc[b, start_idx:j].mean(0)
                    v_idx += 1
                    start = False
                    start_idx = -1
                prev_flag = flag
            if start:
                v_enc[b, v_idx] = q_enc[b, start_idx:j+1].mean(0)
                v_s_.append((start_idx, j))
            w_val_se.append(v_s_)
        return v_enc, w_val_se
    
    def loss(self, y_predict, labels):
        s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e, w_val, w_val_match, w_val_se = y_predict
        gs_num, gs_col, gs_agg, gw_num, gw_conn_op, gw_col, gw_op, gw_val, gw_val_match = labels
        # import pdb; pdb.set_trace()
        device = s_col.device
        bs = s_col.size(0)
        h_cnt = s_col.size(1)
        q_len = w_val.size(-2)
        
        gs_num = torch.tensor(gs_num).to(device).view(-1)
        gs_col = torch.tensor(gs_col).to(device)[:, :h_cnt]#.type(s_col.dtype)#.contiguous().view(-1)
        gs_agg = torch.tensor(gs_agg).to(device)[:, :h_cnt]
        gw_num = torch.tensor(gw_num).to(device).view(-1)
        gw_conn_op = torch.tensor(gw_conn_op).to(device).view(-1)
        gw_col = torch.tensor(gw_col).to(device)[:, :h_cnt].type(torch.float32)#.contiguous().view(-1)
        gw_op = torch.tensor(gw_op).to(device)[:, :h_cnt]
        # gw_val_s = torch.tensor(gw_val_s).to(device)[:, :h_cnt]#.contiguous().view(-1)
        # gw_val_e = torch.tensor(gw_val_e).to(device)[:, :h_cnt]#.contiguous().view(-1)
        gw_val = torch.tensor(gw_val).to(device)[:, :q_len]
        gw_val_match = torch.tensor(gw_val_match).to(device)[:, :h_cnt*4].type(torch.float32)    # max conds is 4
        
        # import pdb; pdb.set_trace()

        # def loss_col(y, g, g_length):
        #     loss = 0
        #     for b, yb in enumerate(y):
        #         g_cnt = g_length[b]
        #         if g_cnt <= 0:
        #             continue
        #         loss += F.binary_cross_entropy_with_logits(yb[:g_cnt, :], g[b, :g_cnt])
        #     return loss

        loss = 0
        # import pdb; pdb.set_trace()
        
        # loss_sn = 0
        loss_sn = F.cross_entropy(s_num, gs_num.view(-1), ignore_index=-1)
       
        # loss_sc = F.cross_entropy(s_col.view(-1, s_col.shape[-2]), gs_col.contiguous().argmax(dim=1).view(-1), ignore_index=-1)
        loss_sc = torch.abs(F.kl_div(s_col.squeeze(-1).log_softmax(dim=-1), gs_col, reduction="batchmean"))
        # loss_sc = F.binary_cross_entropy_with_logits(s_col.squeeze(-1), gs_col)
        
        loss_sa = F.cross_entropy(s_agg.view(-1, s_agg.shape[-1]), gs_agg.contiguous().view(-1), ignore_index=-1)
        
        loss_wn = F.cross_entropy(w_num, gw_num.view(-1), ignore_index=-1)

        # loss_co = 0
        loss_co = F.cross_entropy(w_conn_op, gw_conn_op.view(-1), ignore_index=-1)
        
        # loss_wc = F.binary_cross_entropy_with_logits(w_col.squeeze(-1), gw_col)
        # loss_wc = torch.abs(F.kl_div(w_col.squeeze(-1).log_softmax(dim=-1), gw_col, reduction="batchmean"))
        
        loss_wo = F.cross_entropy(w_op.view(-1, w_op.shape[-1]), gw_op.contiguous().view(-1), ignore_index=-1)

        # loss_ws = F.cross_entropy(w_val_s.view(-1, q_len), gw_val_s.contiguous().view(-1), ignore_index=-1)
        
        # loss_we = F.cross_entropy(w_val_e.view(-1, q_len), gw_val_e.contiguous().view(-1), ignore_index=-1)

        # import pdb; pdb.set_trace()
        loss_wv = F.cross_entropy(w_val.view(-1, 2), gw_val.contiguous().view(-1), ignore_index=-1)
        
        w_col_idx = gw_col > -1
        # w_col_pos_weight = gw_col == 1
        w_col = w_col.view(bs, -1)
        # loss_wc = F.binary_cross_entropy_with_logits(w_col[w_col_idx], gw_col[w_col_idx])

        wv_match_idx = gw_val_match > -1
        w_val_match = w_val_match.view(bs, -1)
        # loss_wv_match = F.binary_cross_entropy_with_logits(w_val_match[wv_match_idx], gw_val_match[wv_match_idx])
        loss_wc = 0
        loss_wv_match = 0
        for b in range(bs):
            # import pdb; pdb.set_trace()
            lbl_w_col = gw_col[b, w_col_idx[b]]
            loss_wc += F.binary_cross_entropy_with_logits(w_col[b, w_col_idx[b]], lbl_w_col)
            lbl_wv_match = gw_val_match[b, wv_match_idx[b]]
            loss_wv_match += F.binary_cross_entropy_with_logits(w_val_match[b, wv_match_idx[b]], lbl_wv_match)
        loss_wc /= bs
        loss_wv_match /= bs# w_val_match_y_predict_cnt
        # loss_wv_match = 1
        # w_val_match_ = w_val_match.view(bs, -1)
        # for b in range(bs):
        #     for j in range(h_cnt):
        #         if gw_val_match[b, j].item() == -1:
        #             continue
        #         w_val_match_cnt += 1
        #         loss_wv_match += F.smooth_l1_loss(w_val_match_[b, j].sigmoid(), gw_val_match[b, j])
        # loss_wv_match = F.smooth_l1_loss(w_val_match.view(bs, -1).sigmoid(), gw_val_match)
        # import pdb; pdb.set_trace()
        # loss_wv_match = F.cross_entropy(w_val_match.view(-1, 2), gw_val_match.contiguous().view(-1), ignore_index=-1)
        # loss_wv_match = 0
        # if iter and iter % 500 == 0:
        #     print(loss_wc.item(), loss_wv.item(), loss_wv_match.item())
        # if math.isnan(loss_wv_match.item()):
        #     import pdb; pdb.set_trace()
        
        loss = loss_sn + loss_sc + loss_sa + loss_co + loss_wn + loss_wc + loss_wo + loss_wv + loss_wv_match

        labels = (gs_num, gs_col, gs_agg, gw_num, gw_conn_op, gw_col, gw_op, gw_val, gw_val_match)

        return loss, labels
    
    def predict(self, y_predict, h_length=None, gwvse=None, records=[], tables={}):
        s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e, w_val, w_val_match, w_val_se = y_predict

        bs = s_col.size(0)
        h_cnt = s_col.size(1)

        s_num = s_num.argmax(dim=1)
        w_num = w_num.argmax(dim=1)
        w_conn_op = w_conn_op.argmax(dim=1)

        topk = 4
        # _, s_col = s_col.squeeze(-1).topk(topk)
        s_col = s_col.squeeze(-1)
        s_agg = s_agg.argmax(dim=-1)
        w_col = w_col.squeeze(-1)
        # _, w_col = w_col.squeeze(-1).topk(topk)
        # w_op = w_op.argmax(dim=-1)
        # w_val_s = w_val_s.squeeze(-1).argmax(dim=-1)
        # w_val_e = w_val_e.squeeze(-1).argmax(dim=-1)
        w_val_match = w_val_match.view(bs, -1)
        # _, w_val_match = w_val_match.topk(topk)

        op_blacklist = {
            "text": set([0, 1]),
        }

        op_pattern = [
            r"(之前|以前|前面|小于|(前\d+)|以内|之内|低于|少于|不超过|不大于|不多于)",
            r"(之后|以后|后面|大于|超过|高于|高出|高过|超出|不少于|不小于)",
        ]

        result = []

        for b in range(bs):
            topk = 3
            q = records[b]["question"]
            q_tok = records[b]["q_tok"]
            table_id = records[b]["table_id"]
            header_types = tables[table_id]["types"]
            # import pdb; pdb.set_trace()
            # label starts from 0
            ps_num_b = s_num[b].item() + 1
            if topk > h_length[b] - 1:
                topk = h_length[b]-1
            _, ps_col_b = s_col[b, :h_length[b]-1].topk(topk)
            ps_col_b = ps_col_b[:ps_num_b].tolist()

                # import pdb; pdb.set_trace()

            ps_col_agg_b = []
            for col in ps_col_b:
                agg = s_agg[b, col].item()
                ps_col_agg_b.append((col, agg))
            pw_conn_op_b = w_conn_op[b].item()

            pw_num_b = w_num[b].item()

            if pw_num_b == 0:
                pw_num_b = 1
                result.append([ps_num_b, set(ps_col_agg_b), pw_num_b, pw_conn_op_b, set(), set(), set()])
                continue

            # import pdb; pdb.set_trace()
            _, pw_col_b = w_col[b, :h_length[b]-1].topk(3)
            # pw_num_b = 3
            pw_col_b = pw_col_b[:3].tolist()
    
            pwv_num = len(w_val_se[b]) if not gwvse else len(gwvse[b])
            # always > 0
            if pwv_num == 0:
                pwv_num = 1
                w_val_se[b] = [(0, len(q_tok)-1)]
                result.append([ps_num_b, set(ps_col_agg_b), pw_num_b, pw_conn_op_b, set(), set(), set()])
                continue
            pw_val_se_b = w_val_se[b]
            # if pwv_num == 0:
                # result.append([None] * 7)
                # continue
            
            # if pw_num_b == 1 and pwv_num == 1:
            #     col_ = pw_col_b[0]
            #     col_type_ = header_types[col_]
            #     op_ = 2
            #     _, ops_ = w_op[b, col_].sort(-1, True)
            #     for op in ops_.tolist():
            #         if col_type_ == "text":
            #             if op in (0, 1):
            #                 continue
            #         op_ = op
            #         break
            #     pw_col_op_b = [(col_, op_)]
            # elif pw_num_b = 

            w_val_match_b = w_val_match[b]
            pw_val_match_b = torch.tensor([-2**31 for _ in range(w_val_match_b.size(0))])
            pw_op_b = []
            # import pdb; pdb.set_trace()
            for col in pw_col_b:
                # if col in ps_col_b:
                #     continue
                # ignore empty column
                if col >= h_length[b] - 1:
                    continue
                # import pdb; pdb.set_trace()
                pw_val_match_b[col*4:col*4+pwv_num] = w_val_match_b[col*4:col*4+pwv_num]
                
                _, ops = w_op[b, col].sort(-1, True)
                ops = ops.tolist()
                # if table_id == "c9899dd4332111e9b4ca542696d6e445":
                #     import pdb; pdb.set_trace()
                op_ = 2
                for i,op1_ in enumerate(ops):
                    col_type = header_types[col]
                    if col_type in ("text", ):
                        if not op1_ in op_blacklist[col_type]:
                            # if op1_ == 3 and i != 0:
                            #     continue
                            op_ = op1_
                            break
                    else:
                        # flag = False
                        # for pat, op2_ in op_pattern.items():
                        #     if re.search(pat, q) is not None:
                        #         op_ = op2_
                        #         flag = True
                        #         break
                        # if flag:
                        #     break
                        op_before_matched = re.search(op_pattern[0], q) is not None
                        op_after_matched = re.search(op_pattern[1], q) is not None
                        if op_before_matched and not op_after_matched:
                            op_ = 1
                        elif op_after_matched and not op_before_matched:
                            op_ = 0
                        else:
                            op_ = op1_
                        break
                pw_op_b.append(op_)
                # pw_op_b.append(op)

            # print(pw_val_se_b)
            
            # _, pw_val_match_b = pw_val_match_b.topk(pw_num_b)
            _, pw_val_match_b = pw_val_match_b.sort(-1, True)
            pw_val_match_b = pw_val_match_b.tolist()[:3 * pwv_num]
            pw_val_match_b_1 = []
            val_matched = set()
            for match in pw_val_match_b:
                col = match // 4
                # ignore empty column
                if col >= h_length[b] - 1:
                    continue
                val_idx = match % 4
                if val_idx >= pwv_num:
                    continue
                col_type = header_types[col]
                wvs, wve = pw_val_se_b[val_idx]
                val_ = "".join(q_tok[wvs:wve+1]).replace("#", "")
                val_type = "real" if isreal(val_) else "text"
                if col_type != val_type:
                    continue
                if len(val_matched) < pwv_num:
                    if not val_idx in val_matched:
                        pw_val_match_b_1.append(match)
                else:
                    pw_val_match_b_1.append(match)
                val_matched.add(val_idx)
            # print(pw_val_match_b_1, pw_num_b)
            pw_val_match_b_1 = pw_val_match_b_1[:pw_num_b]
            if not pw_val_match_b_1:
                pw_val_match_b_1 = pw_val_match_b[:pw_num_b]

            col_op_map = dict([[pw_col_b[i], pw_op_b[i]] for i in range(3)])
            # import pdb; pdb.set_trace()

            pw_col_op_b = []
            for match in pw_val_match_b_1:
                col = match // 4
                val_idx = match % 4
                # try:
                pw_col_op_b.append((col, col_op_map[col]))
                # except:
                #     import pdb; pdb.set_trace()

            
            ps_col_agg_b.sort()
            pw_col_op_b.sort()
            pw_val_se_b.sort()
            pw_val_match_b_1.sort()
            result.append([ps_num_b, set(ps_col_agg_b), pw_num_b, pw_conn_op_b, set(pw_col_op_b), set(pw_val_se_b), set(pw_val_match_b_1)])
        return result
    

    def sql(self, sqli, records, tables):
        COND = {0: ">", 1: "<", 2: "==", 3: "!="}
        CONN = {0: "AND", 1: "OR"}
        AGG = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}

        sql = []
        for b,y in enumerate(sqli):
            q_tok = records[b]["q_tok"]
            table_id = records[b]["table_id"]
            headers = tables[table_id]["header"]
            header_types = tables[table_id]["types"]

            sn, sca, wn, wconn, wco, wv, wvm = y
            wv = list(wv)
            wv.sort()
            
            conn = CONN[wconn]
            wco_map = dict(wco)
            wvm_lst = [(match // 4, match % 4) for match in wvm]
            wva = ["".join(q_tok[wvs:wve+1]) for wvs,wve in wv]
            # print(wva, wv)

            # scols = [f'{AGG[agg]}("{headers[col]}")' for col,agg in sca]
            scols = []
            for col,agg in sca:
                if col >= len(headers):
                    continue
                if agg == 0:
                    ca = f'"{headers[col]}"'
                else:
                    ca = f'{AGG[agg]}("{headers[col]}")'
                scols.append(ca)
            
            wcols = set(wco_map.keys())
            scols_idx = set([col for col,_ in sca])
            for col in wcols:
                if col in scols_idx:
                    continue
                if col >= len(headers):
                    continue
                ca = f'"{headers[col]}"'
                scols.append(ca)
            
            conds_s = []
            conds = []
            for c_idx,v_idx in wvm_lst:
                col_ = headers[c_idx]
                op_ = COND[wco_map[c_idx]]
                val_ = wva[v_idx]

                if c_idx in scols_idx:
                    continue

                val1_ = val_
                if op_ in ("==", "=") and header_types[c_idx] == "text":
                    op_ = "like"
                    val1_ = f"%{val_}%"
                
                cond = f'("{col_}" {op_} "{val1_}")'
                conds_s.append(cond)
                conds.append([col_, op_, val_, c_idx])

            # conds = [f'("{headers[c_idx]}" {COND[wco_map[c_idx]]} "{wva[v_idx]}")' for c_idx, v_idx in wvm_lst]

            def sql_tmpl(scols, table_id, conn, conds_s):
                tmpl = (f'SELECT DISTINCT {",".join(scols)}\n'
                        f'FROM "{table_id}"\n')
                if conds_s:
                    tmpl += (f'WHERE {f" {conn} ".join(conds_s)}')
                return tmpl
            
            sql.append({"sql": sql_tmpl(scols, table_id, conn, conds_s), "tmpl": sql_tmpl, "sel": scols, "table_id": table_id, "conn": conn, "conds": conds})
        
        return sql


    def detail(self, y_predict, labels, col_sel, col_whr, col_match, q_length=[], records=[], tables={}):
        s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e, w_val, w_val_match, w_val_se = y_predict
        gs_num, gs_col, gs_agg, gw_num, gw_conn_op, gw_col, gw_op, gw_val, gw_val_match = labels

        # import pdb; pdb.set_trace()

        bs = s_col.size(0)
        q_len = w_val.size(-2)
        h_cnt = s_col.size(1)

        cnt_s_num = (s_num.argmax(dim=1) - gs_num.view(-1)).eq(0).sum().item()
        cnt_w_num = (w_num.argmax(dim=1) - gw_num.view(-1)).eq(0).sum().item()
        cnt_w_conn_op = (w_conn_op.argmax(dim=1) - gw_conn_op.view(-1)).eq(0).sum().item()

        topk = bs
        if topk > 4:
            topk = 4
        # _, s_col = s_col.squeeze(-1).topk(topk)
        # _, gs_col = gs_col.topk(topk)
        s_agg = s_agg.argmax(dim=-1)
        # _, w_col = w_col.squeeze(-1).topk(topk)
        # _, gw_col = gw_col.topk(topk)
        w_op = w_op.argmax(dim=-1)
        # w_val_s = w_val_s.squeeze(-1).argmax(dim=-1)
        # w_val_e = w_val_e.squeeze(-1).argmax(dim=-1)
        # _, w_val_match = w_val_match.view(bs, -1).topk(topk)
        # _, gw_val_match = gw_val_match.topk(topk)
        # _, w_val_match = w_val_match.view(bs,-1).topk(topk)
        # w_val_match, _ = w_val_match.sort()
        # _, gw_val_match = gw_val_match.topk(topk)
        # gw_val_match, _ = gw_val_match.sort()

        # import pdb; pdb.set_trace()
        def checkv(y, g, q_length, b):
            return y[b, :q_length[b]] == g[b, q_length[b]]
        cnt_w_val = 0
        cnt_w_val_all = 0
        w_val = w_val.argmax(dim=-1)
        wv_idx = gw_val == 1
        for i,q_len in enumerate(q_length):
            y = w_val[i, wv_idx[i]]
            g = gw_val[i, wv_idx[i]]
            if all(y == g):
                cnt_w_val += 1
            if all(w_val[i, :q_len] == gw_val[i, :q_len]):
                cnt_w_val_all += 1

        
        # import pdb; pdb.set_trace()
        # cnt_w_val_match = 0
        # for b, gw_match in enumerate(gw_val_match):
        #     flag = True
        #     for j, label in enumerate(gw_match):
        #         label = int(label)
        #         if label != -1:
        #             if w_val_match[b, j].item() != label:
        #                 flag = False
        #                 break
        #     if flag:
        #         cnt_w_val_match += 1

        def pred_topk(y, g, index):
            cnt = 0
            g_idx = g > -1
            # _, gw_val_match = gw_val_match.topk(topk)
            y = y.view(bs, -1)
            for b, match in enumerate(index):
                pos_cnt = 0
                if type(match) == list:
                    for cell in match:
                        idx,lbl = None, None
                        if type(cell) == list:
                            idx, lbl = cell
                        else:
                            idx = cell
                            lbl = 1
                        if lbl == 1:
                            pos_cnt += 1
                else:
                    pos_cnt = match
                # import pdb; pdb.set_trace()
                _, a = y[b, g_idx[b]].topk(pos_cnt)
                _, b = g[b, g_idx[b]].topk(pos_cnt)
                a = set(a.tolist())
                b = set(b.tolist())
                # b = set(gw_val_match[b, :pos_cnt].tolist())
                if a == b:
                    cnt += 1
            return cnt
        
        def pred_(y, g, index):
            cnt = 0
            for b, row in enumerate(index):
                flag = True
                for j in row:
                    idx = j[0] if type(j) == list else j
                    lbl = j[1] if type(j) == list else g[b, j].item()
                    if y[b, idx].item() != lbl:
                        flag = False
                        break
                if flag:
                    cnt += 1
            return cnt
        

        # def pred_1(y, g, index):
        #     cnt = 0
        #     for b, row in enumerate(index):
        #         flag = True
        #         # for j in range(row):
        #         #     if y[b, j] != g[b, j]:
        #         #         flag = False
        #         #         break
        #         a = set(y[b, :row].tolist())
        #         b = set(g[b, :row].tolist())
        #         if a - b == set():
        #         # if (y[b, :row] - g[b, :row]).eq(0).sum().item() == row:
        #             cnt += 1
        #     return cnt
        
        # import pdb; pdb.set_trace()
        col_whr_cnt = [len(set(whr)) for whr in col_whr]
        cnt_s_agg = pred_(s_agg, gs_agg, col_sel)
        cnt_w_op = pred_(w_op, gw_op, col_whr)
        cnt_s_col = pred_topk(s_col, gs_col, gs_num + 1)   # assume that we always have selected column(s).
        cnt_w_col = pred_topk(w_col, gw_col, col_whr_cnt)
        # cnt_w_val_s = pred_(w_val_s, gw_val_s, col_whr)
        # cnt_w_val_e = pred_(w_val_e, gw_val_e, col_whr)
        # import pdb; pdb.set_trace()
        cnt_w_val_match = pred_topk(w_val_match, gw_val_match, gw_num)

        # cnt_s_col = pred_col(s_col, gs_col, h_length)
        # cnt_w_col = pred_col(w_col, gw_col, h_length)
        # cnt_w_val_s = pred_col(w_val_s, gw_val_s, h_length)
        # cnt_w_val_e = pred_col(w_val_e, gw_val_e, h_length)
        # cnt_s_num = 0
        # cnt_w_conn_op = 0
        # cnt_w_val_match = 0
        cnt_w_val_s = (cnt_w_val, cnt_w_val_all)
        cnt_w_val_e = cnt_w_val_match
        return cnt_s_num, cnt_s_col, cnt_s_agg, cnt_w_num, cnt_w_conn_op, cnt_w_col, cnt_w_op, cnt_w_val_s, cnt_w_val_e
