import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, EncoderDecoderConfig, EncoderDecoderModel, AutoConfig, AutoModelForSeq2SeqLM, BartModel
from params import param

class CueSepPooler(nn.Module):
    def __init__(self, hidden_size, max_num_cue):
        super().__init__()
        self.pack = nn.LSTM(hidden_size + 1, hidden_size//2, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, max_num_cue + 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, hidden_states, cues):
        pack, _ = self.pack(torch.cat([hidden_states, cues], dim=-1))
        pack = self.tanh(pack)
        pack = self.ln(pack)

        fc = self.fc(pack)
        return fc

class CueBert(nn.Module):
    def __init__(self, config, max_num_cue=4):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.max_num_cue = max_num_cue + 1
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=param.bert_path, cache_dir=param.bert_cache, num_labels=self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if param.cue_matrix:
            self.cue = BiaffineClassifier(config.hidden_size, 1024, output_dim=self.num_labels)
        else:
            self.cue = nn.Linear(config.hidden_size, self.num_labels)
        self.bert.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        cue_labels=None,
        cue_sep_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cue_teacher=None
    ):
        r"""
        cue_labels & cue_sep_labels:
            e.g.:
            cue_labels:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
            cue_sep_labels: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
        To define how to seperate the cues
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        exists_label = cue_labels is not None and cue_sep_labels is not None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        cue_logits = self.cue(sequence_output)

        if not param.cue_matrix:
            if param.predict_cuesep:
                return cue_logits, cue_sep_logits
            else:
                return cue_logits
        else:
            return cue_logits

class ScopeBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=param.bert_path, cache_dir=param.bert_cache, num_labels=param.label_dim)
        self.dropout = nn.Dropout(param.dropout)
        if param.matrix:
            if param.augment_cue:
                if param.fact:
                    self.scope = nn.ModuleList([BiaffineClassifier(config.hidden_size, 1024, output_dim=1), 
                                                BiaffineClassifier(config.hidden_size, 1024, output_dim=config.num_labels)])
                else:
                    self.scope = BiaffineClassifier(config.hidden_size, 1024, output_dim=config.num_labels)
                    #self.scope = nn.Linear(1, 4)
            else:
                if param.temp:
                    #self.scope = BiaffineClassifier(config.hidden_size, 1024, output_dim=self.num_labels)
                    self.scope = BiaffineClassifier(config.hidden_size, 1024, output_dim=128)
                    self.convLayer = DConv(input_size=128, channels=128, dilation=[1,2,3,4], dropout=param.dropout)
                    self.fc = nn.Sequential(self.dropout, nn.Linear(128*4, self.num_labels))
                    #self.reg_embs = nn.Embedding(3, 20)
                    #self.lstm = nn.LSTM(config.hidden_size, param.hidden_dim, batch_first=True, bidirectional=True)
                    #self.cln = ConditionalLayerNorm(param.hidden_dim*2, param.hidden_dim*2, conditional=True)
                    #conv_input_size = param.hidden_dim*2 + 20
                    #self.convLayer = DConv(input_size=conv_input_size, channels=param.hidden_dim, dilation=[1,2,3,4], dropout=param.dropout)
                    #self.fc = nn.Sequential(self.dropout, nn.Linear(param.hidden_dim*4, 512), nn.ReLU(), 
                    #                        self.dropout, nn.Linear(512, self.num_labels), nn.ReLU())
                    #self.fc = CoPredictor(self.num_labels, param.hidden_dim, config.hidden_size, param.hidden_dim*4, 512, 0.33)
                elif param.task == 'joint' or param.multi:
                    self.scope = BiaffineClassifier(config.hidden_size, 1024, output_dim=self.num_labels)
                else:
                    self.lstm = nn.LSTM(config.hidden_size+1, 300, batch_first=True, bidirectional=True)
                    self.scope = BiaffineClassifier(300*2, 1024, output_dim=config.num_labels)
        else:
            if param.augment_cue:
                if param.boundary:
                    #self.get_trigram = nn.Conv1d(config.hidden_size, config.hidden_size, 3, padding=1, bias=False)
                    #self.get_trigram.weight = torch.nn.Parameter(torch.ones([config.hidden_size, config.hidden_size, 3]), requires_grad=False)
                    #self.get_trigram.requires_grad_ = False
                    #self.boundary = nn.LSTM(config.hidden_size+1, config.hidden_size//2, bidirectional=True, batch_first=True)
                    self.start_fc = nn.Linear(config.hidden_size, param.boundary_label_num)
                    self.end_fc = nn.Linear(config.hidden_size, param.boundary_label_num)
                    self.scope = nn.Linear(config.hidden_size, config.num_labels)
                else:
                    self.scope = nn.Linear(config.hidden_size, config.num_labels)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.bert.requires_grad_ = False
        #nn.init.xavier_normal_(self.scope.weight)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=None,
        return_dict=None,
        subword_mask=None,
        matrix_mask=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        #sequence_output = outputs[2][-1]
        bs = sequence_output.size(0)
        #sequence_output = torch.sum(sequence_output, dim=1).view(bs, -1, 1)
        #sequence_output = self.dropout(sequence_output)
        if param.augment_cue:
            if param.fact:
                arc_logits = self.sigm(self.scope[0](sequence_output))
                label_logits = self.scope[1](sequence_output)
                logits = [arc_logits, label_logits]
            else:
                if param.boundary:
                    #boundary_logits = self.get_trigram(sequence_output.transpose(1,2)).transpose(1,2)
                    #boundary_logits = self.relu(self.dropout(self.boundary(boundary_logits)[0]))
                    #boundary_logits = self.boundary_fc(boundary_logits)
                    logits = self.dropout(self.scope(sequence_output))
                    start_logits = self.dropout(self.start_fc(sequence_output))
                    end_logits = self.dropout(self.end_fc(sequence_output))
                    #boundary_logits = torch.cat([sequence_output, logits.argmax(-1).unsqueeze(-1)], dim=-1)
                    #boundary_logits = self.relu(self.dropout(self.boundary(boundary_logits)[0]))
                    #start_logits = self.dropout(self.start_fc(boundary_logits))
                    #end_logits = self.dropout(self.end_fc(boundary_logits))
                   
                else:
                    logits = self.dropout(self.scope(sequence_output))
                
        else:
            if param.temp:
                """length = subword_mask.size(1)
                biaf = self.scope(sequence_output)

                min_value = torch.min(sequence_output).item()
                
                # Max pooling word representations from pieces
                subword_matrix = torch.zeros([bs, sequence_output.size(1), sequence_output.size(1)]).to(self.device)
                for ir, (srow, mrow) in enumerate(zip(subword_mask, attention_mask)):
                    row_count = -1
                    for ic, (s, m) in enumerate(zip(srow, mrow)):
                        if m == 0:
                            break
                        if s == 1:
                            row_count += 1
                            subword_matrix[ir, row_count, ic] = 1
                        if s == 0:
                            if ic == 0:
                                row_count = 0
                            subword_matrix[ir, row_count, ic] = 1

                _bert_embs = sequence_output.unsqueeze(1).expand(-1, length, -1, -1)
                _bert_embs = torch.masked_fill(_bert_embs, subword_matrix.eq(0).unsqueeze(-1), min_value)
                word_reps, _ = torch.max(_bert_embs, dim=2)
                word_reps, (hidden, _) = self.lstm(word_reps)
                cln = self.cln(word_reps.unsqueeze(2), word_reps)

                tril_mask = torch.tril(matrix_mask.clone().long())
                reg_inputs = tril_mask + matrix_mask.clone().long()
                reg_emb = self.reg_embs(reg_inputs)

                conv_inputs = torch.cat([reg_emb, cln], dim=-1)

                conv_inputs = torch.masked_fill(conv_inputs, matrix_mask.eq(0).unsqueeze(-1), 0.0)
                conv_outputs = self.convLayer(conv_inputs)
                conv_outputs = torch.masked_fill(conv_outputs, matrix_mask.eq(0).unsqueeze(-1), 0.0)
                logits = self.fc(biaf, conv_outputs)"""
                sequence_output = self.scope(self.relu(sequence_output))
                sequence_output = self.convLayer(sequence_output)
                sequence_output = self.fc(self.dropout(sequence_output))
                logits = sequence_output
            elif param.task == 'joint' or param.multi:
                logits = self.scope(sequence_output)
            else:
                logits = self.lstm(sequence_output)
                logits = self.scope(logits[0])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if param.boundary:
            return logits, start_logits, end_logits
        else:
            return logits
        #if not return_dict:
        #    output = (logits,) + outputs[2:]
        #    return ((loss,) + output) if loss is not None else output

class BiaffineClassifier(nn.Module):
    def __init__(self, emb_dim, hid_dim, output_dim=param.label_dim, dropout=param.biaffine_hidden_dropout):
        super().__init__()
        self.dep = nn.Linear(emb_dim, hid_dim)
        self.head = nn.Linear(emb_dim, hid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.biaffine = PairwiseBiaffine(hid_dim, hid_dim, output_dim)
        self.biaffine = PairwiseBiaffine(hid_dim, hid_dim, output_dim)
        self.output_dim = output_dim
    
    def forward(self, embedding):
        bs = embedding.size(0)
        dep = self.dropout(self.relu(self.dep(embedding)))
        head = self.dropout(self.relu(self.head(embedding)))
        #dep = self.dropout(self.relu(embedding))
        #head = self.dropout(self.relu(embedding))
        #head = dep.clone()
        out = self.biaffine(dep, head)#.view(bs, -1, self.output_dim)
        #out = out.transpose(1, 2)
        #out = self.decoder(self.dropout(out))
        #out = out.transpose(1, 2)
        return out
        

class PairwiseBilinear(nn.Module):
    """ A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""
    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.zeros(input1_size, input2_size, output_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3).contiguous()
        # (N x L1 x L2 x O) + (O) -> (N x L1 x L2 x O)
        output = output + self.bias

        return output

class PairwiseBiaffine(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1)
        return self.W_bilin(input1, input2)

class DConv(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(DConv, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class ConditionalLayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(ConditionalLayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, o1, z):
        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2

class SpanScopeBert(nn.Module):
    def __init__(self, config, num_layers=2, lstm_dropout=0.35, soft_label=False, num_labels=1, *args, **kwargs):
        super().__init__(**kwargs)
        self.soft_label = soft_label
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=param.bert_path, cache_dir=param.bert_cache, num_labels=param.label_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        self.bert.init_weights()
        if param.use_lstm:
            self.bilstm = nn.LSTM(input_size=config.hidden_size,
                                hidden_size=config.hidden_size // 2,
                                batch_first=True,
                                num_layers=num_layers,
                                dropout=lstm_dropout,
                                bidirectional=True)
            self.init_lstm(self.bilstm)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        self.init_linear(self.start_fc.dense)
        if soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
            self.init_linear(self.end_fc.dense_0)
            self.init_linear(self.end_fc.dense_1)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
            self.init_linear(self.end_fc.dense_0)
            self.init_linear(self.end_fc.dense_1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_point=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if param.use_lstm:
            sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.layer_norm(sequence_output)
        ps1 = self.start_fc(sequence_output)
        if start_point is not None:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                start_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                start_logits.zero_()
                start_logits = start_logits.to(self.device)
                start_logits.scatter_(2, start_point.unsqueeze(2), 1)
            else:
                start_logits = start_point.unsqueeze(2).float()

        else:
            start_logits = F.softmax(ps1, -1)
            if not self.soft_label:
                start_logits = torch.argmax(start_logits, -1).unsqueeze(2).float()
        ps2 = self.end_fc(sequence_output, start_logits)
        return ps1, ps2
    
    def init_linear(self, input_):
        #bias = np.sqrt(6.0 / (input_.weight.size(0) + input_.weight.size(1)))
        nn.init.xavier_uniform_(input_.weight)
        if input_.bias is not None:
            input_.bias.data.zero_()

    def init_lstm(self, input_):
        for ind in range(0, input_.num_layers):
            weight = eval('input_.weight_ih_l' + str(ind))
            #bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.orthogonal_(weight)
            weight = eval('input_.weight_hh_l' + str(ind))
            #bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.orthogonal_(weight)
        if input_.bias:
            for ind in range(0, input_.num_layers):
                weight = eval('input_.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[input_.hidden_size: 2 * input_.hidden_size] = 1
                weight = eval('input_.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[input_.hidden_size: 2 * input_.hidden_size] = 1


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)
        

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'bert-base-cased')

    def forward(self, input_ids, padding_mask):
        outputs = self.model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=padding_mask)
        return outputs

class GAN(nn.Module):
    def __init__(self, cache=None):
        super().__init__()
        config = AutoConfig.from_pretrained(param.bert_path)
        #model = ScopeBert(config=config)
        if cache is None:
            cache = param.bert_cache
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=param.bert_path, config=config, cache_dir=cache)
        self.descriminator = BartModel.from_pretrained(pretrained_model_name_or_path=param.bert_path, config=config, cache_dir=cache)
        self.des_fc = nn.Linear(config.d_model, 1)
        self.meanpool =  nn.Linear(265, 1)#nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward_g(self, input_ids=None, attention_mask=None):
        g_output = self.generator(input_ids, attention_mask)[0]
        return g_output
    
    def forward_d(self, input_ids=None, attention_mask=None):
        d_output = self.descriminator(input_ids, attention_mask)[0]
        d_output = self.relu(d_output)
        d_output = self.des_fc(d_output).squeeze(-1).unsqueeze(1)
        d_output = self.sigmoid(self.meanpool(d_output.squeeze(1))).squeeze(1)
        return d_output
        

class SentBert(nn.Module):
    def __init__(self, config, max_num_cue=4):
        super().__init__()
        self.num_labels = config.num_labels
        self.max_num_cue = max_num_cue + 1
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=param.bert_path, cache_dir=param.bert_cache, num_labels=param.label_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cue = nn.Linear(config.hidden_size, self.num_labels)
        self.bert.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        cue_labels & cue_sep_labels:
            e.g.:
            cue_labels:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
            cue_sep_labels: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
        To define how to seperate the cues
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.pooler_output
        sequence_output = self.dropout(sequence_output)
        cue_logits = self.cue(sequence_output)
        return cue_logits