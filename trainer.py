import time
import os
import random
import itertools
from typing import List
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from nltk.translate.bleu_score import corpus_bleu
import util
from util import pack_subword_pred, decode_aug_seq, get_boundary, pack_subword_text
from params import param
import rouge
DEVICE = torch.device('cuda')


class AverageMeter(object):
    '''
    computes and stores the average and current value
    Example:
        >>> loss = AverageMeter()
        >>> for step,batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metrics(pred,target)
        >>>     loss.update(raw_loss.item(),n = 1)
        >>> cur_loss = loss.avg
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed=1029):
    '''
    set the seed for the whole enviornment
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def target_weight_score(score_dict, target_class):
    f1 = []
    prec = []
    rec = []
    num_classes = len(target_class)
    total_support = 0
    for class_ in target_class:
        try:
            f1.append(score_dict[class_]['f1-score'] * score_dict[class_]['support'])
            prec.append(score_dict[class_]['precision'] * score_dict[class_]['support'])
            rec.append(score_dict[class_]['recall'] * score_dict[class_]['support'])
            total_support += score_dict[class_]['support']
        except KeyError:
            num_classes -= 1
            continue
    if total_support == 0:
        return 0, 0, 0
    return np.sum(f1)/total_support, np.sum(prec)/total_support, np.sum(rec)/total_support

class CueTrainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.criterion = criterion
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device = DEVICE
        self.id2label = {y: x for x, y in label2id.items()}
        self.start_epoch = 1
        self.global_step = 0
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path + '\\checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_features, is_bert):
        """
        batch shape:
            [bs*][input_ids, padding_mask, scopes, input_len, cues, subword_mask]
        """
        pbar = tqdm(total=len(data_features), desc='Evaluating')
        valid_loss = AverageMeter()
        wrap_cue_pred = []
        wrap_cue_tar = []
        wrap_sent_pred = []
        wrap_sent_tar = []
        for step, f in enumerate(data_features):
            num_labels = len(self.id2label)
            input_ids = f[0].to(self.device)
            padding_mask = f[1].to(self.device)
            cues = f[2].to(self.device)
            cue_sep = f[3].to(self.device)
            input_lens = f[4]
            subword_mask = f[5].to(self.device)
            token_type = f[6].to(self.device)
            if param.cue_matrix:
                cue_matrix = f[6].to(self.device)
            bs = f[0].size(0)

            self.model.eval()
            with torch.no_grad():
                if param.predict_cuesep:
                    active_padding_mask = padding_mask.view(-1) == 1
                    cue_logits, cue_sep_logits = self.model(input_ids, padding_mask)
                    cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
                    cue_sep_loss = self.criterion(cue_sep_logits.view(-1, 5)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
                    loss = cue_loss + cue_sep_loss
                elif param.cue_matrix:
                    pad_matrix = []
                    for i in range(bs):
                        tmp = padding_mask[i].clone()
                        tmp = tmp.view(param.max_len, 1)
                        tmp_t = tmp.transpose(0, 1)
                        mat = tmp * tmp_t
                        pad_matrix.append(mat)
                    pad_matrix = torch.stack(pad_matrix, 0)
                    active_padding_mask = pad_matrix.view(-1) == 1
                    cue_logits = self.model(input_ids, padding_mask)
                    loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cue_matrix.view(-1)[active_padding_mask])
                else:
                    #if not param.augment_cue and param.task != 'joint':
                    active_padding_mask = padding_mask.view(-1) == 1
                    cue_logits = self.model(input_ids=input_ids, attention_mask=padding_mask, token_type_ids=token_type)
                    loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
            valid_loss.update(val=loss.item(), n=input_ids.size(0))

            if is_bert:
                if param.cue_matrix:
                    tmp_cue_pred = util.matrix_decode_toseq(cue_logits, pad_matrix, cue_label=1)              
                    cue_pred = []
                    cue_tar = []
                    for i in range(bs):
                        pred, tar = pack_subword_pred(tmp_cue_pred[i].detach().cpu().unsqueeze(0), cues[i].detach().cpu().unsqueeze(0),
                                                    subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                        cue_pred.append(pred[0])
                        cue_tar.append(tar[0])
                else:
                    cue_pred, cue_tar = pack_subword_pred(cue_logits.detach().cpu(), cues.detach().cpu(), subword_mask.detach().cpu(), padding_mask.cpu())

            else:
                cue_pred = cue_logits.argmax()
                cue_tar = cues
                if param.predict_cuesep:
                    cue_sep_pred = cue_sep_logits.argmax()
                    cue_sep_tar = cue_sep

            if param.dataset_name == 'sherlock' or param.dataset_name == 'sfu':
                # Post process for Sherlock, separate "n't" words and mark affixal cues
                new_pred = []
                new_tar = []
                if param.multi:
                    for i, seq in enumerate(input_ids):
                        text_seq = data_features.tokenizer.convert_ids_to_tokens(seq)
                        text_string = data_features.tokenizer.decode(seq)
                        for j, _ in enumerate(cue_pred[i]):
                            p, t = pack_subword_pred(cue_pred[i][j].detach().cpu().unsqueeze(0), cue_tar[i][j].detach().cpu().unsqueeze(0),
                                              subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                            new_pred.append(util.postprocess_sher(p[0], cues[i], subword_mask[i], input_lens[i], text_seq, text_string, cue_tar=cue_tar[i][j], sp=sp, st=st))
                            new_tar.append(util.postprocess_sher(t[0], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))
                    wrap_sent_pred.extend(cue_pred)
                    wrap_sent_tar.extend(cue_tar)
                else:
                    for i, seq in enumerate(input_ids):
                        text_seq = data_features.tokenizer.convert_ids_to_tokens(seq)
                        text_string = data_features.tokenizer.decode(seq)
                        new_pred.append(util.postprocess_sher(cue_pred[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))
                        new_tar.append(util.postprocess_sher(cue_tar[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))
                    wrap_cue_pred.extend(cue_pred)
                    wrap_cue_tar.extend(cue_tar)
                    wrap_sent_pred.extend([0 if 1 not in c else 1 for c in cue_pred])
                    wrap_sent_tar.extend([0 if 1 not in c else 1 for c in cue_tar])
            else:
                # Not sherlock or sfu
                for i1, sent in enumerate(cue_pred):
                    for i2, _ in enumerate(sent):
                        wrap_cue_pred.append(int(cue_pred[i1][i2]))
                        wrap_cue_tar.append(int(cue_tar[i1][i2]))
                    new_pred = cue_pred
                    new_tar = cue_tar
                    wrap_sent_pred.extend([0 if 1 not in c else 1 for c in cue_pred])
                    wrap_sent_tar.extend([0 if 1 not in c else 1 for c in cue_tar])

            pbar.update()
            pbar.set_postfix({'loss': valid_loss.avg})

            for i in range(bs):
                input_text = data_features.tokenizer.convert_ids_to_tokens(input_ids[i])
                temp_tar = []
                temp_pred = []
                for si, st in enumerate(cue_tar[i]):
                    if st == 1 and not input_text[si].startswith('[unus'):
                        temp_tar.append(input_text[si])
                for si, sp in enumerate(cue_pred[i]):
                    if sp == 1 and not input_text[si].startswith('[unus'):
                        temp_pred.append(input_text[si])
                if temp_pred == []:
                    temp_pred = ['a']
                if temp_tar == []:
                    temp_tar = ['a']
        
        wrap_cue_tar = list(itertools.chain.from_iterable(wrap_cue_tar))
        wrap_cue_pred = list(itertools.chain.from_iterable(wrap_cue_pred))
        cue_val_info = classification_report(wrap_cue_tar, wrap_cue_pred, output_dict=True, digits=5)
        sent_f1 = f1_score(wrap_sent_tar, wrap_sent_pred)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return cue_val_info, sent_f1


    def train_epoch(self, data_loader, valid_data, epoch, is_bert=True, max_num_cue=4, eval_every=800):
        pbar = tqdm(total=len(data_loader), desc='Training')
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            bs = batch[0].size(0)
            num_labels = len(self.id2label)
            if param.cue_matrix:
                input_ids, padding_mask, cues, cue_sep, input_len, subword_mask, cue_matrix = batch
            else:
                input_ids, padding_mask, cues, cue_sep, input_len, subword_mask, token_type = batch
            active_padding_mask = padding_mask.view(-1) == 1

            if param.cue_matrix:
                pad_matrix = []
                for i in range(bs):
                    tmp = padding_mask[i].clone()
                    tmp = tmp.view(param.max_len, 1)
                    tmp_t = tmp.transpose(0, 1)
                    mat = tmp * tmp_t
                    pad_matrix.append(mat)
                pad_matrix = torch.stack(pad_matrix, 0)
                active_padding_mask = pad_matrix.view(-1) == 1
                cue_logits = self.model(input_ids, padding_mask)
                loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cue_matrix.view(-1)[active_padding_mask])
            else:
                if param.predict_cuesep:
                    cue_logits, cue_sep_logits = self.model(input_ids, padding_mask, cue_teacher=cues)
                    cue_loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])
                    cue_sep_loss = self.criterion(cue_sep_logits.view(-1, max_num_cue+1)[active_padding_mask], cue_sep.view(-1)[active_padding_mask])
                    loss = cue_loss + cue_sep_loss
                else:
                    cue_logits = self.model(input_ids=input_ids, attention_mask=padding_mask, token_type_ids=token_type)
                    
                            
                    loss = self.criterion(cue_logits.view(-1, num_labels)[active_padding_mask], cues.view(-1)[active_padding_mask])

            #if len(self.n_gpu.split(",")) >= 2:
            #    loss = loss.mean()
            
            # scale the loss
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            # gradient clip
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=input_ids.size(0))
            if step % eval_every == 0 and step > eval_every:
                if param.predict_cuesep:
                    cue_log, cue_sep_log = self.valid_epoch(valid_data, is_bert)
                    cue_f1 = target_weight_score(cue_log, ['0', '1', '2'])
                    cue_sep_f1 = target_weight_score(cue_sep_log, ['1', '2', '3', '4'])
                    metric = cue_f1[0] + cue_sep_f1[0]
                else:
                    cue_log = self.valid_epoch(valid_data, is_bert)
                    cue_f1 = target_weight_score(cue_log, ['1'])
                    metric = cue_f1[0]
                if hasattr(self.lr_scheduler,'epoch_step'):
                    self.lr_scheduler.epoch_step(metrics=metric, epoch=epoch)
            pbar.update()
            pbar.set_postfix({'loss': tr_loss.avg})
        info = {'loss': tr_loss.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def train(self, train_data, valid_data, epochs, is_bert=True):
        #seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data, valid_data, epoch, is_bert)

            if param.predict_cuesep:
                cue_log, cue_sep_log = self.valid_epoch(valid_data, is_bert)
                cue_f1 = target_weight_score(cue_log[0], ['1'])
                cue_sep_f1 = target_weight_score(cue_sep_log, ['1', '2', '3', '4'])
                logs = {'loss': train_log['loss'], 'val_cue_f1': cue_f1[0], 'val_cuesep_f1': cue_sep_f1[0]}
                score = cue_f1[0]+cue_sep_f1[0]
            elif param.cue_matrix:
                cue_log, sent_f1 = self.valid_epoch(valid_data, is_bert)
                cue_f1 = target_weight_score(cue_log, ['1'])
                logs = {'loss': train_log['loss'], 'val_cue_f1': cue_f1[0], 'sent_f1': sent_f1}
                score = cue_f1[0]
            else:
                cue_log = self.valid_epoch(valid_data, is_bert)
                cue_f1 = target_weight_score(cue_log[0], ['1', '2'])
                logs = {'loss': train_log['loss'], 'val_cue_f1': cue_f1[0]}
                score = cue_f1[0]

            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(logs)
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=score, epoch=epoch)
            # save log
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=score)
                self.model_checkpoint.epoch_step(current=score, state=state)
                
            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=score)
                if self.early_stopping.stop_training:
                    break

class ScopeTrainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 resume_path=None, training_monitor=None, model_checkpoint=None, alpha=0.5):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.criterion = criterion
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.alpha = alpha

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device = DEVICE
        #self.id2label = {y: x for x, y in label2id.items()}
        self.start_epoch = 1
        self.global_step = 0
        self.end_epoch = 0
        self.arc_loss = torch.nn.BCELoss()
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path + '/checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_features, is_bert, epochs):
        """
        batch shape:
            [bs*][input_ids, padding_mask, scopes, input_len, cues, subword_mask]
        """
        pbar = tqdm(total=len(data_features), desc='Evaluating')
        valid_loss = AverageMeter()
        wrap_scope_pred = []
        wrap_scope_pred_bound = []
        wrap_scope_tar = []
        mat_pred = []
        mat_tar = []
        sm_pred = []
        sm_tar = []
        cm_pred = []
        cm_tar = []
        wrap_logits = []
        wrap_rawtar = []
        mismatch = []
        for step, f in enumerate(data_features):
            num_labels = param.label_dim
            input_ids = f[0].to(self.device)
            padding_mask = f[1].to(self.device)
            input_lens = f[3]
            if not param.multi:
                scopes = f[2].to(self.device)
                cues = f[4].to(self.device)
            else:
                scopes = f[2]
                cues = f[4]
            subword_mask = f[5].to(self.device)
            bs = f[0].size(0)
            if param.matrix:
                scopes_matrix = f[-1].to(self.device)
            self.model.eval()
            with torch.no_grad():
                active_padding_mask = padding_mask.view(-1) == 1
                if param.matrix:
                    pad_matrix = []
                    for i in range(bs):
                        tmp = padding_mask[i].clone()
                        tmp = tmp.view(param.max_len, 1)
                        tmp_t = tmp.transpose(0, 1)
                        mat = tmp * tmp_t
                        pad_matrix.append(mat)
                    pad_matrix = torch.stack(pad_matrix, 0)
                    active_padding_mask = pad_matrix.view(-1) == 1
                    if not param.augment_cue and param.task != 'joint':
                        scope_logits = self.model([input_ids, cues], padding_mask)[0]
                    else:
                        scope_logits = self.model(input_ids, padding_mask, subword_mask=subword_mask, matrix_mask=pad_matrix)[0]
                    
                    if param.fact:
                        # Factorized (arc and label classifier)
                        arc_targets = util.label_to_arc_matrix(scopes_matrix)
                        arc_logits, label_logits = scope_logits
                        arc_logit_masked = arc_logits.view(-1)[active_padding_mask]
                        arc_target_masked = arc_targets.view(-1)[active_padding_mask]
                        arc_mask = arc_logits.view(-1) > 0
                        label_logit_masked = label_logits.view(-1, num_labels)[arc_mask]
                        label_target_masked = scopes_matrix.view(-1)[arc_mask]
                        arc_loss = self.arc_loss(arc_logit_masked, arc_target_masked.float())
                        label_loss = self.criterion(label_logit_masked, label_target_masked)
                        loss = arc_loss + label_loss
                    else:
                        logits_masked = scope_logits.view(-1, num_labels)[active_padding_mask]
                        target_masked = scopes_matrix.view(-1)[active_padding_mask]
                        mat_pred.extend(logits_masked.detach().cpu().argmax(-1).tolist())
                        mat_tar.extend(target_masked.detach().cpu().tolist())
                        loss = self.criterion(logits_masked, target_masked)
                else:
                    #if not param.augment_cue and param.task != 'joint':
                    if param.boundary:
                        if not param.bound_only:
                            scope_logits, start_logits, end_logits = self.model(input_ids, padding_mask)
                            if epochs > 0:
                                boundary, start, end = get_boundary(scopes)
                                start, end = self.get_boundary_dist_map(boundary, input_lens)
                                """pred_bound, _, _ = get_boundary(scope_logits.argmax(-1))
                                bound_pos = pred_bound.nonzero()
                                boundary_mask = torch.zeros_like(padding_mask)
                                for bound in bound_pos:
                                    ci = bound[1]
                                    ri = bound[0]
                                    boundary_mask[ri, ci] = 1
                                    if ci > 1:
                                        boundary_mask[ri, ci-1] = 1
                                    if ci < param.max_len-1:
                                        boundary_mask[ri, ci+1] = 1
                                active_bound_mask = boundary_mask.view(-1) == 1"""
                                """loss = self.alpha*self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask]) +\
                                    ((1-self.alpha)/2)*self.criterion[0](start_logits.view(-1, param.boundary_label_num)[active_padding_mask], start.view(-1)[active_padding_mask]) +\
                                    ((1-self.alpha)/2)*self.criterion[0](end_logits.view(-1, param.boundary_label_num)[active_padding_mask], end.view(-1)[active_padding_mask])"""
                                loss = self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask]) +\
                                    2*self.criterion[1](start_logits.view(-1, param.boundary_label_num)[active_padding_mask], start.view(-1)[active_padding_mask]) +\
                                    2*self.criterion[1](end_logits.view(-1, param.boundary_label_num)[active_padding_mask], end.view(-1)[active_padding_mask])
                            else:
                                loss = self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask])
                            
                        else:
                            boundary, start, end = get_boundary(scopes)
                            start, end = self.get_boundary_dist_map(boundary, input_lens)
                            start_logits, end_logits = self.model(input_ids, padding_mask)
                            loss = self.criterion(start_logits.view(-1, param.boundary_label_num)[active_padding_mask], start.view(-1)[active_padding_mask]) +\
                                self.criterion(end_logits.view(-1, param.boundary_label_num)[active_padding_mask], end.view(-1)[active_padding_mask])
                    else:
                        scope_logits = self.model(input_ids, padding_mask)
                        loss = self.criterion(scope_logits.view(-1, num_labels)[active_padding_mask.view(-1)], scopes.view(-1)[active_padding_mask.view(-1)])
            valid_loss.update(val=loss.item(), n=input_ids.size(0))

            if is_bert: 
                if param.matrix:
                    if param.multi:
                        scope_pred = []
                        scope_tar = []
                        temp_scope_pred, temp_cue_pred = util.multi_matrix_decode_toseq(scope_logits, pad_matrix)
                        if param.dataset_name != 'sherlock' and param.dataset_name != 'sfu':
                            for i in range(bs):
                                sp, st = util.handle_eval_multi(temp_scope_pred[i], scopes[i], temp_cue_pred[i], cues[i])
                                for j, _ in enumerate(sp):
                                    pred, tar = pack_subword_pred(sp[j].detach().cpu().unsqueeze(0), st[j].detach().cpu().unsqueeze(0),
                                                            subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                                    scope_pred.append(pred[0])
                                    scope_tar.append(tar[0])
                        else:
                            for i in range(bs):
                                sp, st = util.handle_eval_multi(temp_scope_pred[i], scopes[i], temp_cue_pred[i], cues[i])
                                scope_pred.append(sp)
                                scope_tar.append(st)
                    else:
                        label_logits = scope_logits[1] if param.fact else scope_logits
                        tmp_scope_pred = util.matrix_decode_toseq(label_logits, pad_matrix)                    
                        scope_pred = []
                        scope_tar = []
                        
                        for i in range(bs):
                            pred, tar = pack_subword_pred(tmp_scope_pred[i].detach().cpu().unsqueeze(0), scopes[i].detach().cpu().unsqueeze(0),
                                                        subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                            scope_pred.append(pred[0])
                            scope_tar.append(tar[0])
                else:
                    scope_pred_bound = []
                    if not param.bound_only:
                        wrap_logits.append(scope_logits.detach().argmax(-1).view(-1)[active_padding_mask].tolist())
                        wrap_rawtar.append(scopes.detach().view(-1)[active_padding_mask].tolist())
                        scope_pred, scope_tar = pack_subword_pred(scope_logits.detach().cpu(), scopes.detach().cpu(), subword_mask.detach().cpu(), padding_mask.cpu())
                        _, cue_tar = pack_subword_pred(scope_logits.detach().cpu(), cues.detach().cpu(), subword_mask.detach().cpu(), padding_mask.cpu())
                        if epochs > 5:
                            scope_pred_b, s_score, s_idx, e_score, e_idx = util.decode_seq_from_bound(start_logits, end_logits, padding_mask, voting=True)
                            for i in range(bs): 
                                scope_pred_bound.append(pack_subword_text(scope_pred_b[i], subword_mask[i], input_lens[i]))
                        else:
                            scope_pred_bound = [[0]*len(sp) for sp in scope_pred]
                    else:
                        #scope_pred_bound, s_score, e_score = util.decode_seq_from_bound(start_logits, end_logits, padding_mask, voting=True)
                        scope_pred = [pack_subword_text(scope_pred[i], subword_mask[i].detach().cpu(), input_lens[i]) for i in range(bs)]
                        _, scope_tar = pack_subword_pred(start_logits.detach().cpu(), scopes.detach().cpu(), subword_mask.detach().cpu(), padding_mask.cpu())
                        _, cue_tar = pack_subword_pred(start_logits.detach().cpu(), cues.detach().cpu(), subword_mask.detach().cpu(), padding_mask.cpu())
            else:
                scope_pred = scope_logits.argmax()
                scope_tar = scopes
                scope_pred = [e.tolist() for e in scope_pred]
                scope_tar = [e.tolist() for e in scope_tar]

            if param.task == 'joint' and not param.multi:
                new_pred = []
                for i, seq in enumerate(input_ids):
                    new_pred.append(util.handle_eval_joint(scope_pred[i], scope_tar[i]))
                scope_pred = new_pred
            if param.dataset_name == 'sherlock' or param.dataset_name == 'sfu':
                # Post process for Sherlock, separate "n't" words and mark affixal cues
                new_pred = []
                new_pred_bound = []
                new_tar = []
                if param.multi:
                    for i, seq in enumerate(input_ids):
                        text_seq = data_features.tokenizer.convert_ids_to_tokens(seq)
                        text_string = data_features.tokenizer.decode(seq)
                        tp = []
                        tt = []
                        for j, _ in enumerate(scope_pred[i]):
                            p, t = pack_subword_pred(scope_pred[i][j].detach().cpu().unsqueeze(0), scope_tar[i][j].detach().cpu().unsqueeze(0),
                                              subword_mask[i].detach().cpu().unsqueeze(0), padding_mask[i].cpu().unsqueeze(0))
                            tp.append(util.postprocess_sher(p[0], cues[i], subword_mask[i], input_lens[i], text_seq, text_string, scope_tar=scope_tar[i][j], sp=sp, st=st))
                            tt.append(util.postprocess_sher(t[0], cues[i], subword_mask[i], input_lens[i], text_seq, text_string))
                        new_pred.append(tp)
                        new_tar.append(tt)

                        for i1, scope_group in enumerate(new_pred):
                            for i2, sent in enumerate(scope_group):
                                ssp, sst = util.full_scope_match(new_pred[i1][i2], new_tar[i1][i2])
                                cp, ct = util.cue_match(new_pred[i1][i2], new_tar[i1][i2])
                                sm_pred.append(ssp)
                                sm_tar.append(sst)
                                cm_pred.append(cp)
                                cm_tar.append(ct)
                                if ssp == 0 and sst ==1:
                                    ori_sent = data_features.tokenizer.convert_ids_to_tokens(input_ids[i1])
                                    ori_sent = pack_subword_text(ori_sent, subword_mask[i1], input_lens[i1])
                                    pred_span = [e if p == 1 else '' for e, p in zip(ori_sent, new_pred[i1][i2])]
                                    tar_span  = [e if p == 1 else '' for e, p in zip(ori_sent, new_tar[i1][i2])]
                                    mismatch.append([ori_sent, pred_span, tar_span])
                                for i3, _ in enumerate(sent):
                                    wrap_scope_pred.append(int(new_pred[i1][i2][i3]))
                                    wrap_scope_tar.append(int(new_tar[i1][i2][i3]))
                else:
                    scope_pred = util.exclude_cue(scope_pred, cue_tar)
                    scope_tar = util.exclude_cue(scope_tar, cue_tar)
                    # Single scope (default case)
                    for i, seq in enumerate(input_ids):
                        text_seq = data_features.tokenizer.convert_ids_to_tokens(seq)
                        text_string = data_features.tokenizer.decode(seq)
                        """if param.bio:
                            temp_pred = []
                            temp_tar = []
                            for p, cue in zip(scope_pred[i], cues[i]):
                                temp_pred.append(p if cue not in [1, 2, 4] else 4)
                            for t, cue in zip(scope_tar[i], cues[i]):
                                temp_tar.append(t if cue not in [1, 2, 4] else 4)
                            pred = util.postprocess_sher(temp_pred, cues[i], subword_mask[i], input_lens[i], text_seq, text_string)
                            tar = util.postprocess_sher(temp_tar, cues[i], subword_mask[i], input_lens[i], text_seq, text_string)
                        else:"""
                        if param.dataset_name == 'sherlock':
                            pred = util.postprocess_sher(scope_pred[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string)
                            
                            
                            pred_bound = util.postprocess_sher(scope_pred_bound[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string)
                            tar = util.postprocess_sher(scope_tar[i], cues[i], subword_mask[i], input_lens[i], text_seq, text_string)
                        else:
                            pred = scope_pred[i]
                            tar = scope_tar[i]
                            pred_bound = scope_pred_bound[i]
                        
                        new_pred.append(pred)
                        new_pred_bound.append(pred_bound)
                        new_tar.append(tar)

                    for i1, sent in enumerate(new_pred):
                        ssp, sst = util.full_scope_match(new_pred[i1], new_tar[i1])
                        cp, ct = util.cue_match(new_pred[i1], new_tar[i1])
                        sm_pred.append(ssp)
                        sm_tar.append(sst)
                        cm_pred.append(cp)
                        cm_tar.append(ct)
                        if ssp == 0 and sst ==1:
                            ori_sent = data_features.tokenizer.convert_ids_to_tokens(input_ids[i1])
                            ori_sent = pack_subword_text(ori_sent, subword_mask[i1], input_lens[i1])
                            pred_span = [e if p == 1 else '' for e, p in zip(ori_sent, new_pred[i1])]
                            tar_span  = [e if p == 1 else '' for e, p in zip(ori_sent, new_tar[i1])]
                            mismatch.append([ori_sent, pred_span, tar_span])
                        for i2, _ in enumerate(sent):
                            wrap_scope_pred.append(int(new_pred[i1][i2]))
                            wrap_scope_tar.append(int(new_tar[i1][i2]))
                    for i1, sent in enumerate(new_pred_bound):
                        for i2, _ in enumerate(sent):
                            wrap_scope_pred_bound.append(int(new_pred_bound[i1][i2]))
            else:
                for i1, sent in enumerate(scope_pred):
                    ssp, sst = util.full_scope_match(scope_pred[i1], scope_tar[i1])
                    cp, ct = util.cue_match(scope_pred[i1], scope_tar[i1])
                    sm_pred.append(ssp)
                    sm_tar.append(sst)
                    cm_pred.append(cp)
                    cm_tar.append(ct)
                    for i2, _ in enumerate(sent):
                        wrap_scope_pred.append(int(scope_pred[i1][i2]))
                        wrap_scope_tar.append(int(scope_tar[i1][i2]))
                    new_pred = scope_pred
                    new_tar = scope_tar

            pbar.update()
            pbar.set_postfix({'loss': valid_loss.avg})

            """for i in range(bs):
                input_text = data_features.tokenizer.convert_ids_to_tokens(input_ids[i])
                temp_tar = []
                temp_pred = []
                for si, st in enumerate(scope_tar[i]):
                    if st == 1 and not input_text[si].startswith('[unus'):
                        temp_tar.append(input_text[si])
                for si, sp in enumerate(scope_pred[i]):
                    if sp == 1 and not input_text[si].startswith('[unus'):
                        temp_pred.append(input_text[si])
                if temp_pred == []:
                    temp_pred = ['a']
                if temp_tar == []:
                    temp_tar = ['a']
                scope_tok_pred.append(' '.join(temp_pred))
                scope_tok_tar.append(' '.join(temp_tar))
            """
                
        if param.label_dim > 3:
            cue_f1 = f1_score(cm_tar, cm_pred)
        else:
            cue_f1 = 1
        scope_match = f1_score(sm_tar, sm_pred)
        if (param.mark_cue or param.matrix) and ('bioscope' in param.dataset_name):
            # For bioscope and sfu, include "cue" into scope if predicting cue
            wrap_scope_tar = [e if e != 3 else 2 for e in wrap_scope_tar]
            wrap_scope_pred = [e if e != 3 else 2 for e in wrap_scope_pred]
        if (param.mark_cue or param.matrix) and ('vet' in param.dataset_name):
            # For bioscope and sfu, include "cue" into scope if predicting cue
            wrap_scope_tar = [e if e != 3 else 1 for e in wrap_scope_tar]
            wrap_scope_pred = [e if e != 3 else 1 for e in wrap_scope_pred]
        wrap_logits = list(itertools.chain.from_iterable(wrap_logits))
        wrap_rawtar = list(itertools.chain.from_iterable(wrap_rawtar))
        logit_val_info = classification_report(wrap_rawtar, wrap_logits, output_dict=True, digits=5)
        scope_val_info = classification_report(wrap_scope_tar, wrap_scope_pred, output_dict=True, digits=5)
        wrap_scope_tar = [e if e!=3 else 1 for e in wrap_scope_tar]
        #bound_scope_info = classification_report(wrap_scope_tar, wrap_scope_pred_bound, output_dict=True, digits=5)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        r = rouge.Rouge()
        #self.logger.info(f'rouge: {r.get_scores(scope_tok_pred, scope_tok_tar, avg=True)}')
        if param.matrix:
            mat_f1 = classification_report(mat_tar, mat_pred, output_dict=True, digits=5)
        self.logger.info('scope match %f', scope_match)
        return scope_val_info, cue_f1, scope_match

    def train_epoch(self, data_loader, valid_data, epochs):
        pbar = tqdm(total=len(data_loader), desc='Training')
        num_labels = param.label_dim
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
            if param.matrix:
                input_ids, padding_mask, scopes, input_lens, cues, subword_mask, scopes_matrix = batch
            else:
                input_ids, padding_mask, scopes, input_lens, cues, subword_mask = batch
            bs = batch[0].size(0)
            active_padding_mask = padding_mask.view(-1) == 1
            if param.matrix:
                pad_matrix = []
                for i in range(bs):
                    tmp = padding_mask[i].clone()
                    tmp = tmp.view(param.max_len, 1)
                    tmp_t = tmp.transpose(0, 1)
                    mat = tmp * tmp_t
                    pad_matrix.append(mat)
                pad_matrix = torch.stack(pad_matrix, 0)
                active_padding_mask = pad_matrix.view(-1) == 1
                #with torch.cuda.amp.autocast():
                if not param.augment_cue and param.task != 'joint':
                    scope_logits = self.model([input_ids, cues], padding_mask)
                else:
                    scope_logits = self.model(input_ids=input_ids, attention_mask=padding_mask, subword_mask=subword_mask, matrix_mask=pad_matrix)
            
                if param.fact:
                    # Factorized (arc and label classifier)
                    arc_targets = util.label_to_arc_matrix(scopes_matrix)
                    arc_logits, label_logits = scope_logits
                    arc_logit_masked = arc_logits.view(-1)[active_padding_mask]
                    arc_target_masked = arc_targets.view(-1)[active_padding_mask]
                    arc_mask = arc_logits.view(-1) > 0
                    label_logit_masked = label_logits.view(-1, num_labels)[arc_mask]
                    label_target_masked = scopes_matrix.view(-1)[arc_mask]
                    arc_loss = self.arc_loss(arc_logit_masked, arc_target_masked.float())
                    label_loss = self.criterion(label_logit_masked, label_target_masked)
                    loss = arc_loss + label_loss
                else:
                    # Unfactorized (Single classifier for both arc and label)
                    logits_masked = scope_logits.view(-1, num_labels)[active_padding_mask]
                    target_masked = scopes_matrix.view(-1)[active_padding_mask]
                    loss = self.criterion(logits_masked, target_masked)
            else:
                #if not param.augment_cue and param.task != 'joint':
                if param.boundary:
                    if not param.bound_only:
                        scope_logits, start_logits, end_logits = self.model(input_ids, padding_mask)

                        if epochs > param.bsl_warmup:
                            boundary, start, end = get_boundary(scopes)
                            start, end = self.get_boundary_dist_map(boundary, input_lens)
                            """pred_bound, _, _ = get_boundary(scope_logits.argmax(-1))
                            bound_pos = pred_bound.nonzero()
                            boundary_mask = torch.zeros_like(padding_mask)
                            for bound in bound_pos:
                                ci = bound[1]
                                ri = bound[0]
                                boundary_mask[ri, ci] = 1
                                if ci > 1:
                                    boundary_mask[ri, ci-1] = 1
                                if ci < param.max_len-1:
                                    boundary_mask[ri, ci+1] = 1
                            active_bound_mask = boundary_mask.view(-1) == 1"""
                            """loss = self.alpha*self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask]) +\
                                ((1-self.alpha)/2)*self.criterion[1](start_logits.view(-1, param.boundary_label_num)[active_padding_mask], start.view(-1)[active_padding_mask]) +\
                                ((1-self.alpha)/2)*self.criterion[1](end_logits.view(-1, param.boundary_label_num)[active_padding_mask], end.view(-1)[active_padding_mask])"""
                            loss = self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask]) +\
                                2*self.criterion[1](start_logits.view(-1, param.boundary_label_num)[active_padding_mask], start.view(-1)[active_padding_mask]) +\
                                2*self.criterion[1](end_logits.view(-1, param.boundary_label_num)[active_padding_mask], end.view(-1)[active_padding_mask])
                        else:
                            loss = self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask])
                    else:
                        start_logits, end_logits = self.model(input_ids, padding_mask)
                        boundary, start, end = get_boundary(scopes)
                        start, end = self.get_boundary_dist_map(boundary, input_lens)
                        loss = self.criterion(start_logits.view(-1, param.boundary_label_num)[active_padding_mask], start.view(-1)[active_padding_mask]) +\
                            self.criterion(end_logits.view(-1, param.boundary_label_num)[active_padding_mask], end.view(-1)[active_padding_mask])
                else:
                    scope_logits = self.model(input_ids=input_ids, attention_mask=padding_mask)#, position_ids=new_pos_ids)
                    loss = self.criterion(scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask])
                #loss = self.criterion[0](scope_logits.view(-1, num_labels)[active_padding_mask], scopes.view(-1)[active_padding_mask])

            #if len(self.n_gpu.split(",")) >= 2:
            #    loss = loss.mean()
            
            # scale the loss
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            # gradient clip
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=input_ids.size(0))
            pbar.update()
            pbar.set_postfix({'loss': tr_loss.avg})
        info = {'loss': tr_loss.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def get_boundary_dist_map(self, input_boundary, input_lens):
        bs = input_boundary.size(0)
        seqlen = input_boundary.size(1)
        start = torch.zeros([bs, seqlen], dtype=torch.long, device='cuda')
        end = torch.zeros([bs, seqlen], dtype=torch.long, device='cuda')
        for rownum, row in enumerate(input_boundary):
            row = row.tolist()
            start_pos = [i for i, x in enumerate(row) if x in [1, 3]]
            end_pos = [i for i, x in enumerate(row) if x in [2, 3]]
            for i, e in enumerate(row):
                if i > input_lens[rownum]:
                    break
                dist = 999
                for s in start_pos:
                    if abs(s - i) < abs(dist):
                        dist = s - i
                if dist < 0:
                    start[rownum, i] = 2#torch.LongTensor([1, 0, 0])
                elif dist > 0:
                    start[rownum, i] = 3#torch.LongTensor([0, 0, 1])
                else:
                    start[rownum, i] = 1#torch.LongTensor([0, 1, 0])
            for i, e in enumerate(row):
                if i > input_lens[rownum]:
                    break
                dist = 999
                for s in end_pos:
                    if abs(s - i) < abs(dist):
                        dist = s - i
                if dist < 0:
                    end[rownum, i] = 2#torch.LongTensor([1, 0, 0])
                elif dist > 0:
                    end[rownum, i] = 3#torch.LongTensor([0, 0, 1])
                else:
                    end[rownum, i] = 1#torch.LongTensor([0, 1, 0])
        return start, end

    def train(self, train_data, valid_data, epochs, is_bert=False):
        #seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data, valid_data, epoch)
            scope_log, cue_f1, scope_match = self.valid_epoch(valid_data, is_bert, epoch)
            #scope_f1 = target_weight_score(scope_log, ['1', '1.0'])
            scope_f1 = target_weight_score(scope_log, ['1'])
            logs = {'loss': train_log['loss'], 'val_scope_token_f1': scope_f1[0]}
            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            if param.task == 'joint':
                self.logger.info('cue_f1 %f', cue_f1)
            self.logger.info(logs)
            self.end_epoch = epoch
            #self.logger.info("The entity scores of valid data : ")
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=scope_f1[0], epoch=epoch)
            # save log
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=logs['val_scope_token_f1'])
                if self.early_stopping.stop_training:
                    break
            torch.cuda.empty_cache()

class Seq2SeqTrainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        #self.criterion = criterion
        #self.optimizer = optimizer
        self.g_loss = torch.nn.CrossEntropyLoss()
        self.d_loss = torch.nn.MSELoss()
        self.g_optimizer = optimizer[0]
        self.d_optimizer = optimizer[1]
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device = DEVICE
        #self.id2label = {y: x for x, y in label2id.items()}
        self.start_epoch = 1
        self.global_step = 0
        self.arc_loss = torch.nn.BCELoss()
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path + '/checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_features, is_bert, gen_only):
        """
        batch shape:
            [bs*][input_ids, padding_mask, scopes, input_len, cues, subword_mask]
        """
        pbar = tqdm(total=len(data_features), desc='Evaluating')
        pred_scopes = []
        tar_scopes = []
        scores = []
        g_loss_monitor = AverageMeter()
        for step, batch in enumerate(data_features):
            self.model.eval()
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, padding_mask, scopes, input_len, cues, target_ids, target_padding_mask, target_subword_mask = batch
            bs = batch[0].size(0)
            active_padding_mask = padding_mask.view(-1) == 1
            with torch.no_grad():
                scope_logits = self.model.forward_g(input_ids, padding_mask)
                if not gen_only:
                    d_logits = self.model.forward_d(scope_logits.argmax(-1))
                    scores.extend(d_logits.tolist())
                gloss = self.g_loss(scope_logits.view(-1, scope_logits.size(-1)), target_ids.view(-1))

                for i in range(bs):
                    pred_sent = data_features.tokenizer.convert_ids_to_tokens(scope_logits[i].argmax(-1))
                    tar_sent = data_features.tokenizer.convert_ids_to_tokens(target_ids[i])
                    temp_pred = decode_aug_seq(pred_sent, target_padding_mask[i].count_nonzero())
                    temp_tar = decode_aug_seq(tar_sent, target_padding_mask[i].count_nonzero())
                    if temp_pred == [0]:
                        temp_pred = [0] * len(temp_tar)
                    while len(temp_pred) < len(temp_tar):
                        temp_pred.append(0)
                    if len(temp_pred) > len(temp_tar):
                        temp_pred = temp_pred[:len(temp_tar)]
                    pred_scopes.append(temp_pred)
                    tar_scopes.append(temp_tar)
                    
            g_loss_monitor.update(val=gloss.item(), n=input_ids.size(0))
            pbar.update()
            pbar.set_postfix({'loss': g_loss_monitor.avg})
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        pred_scopes = list(itertools.chain.from_iterable(pred_scopes))
        tar_scopes = list(itertools.chain.from_iterable(tar_scopes))
        logit_val_info = classification_report(tar_scopes, pred_scopes, output_dict=True, digits=5)
        if len(scores) == 0:
            scores = [0]
        self.logger.info(f'Mean scores: {np.mean(scores)}')
        #r = rouge.Rouge()
        #r_scores = r.get_scores(pred_scopes, tar_scopes, avg=True)
        #bleu1 = corpus_bleu([e.split() for e in tar_scopes], [e.split() for e in pred_scopes], weights=(1,0,0,0))
        #self.logger.info(f'rouge: {r_scores}, bleu1: {bleu1}')
        #return r_scores['rouge-1']['f'], r_scores['rouge-2']['f'], bleu1
        return logit_val_info

    def train_epoch(self, data_loader, valid_data, gen_only):
        pbar = tqdm(total=len(data_loader), desc='Training')
        num_labels = param.label_dim
        d_loss_monitor = AverageMeter()
        g_loss_monitor = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, padding_mask, scopes, input_len, cues, target_ids, target_padding_mask, target_subword_mask = batch
            bs = batch[0].size(0)
            active_padding_mask = padding_mask.view(-1) == 1

            if not gen_only:
                # Train descriminator
                self.d_optimizer.zero_grad()
                y = torch.full((input_ids.size(0),), 1, dtype=torch.float, device=self.device)
                dy = self.model.forward_d(target_ids, target_padding_mask).view(-1)
                d_loss_real = self.d_loss(dy, y)
                d_loss_real.backward()

                fake = self.model.forward_g(input_ids, padding_mask).argmax(-1)

                fake_y = []
                for i in range(bs):
                    pred_sent = data_loader.tokenizer.convert_ids_to_tokens(fake[i])
                    tar_sent = data_loader.tokenizer.convert_ids_to_tokens(target_ids[i])
                    temp_pred = decode_aug_seq(pred_sent, target_padding_mask[i].count_nonzero())
                    temp_tar = decode_aug_seq(tar_sent, target_padding_mask[i].count_nonzero())
                    if temp_pred == [0]:
                        temp_pred = [0] * len(temp_tar)
                    while len(temp_pred) < len(temp_tar):
                        temp_pred.append(0)
                    if len(temp_pred) > len(temp_tar):
                        temp_pred = temp_pred[:len(temp_tar)]
                    temp_y = target_weight_score(classification_report(temp_tar, temp_pred, output_dict=True, digits=5, zero_division=0), ['0','1'])[0]
                    temp = fake[i].tolist()
                    if temp.count(f'{data_loader.tokenizer.additional_special_tokens_ids[-2]}') != temp.count(f'{data_loader.tokenizer.additional_special_tokens_ids[-1]}') or \
                       temp.count(f'{data_loader.tokenizer.additional_special_tokens_ids[-3]}') != temp.count(f'{data_loader.tokenizer.additional_special_tokens_ids[-4]}') :
                        temp_y = 0
                    fake_y.append(temp_y)

                fake_y = torch.Tensor(fake_y).cuda()
                gy = self.model.forward_d(fake).view(-1)
                d_loss_fake = self.d_loss(gy, fake_y)
                d_loss_fake.backward()
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.step()
                d_loss_monitor.update(d_loss.item())
            else:
                d_loss_monitor.update(999)
            
            # Train Generator
            self.g_optimizer.zero_grad()
            gen_scope = self.model.forward_g(input_ids, padding_mask)
            g_loss = self.g_loss(gen_scope.view(-1, gen_scope.size(-1)), target_ids.view(-1))
            temp_scope = gen_scope.argmax(-1)
            g_loss.backward()
            self.g_optimizer.step()
            g_loss_monitor.update(g_loss.item())

            #scope_logits = self.model(input_ids, padding_mask)[0]
            #loss = self.criterion(scope_logits.view(-1, scope_logits.size(-1)), target_ids.view(-1))

            #if len(self.n_gpu.split(",")) >= 2:
            #    loss = loss.mean()
            
            # scale the loss
            """if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            # gradient clip
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=input_ids.size(0))"""
            pbar.update()
            pbar.set_postfix({'dloss': d_loss_monitor.avg, 'gloss': g_loss_monitor.avg})
        info = {'dloss': d_loss_monitor.avg, 'gloss': g_loss_monitor.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def train(self, train_data, valid_data, epochs, is_bert=False):
        #seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            if epoch < 7:
                gen_only = True
            else:
                gen_only = False
            #self.alpha = self.alpha - 0.01
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data, valid_data, gen_only)
            #r1, r2, b1 = self.valid_epoch(valid_data, is_bert)
            logit_val_info = self.valid_epoch(valid_data, is_bert, gen_only)
            scope_f1 = target_weight_score(logit_val_info, ['1'])
            logs = {'scope token f1': scope_f1[0]} #, 'val_scope_token_f1': scope_f1[0]}
            self.logger.info(logs)
            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            #if param.task == 'joint':
            #    self.logger.info('cue_f1 %f', cue_f1)
            #self.logger.info(logs)
            #self.logger.info('scope match %f', scope_match)
            #self.logger.info("The entity scores of valid data : ")
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=scope_f1[0], epoch=epoch)
            # save log
            #if self.training_monitor:
            #    self.training_monitor.epoch_step({'r1':r1})

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=scope_f1[0])
                self.model_checkpoint.epoch_step(current=scope_f1[0], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=scope_f1[0])
                if self.early_stopping.stop_training:
                    break


class SentTrainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.criterion = criterion
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device = DEVICE
        self.id2label = {y: x for x, y in label2id.items()}
        self.start_epoch = 1
        self.global_step = 0
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path + '\\checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_features, is_bert):
        """
        batch shape:
            [bs*][input_ids, padding_mask, scopes, input_len, cues, subword_mask]
        """
        pbar = tqdm(total=len(data_features), desc='Evaluating')
        valid_loss = AverageMeter()
        wrap_cue_pred = []
        wrap_cue_tar = []
        wrap_sent_pred = []
        wrap_sent_tar = []
        for step, f in enumerate(data_features):
            num_labels = 2
            input_ids = f[0].to(self.device)
            padding_mask = f[1].to(self.device)
            cues = f[2].to(self.device)
            cue_sep = f[3].to(self.device)
            input_lens = f[4]
            subword_mask = f[5].to(self.device)
            if param.cue_matrix:
                cue_matrix = f[6].to(self.device)
            bs = f[0].size(0)
            sent_tar = [0 if 1 not in c else 1 for c in cues]
            sent_tar = torch.LongTensor(sent_tar).to(self.device)
            self.model.eval()
            with torch.no_grad():
                #if not param.augment_cue and param.task != 'joint':
                active_padding_mask = padding_mask.view(-1) == 1
                cue_logits = self.model(input_ids, padding_mask, return_dict=True)
                loss = self.criterion(cue_logits.view(-1, num_labels), sent_tar)
            valid_loss.update(val=loss.item(), n=input_ids.size(0))
            sent_pred = cue_logits.argmax(-1)
            
            wrap_sent_pred.extend(sent_pred.cpu().tolist())
            wrap_sent_tar.extend(sent_tar.cpu().tolist())
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})

        sent_f1 = f1_score(wrap_sent_tar, wrap_sent_pred)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return sent_f1


    def train_epoch(self, data_loader, valid_data, epoch, is_bert=True, max_num_cue=4, eval_every=800):
        pbar = tqdm(total=len(data_loader), desc='Training')
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            bs = batch[0].size(0)
            num_labels = 2
            if param.cue_matrix:
                input_ids, padding_mask, cues, cue_sep, input_len, subword_mask, cue_matrix = batch
            else:
                input_ids, padding_mask, cues, cue_sep, input_len, subword_mask = batch
            active_padding_mask = padding_mask.view(-1) == 1
            sent_tar = [0 if 1 not in c else 1 for c in cues]
            sent_tar = torch.LongTensor(sent_tar).to(self.device)
        
            cue_logits = self.model(input_ids, padding_mask, return_dict=True)
            loss = self.criterion(cue_logits.view(-1, num_labels), sent_tar)

            #if len(self.n_gpu.split(",")) >= 2:
            #    loss = loss.mean()
            
            # scale the loss
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            # gradient clip
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=input_ids.size(0))
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        info = {'loss': tr_loss.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def train(self, train_data, valid_data, epochs, is_bert=True):
        #seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data, valid_data, epoch, is_bert)

            cue_log = self.valid_epoch(valid_data, is_bert)
            logs = {'loss': train_log['loss'], 'val_cue_f1': cue_log}
            score = cue_log

            #logs = dict(train_log, **cue_log['weighted avg'], **cue_sep_log['weighted avg'])
            #show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(logs)
            '''for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)'''

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=score, epoch=epoch)
            # save log
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=score)
                self.model_checkpoint.epoch_step(current=score, state=state)
                
            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(current=score)
                if self.early_stopping.stop_training:
                    break
