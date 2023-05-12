import random
import math
import copy
import gc
from typing import List, Tuple, T, Iterable, Union, NewType

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer
from transformers.file_utils import cached_path
import numpy as np
import gensim
import _pickle
from tqdm import tqdm
from util import pad_sequences, del_list_idx, get_boundary
from data import RawData
from params import param

class WikiTemp():
    def __init__(self) -> None:
        self.cues = None
        self.scopes = None


class InputExample():
    def __init__(self, guid, sent: str, subword_mask=None):
        """
            sent: whitespace separated input sequence string
        """
        self.guid = guid
        self.sent = sent
        self.subword_mask = subword_mask


class CueExample(InputExample):
    def __init__(self, cues, cue_sep, num_cues, **kwargs):
        super().__init__(**kwargs)
        self.cues = cues
        self.num_cues = num_cues
        self.cue_sep = cue_sep


class SherCueExample(CueExample):
    """Combine the n't word in sherlock and sfu dataset"""
    def __init__(self, example):
        super().__init__(guid=example.guid, sent=example.sent, subword_mask=example.subword_mask, 
                         num_cues=example.num_cues, cues=example.cues, cue_sep=example.cue_sep)
        self.cues, self.cue_sep, self.sent = self.combine_nt(self.cues, self.cue_sep, self.sent)
        self.cues = [2 if e == 3 else 1 for e in self.cues]
        self.cues = self.cues_to_bio(self.cues, self.cue_sep)

    def cues_to_bio(self, cues, cue_seps):
        """
        pad:0, B:1, I:2, O:3
        """
        prev_c = -1
        prev_cs = -1
        temp = []
        for c, cs in zip(cues, cue_seps):
            if c == 2:
                prev_c = 2
                temp.append(3)
            elif c == 1:
                if prev_c == 1:
                    if prev_cs == cs:
                        temp.append(2)
                    else:
                        temp.append(1)
                else:
                    temp.append(1)
                prev_cs = cs
        return temp
    
    def combine_nt(self, cues, cue_seps, sc_sent):
        if 'n\'t' in sc_sent[0]:
            num_cues = len(cues)
            new_cues = [[] for i in range(num_cues)]
            new_cue_seps = [[] for i in range(num_cues)]
            new_sc_sent = [[] for i in range(num_cues)]
            for i, (cue, cue_sep, sent) in enumerate(zip(cues, cue_seps, sc_sent)):
                for c, cs, token in zip(cue, cue_sep, sent):
                    if token != 'n\'t':
                        new_cues[i].append(c)
                        new_cue_seps[i].append(cs)
                        new_sc_sent[i].append(token)
                    else:
                        new_cues[i][-1] = c
                        new_cue_seps[i][-1] = cs
                        new_sc_sent[i][-1] += token
            return new_cues, new_cue_seps, new_sc_sent
        else:
            return cues, cue_seps, sc_sent
    
class ScopeExample(InputExample):
    def __init__(self, cues: List[int], scopes: List[T], sc_sent: List[str], seg: List[T], num_cues=None, **kwargs):
        super().__init__(**kwargs)
        if num_cues is None:
            self.num_cues = len(scopes)
        else:
            self.num_cues = num_cues
        self.cues = cues
        self.scopes = scopes
        self.sc_sent = sc_sent
        self.seg = seg

class SherScopeExample(ScopeExample):
    """Combine the n't word in sherlock and sfu dataset"""
    def __init__(self, example):
        super().__init__(guid=example.guid, sent=example.sent, subword_mask=example.subword_mask,
                         cues=example.cues, scopes=example.scopes, sc_sent=example.sc_sent, seg=example.seg)
        self.cues, self.scopes, self.sc_sent, self.seg = self.combine_nt(self.cues, self.scopes, self.sc_sent, self.seg)
    
    def combine_nt(self, cues, scopes, sc_sent, seg):
        if 'n\'t' in sc_sent[0]:
            num_cues = len(cues)
            new_cues = [[] for i in range(num_cues)]
            new_scopes = [[] for i in range(num_cues)]
            new_sc_sent = [[] for i in range(num_cues)]
            new_seg = [[] for i in range(num_cues)]
            for i, (cue, scope, sent) in enumerate(zip(cues, scopes, sc_sent)):
                for c, s, token, ss in zip(cue, scope, sent, seg):
                    if token != 'n\'t':
                        new_cues[i].append(c)
                        new_scopes[i].append(s)
                        new_sc_sent[i].append(token)
                        new_seg[i].append(ss)
                    else:
                        new_cues[i][-1] = c
                        new_scopes[i][-1] = s
                        new_sc_sent[i][-1] += token
                        new_seg[i][-1] = ss
            return new_cues, new_scopes, new_sc_sent, new_seg[0]
        else:
            return cues, scopes, sc_sent, seg
    
    def mark_affix_scope(self, cues, scopes):
        new_scopes = scopes.copy()
        for i, (cue, scope) in enumerate(zip(cues, scopes)):
            if 4 in cue:
                new_scope = scope.copy()
                for j, c in enumerate(cue):
                    if c == 4:
                        new_scope[j] = 1
                new_scopes[i] = new_scope
        return new_scopes
            
ExampleLike = Union[CueExample, ScopeExample, InputExample, SherScopeExample]


class CueFeature():
    def __init__(self, guid, or_sent, sent, input_ids, padding_mask, subword_mask, input_len, cues, cue_sep, num_cues, token_type_ids):
        self.guid = guid
        self.or_sent = or_sent
        self.sent = sent
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.cue_sep = cue_sep
        self.num_cues = num_cues
        self.token_type_ids = token_type_ids


class ScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len,
                 cues, scopes, num_cues, seg):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.scopes = scopes
        self.num_cues = num_cues
        self.seg = seg

class MultiScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len, 
                 cues, scopes, num_cues, master_mat=None):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.num_cues = num_cues
        self.cues = cues
        self.scopes = scopes
        self.master_mat = master_mat

class PipelineScopeFeature(object):
    def __init__(self, guid, or_sent, sents, input_ids, padding_mask, subword_mask, input_len, cues, cue_match, gold_scopes, gold_num_cues):
        self.guid = guid
        self.or_sent = or_sent
        self.sents = sents
        self.input_ids = input_ids
        self.padding_mask = padding_mask
        self.subword_mask = subword_mask
        self.input_len = input_len
        self.cues = cues
        self.cue_match = cue_match
        self.gold_scopes = gold_scopes
        self.gold_num_cues = gold_num_cues

class AugFeature(ScopeFeature):
    def __init__(self, feat) -> None:
        super().__init__(feat.guid, feat.or_sent, feat.sents, feat.input_ids, feat.padding_mask,
                         feat.subword_mask, feat.input_len, feat.cues, feat.scopes, feat.num_cues, feat.seg)
        self.aug_sent = []
        self.target_ids = []
        self.target_padding_mask = []
        self.target_subword_mask = []

    def augment(self, temp_sent, temp_scope, temp_subword):
        cue_id = [i for i, x in enumerate(temp_scope) if x == 3]
        cue_tok = []
        for i in cue_id:
            cue_tok.append(temp_sent[i])
        cue_tok = cue_tok[1:-1]
        cue_subword = [1 if i == 0 else 0 for i, _ in enumerate(cue_tok)]
        prev = 2
        pos = 0
        ## Mapping for additional special token:
        ## [ScopeStart] -> [unused50] (50)
        ## [ScopeEnd] -> [unused51] (51)
        ## [CueIs] -> [unused52] (52)
        ## [CueIsEnd] -> [unused53] (53)
        ## Sentence will be re-formulated as:
        ## This is [unused1] not [unused1] an apple. ->
        ## This is [unused1] not [unused1] [ScopeStart] an apple [CueIs] not [CueIsEnd] [ScopeEnd].
        new_text = []
        new_subwords = []
        for token, scope, sub in zip(temp_sent, temp_scope, temp_subword):
            if scope == 1:
                if pos != 0 and pos != len(temp_sent)-1:
                    if scope == prev:
                        # continued cue, don't care is subword or multi word cue
                        new_subwords.append(sub)
                        new_text.append(token)
                    else:
                        # left bound
                        new_text.append('[scope]')
                        new_subwords.append(1)
                        new_text.append(token)
                        new_subwords.append(1)
                elif pos == 0:
                    # at pos 0
                    new_text.append('[scope]')
                    new_subwords.append(1)
                    new_text.append(token)
                    new_subwords.append(1)
                else:
                    # at eos
                    new_text.append(token)
                    new_subwords.append(sub)
                    #new_text.append('[unused52]')
                    #new_subwords.append(1)
                    #new_text.extend(cue_tok)
                    #new_subwords.extend(cue_subword)
                    #new_text.append('[unused53]')
                    #new_subwords.append(1)
                    new_text.append('[\\scope]')
                    new_subwords.append(1)
            else:
                if prev == 1:
                    # end of scope, insert right bound before current pos
                    #new_text.append('[unused52]')
                    #new_subwords.append(1)
                    #new_text.extend(cue_tok)
                    #new_subwords.extend(cue_subword)
                    #new_text.append('[unused53]')
                    #new_subwords.append(1)
                    new_text.append('[\\scope]')
                    new_subwords.append(1)
                    new_text.append(token)
                    new_subwords.append(sub)
                else:
                    new_text.append(token)
                    new_subwords.append(sub)
            prev = scope
            pos += 1
        return new_text, new_subwords

class EDUFeature(ScopeFeature):
    def __init__(self, feat) -> None:
        super().__init__(feat.guid, feat.or_sent, feat.sents, feat.input_ids, feat.padding_mask,
                         feat.subword_mask, feat.input_len, feat.cues, feat.scopes, feat.num_cues, feat.seg)
        
    def append_seg(self):
        temp_sent = self.sents.copy()
        temp_scope = self.scopes.copy()
        temp_cue = self.cues.copy()
        temp_subword = self.subword_mask.copy()
        temp_seg = self.seg[0].copy()
        
        wrap_sents = []
        wrap_input_id = []
        wrap_scopes = []
        wrap_cues = []
        wrap_subwords = []
        wrap_padding_mask = []
        wrap_input_len = []
        for c in range(self.num_cues):
            prev = 0
            prev_scope = -1
            prev_cue = -1
            pos = 0
            
            new_text = []
            new_subwords = []
            new_scopes = []
            new_cues = []
            for token, scope, cue, sub, ss in zip(temp_sent[c], temp_scope[c], temp_cue[c], temp_subword[c], temp_seg[c]):
                if ss != prev:
                    new_text.append(['[SEP]'])
                    new_subwords.append(1)
                    new_scopes.append(prev_scope)
                    new_cues.append(prev_cue)
                    new_text.append(token)
                    new_subwords.append(sub)
                    new_scopes.append(scope)
                    new_cues.append(cue)
                prev_scope = scope
                prev_cue = cue
                prev = ss
                pos += 1
            input_ids = self.tokenizer.convert_tokens_to_ids(new_text)
            padding_mask = [1] * len(input_ids)
            input_len = len(input_ids)
            wrap_sents.append(new_text)
            wrap_input_id.append(input_ids)
            wrap_subwords.append(new_subwords)
            wrap_cues.append(new_cues)
            wrap_scopes.append(new_scopes)
            wrap_padding_mask.append(padding_mask)
            wrap_input_len.append(input_len)
        self.input_ids = wrap_input_id
        self.sents = wrap_sents
        self.subword_mask = wrap_subwords
        self.input_len = wrap_input_len
        self.cues = wrap_cues
        self.scopes = wrap_scopes
        self.padding_mask = wrap_padding_mask


FeatureLike = Union[CueFeature, ScopeFeature, PipelineScopeFeature, MultiScopeFeature]

bio_to_id = {'<PAD>': 0, 'B': 1, 'I': 2, 'O': 3, 'C':4}

def scope_to_bio(scope):
    """
    Label: 1: scope, 2: O, 3: cue
    BIO: 1: Bs, 2:Ic, 3:O, 4:cue
    """
    temp = []
    prev = -1
    for i, s in enumerate(scope):
        if i == 0:
            # at pos 0, always [CLS] token
            temp.append(3)
            prev = s
            continue
        if s in [1, 3]:
            if s != prev:
                # cue followed by scope / scope followed by cue, not a B
                if (s == 1 and prev == 3):
                    prev = s
                    temp.append(2)
                elif (s == 3 and prev == 1):
                    prev = s
                    temp.append(4)
                else:
                    prev = s
                    temp.append(1)
            else:
                if s == 3:
                    prev = s
                    temp.append(4)
                else:
                    prev = s
                    temp.append(2)
        else:
            prev = s
            temp.append(3)
    return temp
            

def single_cue_to_matrix_pad(cues: List, input_len: int) -> torch.LongTensor:
    mat = np.zeros((param.max_len, param.max_len), dtype=np.int)
    for i in range(input_len):
        if cues[i] != 2:
            for j in range(input_len):
                mat[i][j] = cues[j]
        else:
            for j in range(input_len):
                mat[i][j] = 2
    return torch.LongTensor(mat)

def single_scope_to_link_matrix_pad(scope: List, cues: List, input_len: int,
                                    mode=param.m_dir, cue_mode=param.cue_mode) -> torch.LongTensor:
    """
    To convert the scope list (single cue) to a link matrix that represents
    the relation (undirected) link between eachother token.
    cue_mode diag:
        Cue <-> Scope: 1
        Noncue <-> Noncue: 2
        Cue (ROOT) <-> Cue: 3
        Pad: 0
    
    cue_mode root:
        Outward link: 1 (cue -> scope, ROOT -> cue)
        Not linked / inward link: 2 (scope -> cue, scope -> scope)
        Pad: 0

    params:
        mode: ['d1', 'd2', 'ud'] d for directed, ud for undirected
        cue_mode: ['root', 'diag'] 'root' for adding an additional dummy [ROOT] token
        to indicate the link between [ROOT] and cue, avoiding labelling diagonal element of matrix
            'diag' for labelling diagonal
    """
    temp_scope = []
    for i, e in enumerate(scope):
        if e == 2:
            if cues[i] != 3:
                temp_scope.append(3)
            else:
                temp_scope.append(e)
        else:
            temp_scope.append(e)
    mat_dim = param.max_len
    mat = np.zeros((mat_dim, mat_dim), dtype=np.int)
    if cue_mode == 'root':
        # for root mode, force the direction to be undirected (d2)
        for i in range(input_len):
            # scan through the matrix by row to fill
            if scope[i] == 3:
                # The row at cue
                mat[0][i] = 1
                mat[i][0] = 2
                for j in range(input_len):
                    if scope[j] != 3:
                        mat[i][j] = scope[j]
                    else:
                        mat[i][j] = 2
            else:
                mat[0][i] = 2
                for j in range(input_len):
                    if scope[j] == 3:
                        if scope[i] == 1:
                            mat[i][j] = 2
                        else:
                            mat[i][j] = scope[i]
                    else:
                        mat[i][j] = 2
            mat[i][i] = 2
    elif cue_mode == 'diag':
        # scan through the matrix by row to fill
        for i in range(input_len):
            if scope[i] == 3:
                # The row at cue
                for j in range(input_len):
                    mat[i][j] = scope[j]
            else:
                for j in range(input_len):
                    if scope[j] == 3:
                        if scope[i] == 1 and mode == 'd1':
                            mat[i][j] = 4
                        elif scope[i] == 1 and mode == 'd2':
                            mat[i][j] = 2
                        else:
                            mat[i][j] = scope[i]
                    else:
                        mat[i][j] = 2
    mat = torch.LongTensor(mat)
    return mat

def multi_scope_to_link_matrix_pad(scopes: List[List], cues: List[List], input_lens: List[int], 
                                   mode=param.m_dir, cue_mode=param.cue_mode) -> np.ndarray:
    """
    To convert the scope list (single cue) to a link matrix that represents
    the directed relation link between eachother token.
    Outward link: 1 (cue -> scope)
        Not linked / inward link: 2 (scope -> cue, scope -> scope)
        3 (cue -> cue)
        Pad: 0
    params:
        similar to single case, except cue_mode is fixed to 'diag'
    """
    num_cues = len(cues)
    if num_cues == 1:
        return single_scope_to_link_matrix_pad(scopes[0], cues[0], input_lens[0], 'd2')
    else:

        mat_dim = param.max_len
        all_sub_matrix = []
        for c in range(num_cues):
            mat = np.zeros((mat_dim, mat_dim), dtype=np.int)
            scope = scopes[c]
            input_len = input_lens[c]
            if cue_mode == 'root':
                # for root mode, force the direction to be undirected (d2)
                for i in range(input_len):
                    # scan through the matrix by row to fill
                    if scope[i] == 3:
                        # The row at cue
                        mat[0][i] = 1
                        mat[i][0] = 2
                        for j in range(input_len):
                            if scope[j] != 3:
                                mat[i][j] = scope[j]
                            else:
                                mat[i][j] = 2
                    else:
                        mat[0][i] = 2
                        for j in range(input_len):
                            if scope[j] == 3:
                                if scope[i] == 1:
                                    mat[i][j] = 2
                                else:
                                    mat[i][j] = scope[i]
                            else:
                                mat[i][j] = 2
                    mat[i][i] = 2
            elif cue_mode == 'diag':
                # scan through the matrix by row to fill
                for i in range(input_len):
                    if scope[i] == 3:
                        # The row at cue
                        for j in range(input_len):
                            mat[i][j] = scope[j]
                    else:
                        for j in range(input_len):
                            if scope[j] == 3:
                                if scope[i] == 1 and mode == 'd1':
                                    mat[i][j] = 4
                                elif scope[i] == 1 and mode == 'd2':
                                    mat[i][j] = 2
                                else:
                                    mat[i][j] = scope[i]
                            else:
                                mat[i][j] = 2
            all_sub_matrix.append(mat)
        master_mat = np.zeros((mat_dim, mat_dim), dtype=np.int)
        for i in range(len(scopes[c])):
            for j in range(len(scopes[c])):
                master_mat[i][j] = 2
        for m in all_sub_matrix:
            for i in range(len(scopes[c])):
                for j in range(len(scopes[c])):
                    if m[i][j] == 1:
                        master_mat[i][j] = 1
                    if m[i][j] == 3:
                        master_mat[i][j] = 3
        master_mat = torch.LongTensor(master_mat)
    return master_mat

class Processor(object):
    def __init__(self):
        self.tokenizer = None

    @classmethod
    def read_data(cls, input_file, dataset_name=None) -> RawData:
        return RawData(input_file, dataset_name=dataset_name)

    def create_examples(self, data: RawData, example_type: str, dataset_name: str, cue_or_scope: str, cached_file=None,
                        test_size=0.15, val_size=0.15) -> Union[List[ExampleLike], Tuple[List, List, List]]:
        """
        Create packed example format for input data. Do train-test split if specified.

        params:
            data (RawData): Though it's not "raw". Already contains tag information
            example_type (str): "train", "test", "dev", "split". If set as split, 
                will perform train-test split as well. 
            cue_or_scope (str): cue or scope.
            cached_file (NoneType | str): if specified, save the packed examples to cache file.
        returns:
            examples (list[ExampleLike]): example of specified type
            examples (cues, scopes)>>>(tuple[tuple[train, dev, test], tuple[train, dev, test]]): overload for split.
        """
        assert example_type.lower() in [
            'train', 'test', 'dev', 'split', 'joint_train', 'joint_dev', 'joint_test', 'joint'], 'Wrong example type.'
        assert cue_or_scope in [
            'cue', 'scope', 'raw'], 'cue_or_scope: Must specify cue of scope, or raw to perform split and get vocab'

        cue_examples = []
        non_cue_examples = []
        non_cue_scopes = []
        scope_examples = []
        cue_i = 0
        non_cue_i = 0
        if dataset_name == 'sherlock':
            if example_type.startswith('joint'):
                t = example_type[6:]
            else:
                t = example_type
            seg_file = param.seg_path[dataset_name][t]
            with open(seg_file, 'rb') as sf:
                seg = _pickle.load(sf)
        elif dataset_name != 'wiki' and dataset_name != 'vet':
            seg_file = param.seg_path[dataset_name]
            with open(seg_file, 'rb') as sf:
                seg = _pickle.load(sf)
        if dataset_name == 'wiki':
            temp_data = data
            data = WikiTemp()
            data.cues = temp_data
        for i, _ in enumerate(data.cues[0]):
            sentence = data.cues[0][i]
            cues = data.cues[1][i]
            cue_sep = data.cues[2][i]
            num_cues = data.cues[3][i]
            sent = ' '.join(sentence)
            if num_cues > 0:
                guid = '%s-%d' % (example_type, cue_i)
                cue_i += 1
                cue_examples.append(CueExample(guid=guid, sent=sent, cues=cues,
                                            cue_sep=cue_sep, num_cues=num_cues, subword_mask=None))
            else:
                guid = '%s-%d' % (example_type, non_cue_i)
                non_cue_i += 1
                non_cue_examples.append(CueExample(guid='nc'+guid, sent=sent, cues=cues,
                                            cue_sep=cue_sep, num_cues=num_cues, subword_mask=None))
                t_seg = [0 for _ in sentence]
                non_cue_scopes.append(ScopeExample(guid='nc'+guid, sent=sent, cues=[cues], num_cues=0,
                                            scopes=[[2 for c in cues]], sc_sent=[sentence], subword_mask=None, seg=t_seg))

        for i, _ in enumerate(data.scopes[0]):
            guid = '%s-%d' % (example_type, i)
            or_sent = data.scopes[0][i]
            sentence = data.scopes[1][i]
            cues = data.scopes[2][i]
            sent = ' '.join(or_sent)
            scopes = data.scopes[3][i]
            t_seg = [0 for _ in sentence[0]]
            scope_examples.append(ScopeExample(guid=guid, sent=sent, cues=cues,
                                               scopes=scopes, sc_sent=sentence, subword_mask=None, seg=t_seg))#seg[i]))
        
        if example_type.lower() in ('train', 'test', 'dev'):
            if cue_or_scope.lower() == 'cue':
                cue_examples.extend(non_cue_examples)
                if cached_file is not None:
                    print('Saving examples into cached file %s', cached_file)
                    torch.save(cue_examples, cached_file)
                return cue_examples
            elif cue_or_scope.lower() == 'scope':
                if cached_file is not None:
                    print('Saving examples into cached file %s', cached_file)
                    torch.save(scope_examples, cached_file)
                return scope_examples
        elif example_type.lower() == 'split':
            # Note: if set example type to split, will return both cue and scope examples
            
            scope_len = len(scope_examples)
            train_len = math.floor((1 - test_size - val_size) * scope_len)
            test_len = math.floor(test_size * scope_len)
            val_len = scope_len - train_len - test_len
            scope_index = list(range(scope_len))
            train_index = random.sample(scope_index, k=train_len)
            scope_index = del_list_idx(scope_index, train_index)
            test_index = random.sample(scope_index, k=test_len)
            scope_index = del_list_idx(scope_index, test_index)
            dev_index = scope_index.copy()

            train_cue = [cue_examples[i] for i in train_index]
            test_cue = [cue_examples[i] for i in test_index]
            dev_cue = [cue_examples[i] for i in dev_index]
            train_scope = [scope_examples[i] for i in train_index]
            test_scope = [scope_examples[i] for i in test_index]
            dev_scope = [scope_examples[i] for i in dev_index]

            random_state = np.random.randint(1, 2020)
            tr_nocue_, te_nocue = train_test_split(
                non_cue_examples, test_size=test_size, random_state=random_state)
            _, te_non_cue_sents = train_test_split(
                data.non_cue_sents, test_size=test_size, random_state=random_state)
            random_state2 = np.random.randint(1, 2020)
            tr_nocue, de_nocue = train_test_split(tr_nocue_, test_size=(
                val_size / (1 - test_size)), random_state=random_state2)
            train_cue.extend(tr_nocue)
            dev_cue.extend(de_nocue)
            test_cue.extend(te_nocue)
            for c, _ in enumerate(train_cue):
                train_cue[c].guid = f'train-{c}'
            for c, _ in enumerate(test_cue):
                test_cue[c].guid = f'test-{c}'
            for c, _ in enumerate(dev_cue):
                dev_cue[c].guid = f'dev-{c}'
            for c, _ in enumerate(train_scope):
                train_scope[c].guid = f'train-{c}'
            for c, _ in enumerate(test_scope):
                test_scope[c].guid = f'test-{c}'
            for c, _ in enumerate(dev_scope):
                dev_scope[c].guid = f'dev-{c}'
            if cached_file is not None:
                print('Saving examples into cached file %s', cached_file)
                torch.save(train_cue, f'{param.base_path}/split/train_cue_{cached_file}')
                torch.save(test_cue, f'{param.base_path}/split/test_cue_{cached_file}')
                torch.save(dev_cue, f'{param.base_path}/split/dev_cue_{cached_file}')
                torch.save(train_scope, f'{param.base_path}/split/train_scope_{cached_file}')
                torch.save(test_scope, f'{param.base_path}/split/test_scope_{cached_file}')
                torch.save(dev_scope, f'{param.base_path}/split/dev_scope_{cached_file}')
                torch.save(te_non_cue_sents, f'{param.base_path}/split/ns_{cached_file}')
            return (train_cue, dev_cue, test_cue), (train_scope, dev_scope, test_scope)
        elif 'joint' in example_type.lower():
            if dataset_name.lower() == 'sherlock':
                if cue_or_scope.lower() == 'cue':
                    cue_examples.extend(non_cue_examples)
                    return cue_examples
                elif cue_or_scope.lower() == 'scope':
                    scope_examples.extend(non_cue_scopes)
                    return scope_examples
            else:
                scope_len = len(scope_examples)
                train_len = math.floor((1 - test_size - val_size) * scope_len)
                test_len = math.floor(test_size * scope_len)
                val_len = scope_len - train_len - test_len
                scope_index = list(range(scope_len))
                train_index = random.sample(scope_index, k=train_len)
                scope_index = del_list_idx(scope_index, train_index)
                test_index = random.sample(scope_index, k=test_len)
                scope_index = del_list_idx(scope_index, test_index)
                dev_index = scope_index.copy()

                train_cue = [cue_examples[i] for i in train_index]
                test_cue = [cue_examples[i] for i in test_index]
                dev_cue = [cue_examples[i] for i in dev_index]
                train_scope = [scope_examples[i] for i in train_index]
                test_scope = [scope_examples[i] for i in test_index]
                dev_scope = [scope_examples[i] for i in dev_index]

                random_state = np.random.randint(1, 2020)
                tr_nocue_, te_nocue = train_test_split(
                    non_cue_examples, test_size=test_size, random_state=random_state)
                random_state2 = np.random.randint(1, 2020)
                tr_nocue, de_nocue = train_test_split(tr_nocue_, test_size=(
                    val_size / (1 - test_size)), random_state=random_state2)

                tr_nocue_s_, te_nocue_s = train_test_split(
                    non_cue_scopes, test_size=test_size, random_state=random_state)
                tr_nocue_s, de_nocue_s = train_test_split(tr_nocue_s_, test_size=(
                    val_size / (1 - test_size)), random_state=random_state2)
                train_cue.extend(tr_nocue)
                dev_cue.extend(de_nocue)
                test_cue.extend(te_nocue)
                train_scope.extend(tr_nocue_s)
                dev_scope.extend(de_nocue_s)
                test_scope.extend(te_nocue_s)
                for c, _ in enumerate(train_cue):
                    train_cue[c].guid = f'train-{c}'
                for c, _ in enumerate(test_cue):
                    test_cue[c].guid = f'test-{c}'
                for c, _ in enumerate(dev_cue):
                    dev_cue[c].guid = f'dev-{c}'
                for c, _ in enumerate(train_scope):
                    train_scope[c].guid = f'train-{c}'
                for c, _ in enumerate(test_scope):
                    test_scope[c].guid = f'test-{c}'
                for c, _ in enumerate(dev_scope):
                    dev_scope[c].guid = f'dev-{c}'
                if cached_file is not None:
                    print('Saving examples into cached file %s', cached_file)
                    torch.save(train_cue, f'{param.base_path}/split/joint_train_cue_{cached_file}')
                    torch.save(test_cue, f'{param.base_path}/split/joint_test_cue_{cached_file}')
                    torch.save(dev_cue, f'{param.base_path}/split/joint_dev_cue_{cached_file}')
                    torch.save(train_scope, f'{param.base_path}/split/joint_train_scope_{cached_file}')
                    torch.save(test_scope, f'{param.base_path}/split/joint_test_scope_{cached_file}')
                    torch.save(dev_scope, f'{param.base_path}/split/joint_dev_scope_{cached_file}')
                return (train_cue, dev_cue, test_cue), (train_scope, dev_scope, test_scope)

    def ex_combine_nt(self, data: List[ExampleLike], cue_or_scope: str) -> List[ExampleLike]:
        tmp_data = []
        if cue_or_scope == 'cue':
            for item in data:
                tmp_data.append(SherCueExample(item))
        elif cue_or_scope == 'scope':
            for item in data:
                tmp_data.append(SherScopeExample(item))
        else:
            raise NameError("Need to specify either cue or scope")
        return tmp_data

    def load_examples(self, file: str):
        """
        Load a pre-saved example binary file. Or anything else.

        Warning: torch.load() uses pickle module implicitly, which is known to be insecure. 
        It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.
        Never load data that could have come from an untrusted source, or that could have been tampered with.
        Only load data you trust.
        """
        return torch.load(file)
    
    def create_feature_from_example(self, example: ExampleLike, cue_or_scope: str,
                                    max_seq_len: int = 128, is_bert=False) -> List[FeatureLike]:
        """
        Tokenize, convert (sub)words to ids, and augmentation for cues
        """
        features = []
        if cue_or_scope == 'scope':
            guid = example.guid
            wrap_input_id = []
            wrap_subword_mask = []
            wrap_sents = []
            wrap_cues = []
            wrap_scopes = []
            wrap_padding_mask = []
            wrap_input_len = []
            wrap_seg = []
            num_cue = example.num_cues
            if param.task == 'joint':
                if num_cue == 0:
                    num_cue = 1
            for c in range(num_cue):
                sent = example.sc_sent[c]
                cues = example.cues[c]
                scopes = example.scopes[c]
                seg = [0]*len(example.scopes[c])
                if is_bert:
                    # For BERT model
                    """temp_sent = []
                    temp_scope = []
                    temp_cues = []
                    temp_mask = []
                    temp_seg = []
                    for word, cue, scope, ss in zip(sent, cues, scopes, seg):
                        subwords = self.tokenizer.tokenize(word)
                        for count, subword in enumerate(subwords):
                            mask = 1
                            if count > 0:
                                mask = 0
                            temp_mask.append(mask)
                            if param.ignore_multiword_cue:
                                if cue == 2:
                                    cue = 1
                            temp_cues.append(cue)
                            temp_scope.append(scope)
                            temp_sent.append(subword)
                            temp_seg.append(ss)"""


                    full = self.tokenizer(sent, is_split_into_words=True)
                    temp_sent = self.tokenizer.convert_ids_to_tokens(full['input_ids'])
                    

                    temp_mask = []
                    temp_scope = []
                    temp_cues = []
                    temp_seg = []
                    tok_counter = 0
                    hold = ''
                    prev_counter = 999
                    temp_newsent = []
                    for i, tok in enumerate(temp_sent):
                        # Assign special token subword mask -1
                        # '[NP]' needs to be treated differently, as part of word
                        if tok in [self.tokenizer.cls_token, self.tokenizer.sep_token]:
                            temp_scope.append(2)
                            temp_mask.append(1)
                            temp_cues.append(3)
                            temp_seg.append(0)
                            continue
                        if sent[tok_counter] == '':
                            tok_counter+=1
                        if tok.startswith('##'):
                            tok = tok[2:]
                        if tok.startswith('Ġ'):
                            tok = tok[1:]
                        if tok.startswith('▁'):
                            tok = tok[1:]
                        # If BERT token matches simple split token, append position as subword mask
                        if sent[tok_counter] == tok:
                            temp_mask.append(1)
                            temp_scope.append(scopes[tok_counter])
                            temp_cues.append(cues[tok_counter])
                            temp_seg.append(seg[tok_counter])
                            temp_newsent.append(tok)
                            prev_counter = tok_counter
                            tok_counter+=1
                            hold = ''
                        # Else, combine the next BERT token until there is a match (e.g.: original: "Abcdefgh", BERT: "Abc", "def", "gh")
                        # each subword of full word are assigned same full word position
                        else:
                            hold+=tok
                            temp_mask.append(1 if prev_counter != tok_counter else 0)
                            temp_scope.append(scopes[tok_counter])
                            temp_cues.append(cues[tok_counter])
                            temp_seg.append(seg[tok_counter])
                            prev_counter = tok_counter
                            if sent[tok_counter] == hold:
                                temp_newsent.append(hold)
                                hold = ''
                                tok_counter+=1
                        # if combined token length larger than 50, most likely something wrong happened
                        if len(hold)>50:
                            err = True
                    new_text = []
                    new_cues = []
                    new_scopes = []
                    new_masks = []
                    new_seg = []
                    pos = 0
                    prev = 3
                    prevl = 0
                    for token, cue, label, mask, ss in zip(temp_sent, temp_cues, temp_scope, temp_mask, temp_seg):
                        # Process the cue augmentation.
                        # Different from the original repo, the strategy is indicate the cue border
                        if param.augment_cue == 'surround':
                            if cue == 0:
                                new_masks.append(mask)
                                new_text.append(token)
                                new_scopes.append(label)
                                new_cues.append(cue)
                                new_seg.append(ss)
                                prev = 0
                            elif cue != 3:
                                if pos != 0 and pos != len(temp_sent)-1:
                                    if cue == prev or prev == 0:
                                        # continued cue, don't care is subword or multi word cue
                                        new_masks.append(mask)
                                        new_text.append(token)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                        new_seg.append(ss)
                                    else:
                                        # left bound
                                        new_text.append('[cue]')
                                        new_cues.append(cue)
                                        new_masks.append(1)
                                        new_scopes.append(label)
                                        new_seg.append(ss)
                                        new_text.append(token)
                                        new_masks.append(0)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                        new_seg.append(ss)
                                elif pos == 0:
                                    # at pos 0
                                    new_text.append('[cue]')
                                    new_masks.append(1)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                                    new_text.append(token)
                                    new_masks.append(0)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                                else:
                                    # at eos
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                                    new_text.append('[\cue]')
                                    new_masks.append(0)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                            else:
                                if cue == prev or prev == 0:
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                                else:
                                    # current non cue, insert right bound before current pos
                                    new_text.append('[\cue]')
                                    new_masks.append(0)
                                    new_scopes.append(prevl)
                                    new_cues.append(prev)
                                    new_seg.append(prevl)
                                    new_text.append(token)
                                    new_masks.append(mask)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                            prev = cue
                            prevl = label
                            pos += 1
                        elif param.augment_cue == 'front':
                            if cue != 3:
                                if pos != 0:
                                    if cue == prev:
                                        # continued cue, don't care is subword or multi word cue
                                        new_masks.append(mask)
                                        new_text.append(token)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                        new_seg.append(ss)
                                    else:
                                        # left bound
                                        new_text.append('[cue]')
                                        new_cues.append(cue)
                                        new_masks.append(1)
                                        new_scopes.append(label)
                                        new_seg.append(ss)
                                        new_text.append(token)
                                        new_masks.append(0)
                                        new_scopes.append(label)
                                        new_cues.append(cue)
                                        new_seg.append(ss)
                                else:
                                    # at pos 0
                                    new_text.append('[cue]')
                                    new_masks.append(1)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                                    new_text.append(token)
                                    new_masks.append(0)
                                    new_scopes.append(label)
                                    new_cues.append(cue)
                                    new_seg.append(ss)
                            else:
                                new_text.append(token)
                                new_masks.append(mask)
                                new_scopes.append(label)
                                new_cues.append(cue)
                                new_seg.append(ss)
                            prev = cue
                            prevl = ss
                            pos += 1
                        elif param.augment_cue is False:
                            new_masks.append(mask)
                            new_text.append(token)
                            new_scopes.append(label)
                            new_cues.append(cue)
                            new_seg.append(ss)

                    if len(new_text) >= max_seq_len - 1:
                        new_text = new_text[0:(max_seq_len - 2)]
                        new_cues = new_cues[0:(max_seq_len - 2)]
                        new_masks = new_masks[0:(max_seq_len - 2)]
                        new_scopes = new_scopes[0:(max_seq_len - 2)]
                        new_seg = new_seg[0:(max_seq_len - 2)]
                    #new_text.insert(0, self.tokenizer.cls_token)
                    #new_text.append(self.tokenizer.sep_token)
                    #new_masks.insert(0, 1)
                    #new_masks.append(1)
                    #new_cues.insert(0, 3)
                    #new_cues.append(3)
                    #new_scopes.insert(0, 2)
                    #new_scopes.append(2)
                    #new_seg.append(new_seg[-1])
                    #new_seg.insert(0, 0)
                    if param.cue_mode == 'root':
                        new_text.insert(1, '[ROOT]')
                        new_masks.insert(1, 1)
                        new_cues.insert(1, 3)
                        new_scopes.insert(1, 2)
                        new_seg.insert(1, 0)
                    input_ids = self.tokenizer.convert_tokens_to_ids(
                        new_text)
                    padding_mask = [1] * len(input_ids)
                    input_len = len(input_ids)
                    wrap_sents.append(new_text)
                    wrap_input_id.append(input_ids)
                    wrap_subword_mask.append(new_masks)
                    wrap_cues.append(new_cues)
                    wrap_scopes.append(new_scopes)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)
                    wrap_seg.append(new_seg)
                else:
                    # For non-BERT (non-BPE tokenization)
                    sent = example.sc_sent[c].copy()
                    cues = example.cues[c].copy()
                    if param.ignore_multiword_cue:
                        for i, c in enumerate(cues):
                            if c == 2:
                                cues[i] = 1
                    scopes = example.scopes[c].copy()
                    segs = example.seg.copy()

                    words = self.tokenizer.tokenize(sent)
                    words.insert(0, '[CLS]')
                    words.append('[SEP]')
                    cues.insert(0, 3)
                    cues.append(3)
                    scopes.insert(0, 2)
                    scopes.append(2)
                    segs.append(0)
                    segs.insert(0, 0)
                    input_ids = self.tokenizer.convert_tokens_to_ids(words)
                    padding_mask = [1] * len(input_ids)
                    input_len = len(input_ids)
                    wrap_sents.append(words)
                    wrap_input_id.append(input_ids)
                    wrap_subword_mask = None
                    wrap_cues.append(cues)
                    wrap_scopes.append(scopes)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)
                    wrap_seg.append(segs)
                    #assert all(each_len == len(words) for each_len in seq_len)

            feature = ScopeFeature(guid=guid, or_sent=example.sent, sents=wrap_sents, input_ids=wrap_input_id,
                                   padding_mask=wrap_padding_mask, subword_mask=wrap_subword_mask,
                                   input_len=wrap_input_len, cues=wrap_cues, scopes=wrap_scopes, num_cues=example.num_cues, seg=wrap_seg)

            features.append(feature)
        else:
            ## Cue
            sent = example.sent.split()
            guid = example.guid
            num_cues = example.num_cues
            if is_bert:
                # For BERT model
                query = ['Which', 'words', 'express', 'negation', '?']
                full = self.tokenizer(sent, query, is_split_into_words=True)
                sent += query
                temp_sent = self.tokenizer.convert_ids_to_tokens(full['input_ids'])
                token_type_ids = full['token_type_ids']
                cues = example.cues
                cuesep = example.cue_sep
                cues += [3, 3, 3, 3, 3]
                cuesep += [0, 0, 0, 0, 0]
                temp_mask = []
                temp_cues = []
                temp_cuesep = []
                tok_counter = 0
                hold = ''
                prev_counter = 999
                new_word = False
                temp_newsent = []
                for i, tok in enumerate(temp_sent):
                    # Assign special token subword mask -1
                    # '[NP]' needs to be treated differently, as part of word
                    if tok in [self.tokenizer.cls_token, self.tokenizer.sep_token]:
                        temp_mask.append(1)
                        temp_cues.append(3)
                        temp_cuesep.append(0)
                        continue
                    if sent[tok_counter] == '':
                        tok_counter+=1
                    if tok.startswith('##'):
                        tok = tok[2:]
                    if tok.startswith('Ġ'):
                        tok = tok[1:]
                    if tok.startswith('▁'):
                        tok = tok[1:]
                    # If BERT token matches simple split token, append position as subword mask
                    if sent[tok_counter] == tok:
                        temp_mask.append(1)
                        temp_cues.append(cues[tok_counter])
                        temp_cuesep.append(cuesep[tok_counter])
                        temp_newsent.append(tok)
                        prev_counter = tok_counter
                        tok_counter+=1
                        hold = ''
                        new_word = True
                    # Else, combine the next BERT token until there is a match (e.g.: original: "Abcdefgh", BERT: "Abc", "def", "gh")
                    # each subword of full word are assigned same full word position
                    else:
                        hold+=tok
                        temp_mask.append(1 if prev_counter != tok_counter else 0)
                        temp_cues.append(2 if cues[tok_counter] == 1 and not new_word else cues[tok_counter])
                        #if 2 in temp_cues:
                        #    print()
                        temp_cuesep.append(cuesep[tok_counter])
                        prev_counter = tok_counter
                        new_word = False
                        if sent[tok_counter] == hold:
                            temp_newsent.append(hold)
                            hold = ''
                            tok_counter+=1
                            new_word = True

                    # if combined token length larger than 50, most likely something wrong happened
                    if len(hold)>50:
                        err = True
                
                new_text = temp_sent
                new_cues = temp_cues
                new_cuesep = temp_cuesep
                subword_mask = temp_mask
                if len(new_text) >= max_seq_len - 1:
                    new_text = new_text[0:(max_seq_len - 2)]
                    new_cues = new_cues[0:(max_seq_len - 2)]
                    new_cuesep = new_cuesep[0:(max_seq_len - 2)]
                    subword_mask = subword_mask[0:(max_seq_len - 2)]
                    token_type_ids = token_type_ids[0:(max_seq_len - 2)]

                input_ids = self.tokenizer.convert_tokens_to_ids(new_text)
                padding_mask = [1] * len(input_ids)
                input_len = len(input_ids)
                feature = CueFeature(guid=guid, or_sent=sent, sent=new_text, input_ids=input_ids,
                                     padding_mask=padding_mask, subword_mask=subword_mask, token_type_ids=token_type_ids,
                                     input_len=input_len, cues=new_cues, cue_sep=new_cuesep, num_cues=num_cues)
            else:
                # For non-BERT (non-BPE tokenization)
                cues = example.cues
                if param.ignore_multiword_cue:
                    for i, c in enumerate(cues):
                        if c == 2:
                            cues[i] = 1
                cues.insert(0, 2)
                cues.append(2)
                cue_sep = example.cue_sep
                cue_sep.insert(0, 0)
                cue_sep.append(0)
                words = self.tokenizer.tokenize(sent)
                words.insert(0, '[CLS]')
                words.append('[SEP]')
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                padding_mask = [1] * len(input_ids)
                input_len = len(input_ids)
                feature = CueFeature(guid=guid, or_sent=sent, sent=words, input_ids=input_ids,
                                     padding_mask=padding_mask, subword_mask=None, token_type_ids=None,
                                     input_len=input_len, cues=cues, cue_sep=cue_sep, num_cues=num_cues)
            features.append(feature)

        return features


    def create_features(self, data: List[ExampleLike], cue_or_scope: str,
                        max_seq_len: int = 128, is_bert=False) -> List[Union[CueFeature, ScopeFeature]]:
        """
        Create packed 
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for example in data:
            f = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)
            if param.ignore_multi_negation:
                # For joint prediction, forced ignore multiple negation sentences
                if f[0].num_cues > 1:
                    continue
            if param.task == 'scope':
                if f[0].num_cues == 0:
                    continue
            features.append(f[0])
        with open('seg\\train_seg_s.bin', 'rb') as file:
            t = _pickle.load(file)
        return features

    def create_features_aug(self, data: List[FeatureLike]) -> List[AugFeature]:
        features = []
        for feat in data:
            f = AugFeature(feat)
            for i in range(f.num_cues):
                new_text, new_subword = f.augment(f.sents[i], f.scopes[i], f.subword_mask[i])
                f.aug_sent.append(new_text)
                f.target_subword_mask.append(new_subword)
                f.target_ids.append(self.tokenizer.convert_tokens_to_ids(new_text))
                f.target_padding_mask.append([1] * len(f.input_ids[0]))
            features.append(f)
        return features

    def create_features_edu(self, data: List[FeatureLike]) -> List[AugFeature]:
        features = []
        for feat in data:
            f = EDUFeature(feat)
            features.append(f)
        return features
    
    def create_features_multi(self, data: List[ExampleLike], cue_or_scope: str,
                              max_seq_len: int = 128, is_bert=False) -> List[Union[CueFeature, ScopeFeature]]:
        """
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for example in data:
            num_cues = example.num_cues
            if num_cues == 1:
                tmp = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)[0]
                mat = self.single_scope_to_matrix(tmp.scopes[0], tmp.cues[0], len(tmp.scopes[0]))
                feature = MultiScopeFeature(tmp.guid, tmp.or_sent, tmp.sents, tmp.input_ids, tmp.padding_mask,
                                            tmp.subword_mask, tmp.input_len, tmp.cues, tmp.scopes, num_cues, mat)
                features.append(feature)
            elif num_cues == 0:
                tmp = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)[0]
                mat = self.single_scope_to_matrix(tmp.scopes[0], tmp.cues[0], len(tmp.scopes[0]))
                feature = MultiScopeFeature(tmp.guid, tmp.or_sent, tmp.sents, tmp.input_ids, tmp.padding_mask,
                                            tmp.subword_mask, tmp.input_len, tmp.cues, tmp.scopes, num_cues, mat)
                features.append(feature)
            else:
                tmp = self.create_feature_from_example(example, cue_or_scope, max_seq_len, is_bert)[0]
                input_lens = [len(e) for e in tmp.cues]
                combined_matrix = self.multi_scopes_to_matrix(tmp.scopes, tmp.cues, input_lens, num_cues)
                feature = MultiScopeFeature(tmp.guid, tmp.or_sent, tmp.sents, tmp.input_ids, tmp.padding_mask,
                                            tmp.subword_mask, input_lens, tmp.cues, tmp.scopes, num_cues, combined_matrix)
                features.append(feature)
        return features
        


    def create_features_pipeline(self, cue_input: List[CueFeature], scope_input: List[ScopeFeature], cue_model, 
                                 max_seq_len: int, is_bert=False, non_cue_examples=None):
        """
        Create scope feature for pipeline TESTING. The cue info was predicted with a trained cue 
        model, instead of the golden cues. Training of scope model will be normal scope model, with golden cue input.
        calling cue_model returns the logits of cue label and the separation.
        It shouldn't be batched input as the output batch size is not controlable, due to different number of cues
        """
        assert self.tokenizer is not None, 'Execute self.get_tokenizer() first to get the corresponding tokenizer.'
        features = []
        for counter, cue_ex in tqdm(enumerate(cue_input), desc='Processing input', total=len(cue_input)):
            wrap_sents = []
            wrap_input_id = []
            wrap_subword_mask = []
            wrap_cues = []
            wrap_padding_mask = []
            wrap_input_len = []
            sent = cue_ex.sent
            tmp_mask = []
            gold_nc = np.max(cue_ex.cue_sep)
            if is_bert:
                tmp_text = []
                tmp_cue = []
                tmp_sep = []
                sent_list = sent.split(' ')
                for word, cue, sep in zip(sent_list, cue_ex.cues, cue_ex.cue_sep):
                    subwords = self.tokenizer.tokenize(word)
                    for i, subword in enumerate(subwords):
                        mask = 1
                        if i > 0:
                            mask = 0
                        tmp_mask.append(mask)
                        tmp_cue.append(cue)
                        tmp_text.append(subword)
                        tmp_sep.append(sep)
                if len(tmp_text) >= max_seq_len - 1:
                    tmp_text = tmp_text[0:(max_seq_len - 2)]
                    tmp_mask = tmp_mask[0:(max_seq_len - 2)]
                    tmp_cue = tmp_cue[0:(max_seq_len - 2)]
                    tmp_sep = tmp_sep[0:(max_seq_len - 2)]
                tmp_text.insert(0, self.tokenizer.cls_token)
                tmp_text.append(self.tokenizer.sep_token)
                tmp_mask.insert(0, 1)
                tmp_mask.append(1)
                tmp_cue.insert(0, 3)
                tmp_cue.append(3)
                tmp_sep.insert(0, 0)
                tmp_sep.append(0)
                tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_text)

                tmp_pad_mask = [1] * len(tmp_input_ids)
                tmp_input_lens = len(tmp_input_ids)
                tmp_pad_mask_in = tmp_pad_mask.copy()
                while len(tmp_input_ids) < max_seq_len:
                    tmp_input_ids.append(0)
                    tmp_pad_mask_in.append(0)
                    tmp_mask.append(0)
                    tmp_cue.append(0)
                    tmp_sep.append(0)
                tmp_input_ids = torch.LongTensor(tmp_input_ids).unsqueeze(0).cuda()
                tmp_pad_mask_in = torch.LongTensor(tmp_pad_mask_in).unsqueeze(0).cuda()
                cue_logits, cue_sep_logits = cue_model(
                    tmp_input_ids, attention_mask=tmp_pad_mask_in)
                pred_cues = torch.argmax(cue_logits, dim=-1).squeeze()
                pred_cue_sep = torch.argmax(cue_sep_logits, dim=-1).squeeze()
                # e.g.
                # pred_cue:     [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3]
                # pred_cue_sep: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
                nu = pred_cue_sep.clone()
                pred_num_cues = nu.max().item()
                sep_cues = [[3] * tmp_input_lens for c in range(pred_num_cues)]
                for c in range(pred_num_cues):
                    for i, e in enumerate(tmp_pad_mask):
                        if e == 1:
                            if pred_cue_sep[i].item() == c+1:
                                sep_cues[c][i] = pred_cues[i].item()
                
                gold_sep_cues = [[3] * tmp_input_lens for c in range(gold_nc)]
                for gc in range(gold_nc):
                    for i, e in enumerate(tmp_pad_mask):
                        if e == 1:
                            if tmp_sep[i] == gc+1:
                                gold_sep_cues[gc][i] = tmp_cue[i]
                nc = max(pred_num_cues, gold_nc)
                cue_match = [-1 for c in range(nc)]
                for pc in range(pred_num_cues):
                    pred_cue_pos = [index for index, v in enumerate(sep_cues[pc]) if v == 1 or v == 2 or v == 4]
                    for gc in range(gold_nc):
                        gold_cue_pos = [index for index, v in enumerate(gold_sep_cues[gc]) if v == 1 or v == 2 or v == 4]
                        match = bool(set(pred_cue_pos) & set(gold_cue_pos))
                        if match:
                            cue_match[pc] = gc


                for c in range(pred_num_cues):
                    new_text = []
                    new_cues = []
                    new_masks = []
                    pos = 0
                    prev = 3
                    for token, cue, mask in zip(tmp_text, sep_cues[c], tmp_mask):
                        # Process the cue augmentation.
                        # Different from the original repo, the strategy is indicate the cue border
                        if cue != 3:
                            if pos != 0 and pos != len(tmp_text)-1:
                                if cue == prev:
                                    # continued cue, don't care is subword or multi word cue
                                    new_masks.append(mask)
                                    new_text.append(token)
                                    new_cues.append(cue)
                                else:
                                    # left bound
                                    new_text.append(f'[unused{cue}]')
                                    new_cues.append(cue)
                                    new_masks.append(mask)
                                    new_text.append(token)
                                    new_masks.append(0)
                                    new_cues.append(cue)
                            elif pos == 0:
                                # at pos 0
                                new_text.append(f'[unused{cue}]')
                                new_masks.append(mask)
                                new_cues.append(cue)
                                new_text.append(token)
                                new_masks.append(0)
                                new_cues.append(cue)
                            else:
                                # at eos
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                                new_text.append(f'[unused{cue}]')
                                new_masks.append(0)
                                new_cues.append(cue)
                        else:
                            if cue == prev:
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                            else:
                                # current non cue, insert right bound before current pos
                                new_text.append(f'[unused{[prev]}]')
                                new_masks.append(0)
                                new_cues.append(prev)
                                new_text.append(token)
                                new_masks.append(mask)
                                new_cues.append(cue)
                        prev = cue
                        pos += 1
                    input_id = self.tokenizer.convert_tokens_to_ids(new_text)
                    padding_mask = [1] * len(input_id)
                    input_len = len(input_id)
                    while len(input_id) < max_seq_len:
                        input_id.append(0)
                        padding_mask.append(0)
                        new_masks.append(0)
                        new_cues.append(0)
                    assert len(input_id) == max_seq_len
                    assert len(padding_mask) == max_seq_len
                    assert len(new_masks) == max_seq_len
                    assert len(new_cues) == max_seq_len
                    wrap_sents.append(new_text)
                    wrap_input_id.append(input_id)
                    wrap_subword_mask.append(new_masks)
                    wrap_cues.append(new_cues)
                    wrap_padding_mask.append(padding_mask)
                    wrap_input_len.append(input_len)

            if cue_ex.guid.startswith('nc'):
                gold_num_cue = gold_nc
                gold_scopes = [[2 for i in sent_list]]
            else:
                temp_scopes = scope_input[counter].scopes.copy()
                for c in range(scope_input[counter].num_cues):
                    temp_scopes[c].insert(0, 2)
                    temp_scopes[c].append(2)
                gold_scopes = temp_scopes
                gold_num_cue = gold_nc

            feature = PipelineScopeFeature(guid=cue_ex.guid, or_sent=cue_ex.sent, sents=wrap_sents, input_ids=wrap_input_id,
                                           padding_mask=wrap_padding_mask, subword_mask=wrap_subword_mask,
                                           input_len=wrap_input_len, cues=wrap_cues, cue_match=cue_match, gold_scopes=gold_scopes, gold_num_cues=gold_num_cue)
            features.append(feature)
        return features

    def create_dataset(self, features: List[FeatureLike], cue_or_scope: str, example_type: str,
                       is_sorted=False, is_bert=False) -> Union[Dataset, TensorDataset]:
        """
        Pack the features to dataset. If cue_or_scope is cue or scope, 
        return a TensorDataset for faster processing.
        If cue_or_scope is pipeline and example_type is dev or test, return a Dataset,
        in which the features still keep to the packed scope format.

        params:
            features (list): collection of features
            cue_or_scope (str): in ['cue', 'scope', 'pipeline']
            example_type (str): in ['train', 'dev', 'test']
            is_sorted (bool): to specify whether to sort the dataset or not. Save time and space when dumping.
            is_bert (bool)

        return:
            cue_or_scope == cue:
                TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_mask)
            cue_or_scope == scope:
        """

        if is_sorted:
            print('sorted data by th length of input')
            features = sorted(
                features, key=lambda x: x.input_len, reverse=True)
        if cue_or_scope.lower() == 'cue':
            if is_bert:
                input_ids = []
                padding_mask = []
                subword_mask = []
                cues = []
                cue_sep = []
                input_len = []
                token_type = []
                for feature in features:
                    input_ids.append(feature.input_ids)
                    padding_mask.append(feature.padding_mask)
                    subword_mask.append(feature.subword_mask)
                    cues.append(feature.cues)
                    cue_sep.append(feature.cue_sep)
                    input_len.append(feature.input_len)
                    token_type.append(feature.token_type_ids)

                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=self.tokenizer.pad_token_id, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
                cue_sep = pad_sequences(cue_sep,
                                        maxlen=param.max_len, value=0, padding="post",
                                        dtype="long", truncating="post").tolist()
                subword_mask = pad_sequences(subword_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                token_type = pad_sequences(token_type,
                                             maxlen=param.max_len, value=1, padding="post",
                                             dtype="long", truncating="post").tolist()                          

                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                cues = torch.LongTensor(cues)
                cue_sep = torch.LongTensor(cue_sep)
                input_len = torch.LongTensor(input_len)
                subword_mask = torch.LongTensor(subword_mask)
                token_type = torch.LongTensor(token_type)
                cue_matrix = self.cue_to_matrix(cues, input_len)
                if param.cue_matrix:
                    return TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_mask, cue_matrix)
                else:
                    return TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_mask, token_type)
            else:
                # None BERT
                input_ids = []
                padding_mask = []
                cues = []
                cue_sep = []
                input_len = []
                for feature in features:
                    input_ids.append(feature.input_ids)
                    padding_mask.append(feature.padding_mask)
                    cues.append(feature.cues)
                    cue_sep.append(feature.cue_sep)
                    input_len.append(feature.input_len)

                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=0, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
                cue_sep = pad_sequences(cue_sep,
                                        maxlen=param.max_len, value=0, padding="post",
                                        dtype="long", truncating="post").tolist()

                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                cues = torch.LongTensor(cues)
                cue_sep = torch.LongTensor(cue_sep)
                input_len = torch.LongTensor(input_len)
                subword_masks = torch.zeros_like(input_ids)
                cue_matrix = self.cue_to_matrix(cues, input_len)
                if param.cue_matrix:
                    return TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_masks, cue_matrix)
                else:
                    return TensorDataset(input_ids, padding_mask, cues, cue_sep, input_len, subword_masks)
        elif cue_or_scope.lower() == 'scope':
            if is_bert:
                input_ids = []
                padding_mask = []
                subword_mask = []
                scopes = []
                input_len = []
                cues = []
                for feature in features:
                    num_cue = feature.num_cues
                    if param.task == 'joint':
                        if num_cue == 0:
                            num_cue = 1
                    for cue_i in range(num_cue):
                        input_ids.append(feature.input_ids[cue_i])
                        padding_mask.append(feature.padding_mask[cue_i])
                        scopes.append(feature.scopes[cue_i])
                        subword_mask.append(feature.subword_mask[cue_i])
                        input_len.append(feature.input_len[cue_i])
                        cues.append(feature.cues[cue_i])

                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=self.tokenizer.pad_token_id, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                
                scopes = pad_sequences(scopes,
                                    maxlen=param.max_len, value=0, padding="post",
                                    dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
                subword_mask = pad_sequences(subword_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()

                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                input_len = torch.LongTensor(input_len)
                scopes = torch.LongTensor(scopes)
                cues = torch.LongTensor(cues)
                subword_masks = torch.LongTensor(subword_mask)
                scopes_matrix = self.scope_to_matrix(scopes, cues, input_len)
                if param.matrix:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks, scopes_matrix)
                else:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks)
            else:
                input_ids = []
                padding_mask = []
                scopes = []
                input_len = []
                cues = []
                for feature in features:
                    for cue_i in range(feature.num_cues):
                        input_ids.append(feature.input_ids[cue_i])
                        padding_mask.append(feature.padding_mask[cue_i])
                        scopes.append(feature.scopes[cue_i])
                        input_len.append(feature.input_len[cue_i])
                        cues.append(feature.cues[cue_i])
                        assert len(feature.scopes[cue_i]) == len(feature.cues[cue_i])
                input_ids = pad_sequences(input_ids,
                                          maxlen=param.max_len, value=0, padding="post",
                                          dtype="long", truncating="post").tolist()
                padding_mask = pad_sequences(padding_mask,
                                             maxlen=param.max_len, value=0, padding="post",
                                             dtype="long", truncating="post").tolist()
                scopes = pad_sequences(scopes,
                                       maxlen=param.max_len, value=0, padding="post",
                                       dtype="long", truncating="post").tolist()
                cues = pad_sequences(cues,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
                input_ids = torch.LongTensor(input_ids)
                padding_mask = torch.LongTensor(padding_mask)
                scopes = torch.LongTensor(scopes)
                input_len = torch.LongTensor(input_len)
                cues = torch.LongTensor(cues)
                subword_masks = torch.zeros_like(input_ids)
                scopes_matrix = self.scope_to_matrix(scopes, cues, input_len)
                if param.matrix:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks, scopes_matrix)
                else:
                    return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, subword_masks)
        elif cue_or_scope.lower() == 'pipeline' and example_type.lower() == 'test':
            return Dataset(features)
        elif cue_or_scope.lower() == 'multi': 
            # feature: MultiScopeFeature
            master_mat = []
            padding_masks = []
            subword_masks = []
            input_len = []
            input_ids = []
            scopes = []
            cues = []
            for feature in features:
                num_cue = feature.num_cues
                if num_cue == 0:
                    num_cue = 1
                input_ids.append(feature.input_ids[0])
                padding_masks.append(feature.padding_mask[0])
                subword_masks.append(feature.subword_mask[0])
                input_len.append(feature.input_len[0])

                scopes_padded = [[0] for i in range(param.max_num_cues)]
                cues_padded = [[0] for i in range(param.max_num_cues)]
                for i in range(num_cue):
                    scopes_padded[i] = feature.scopes[i]
                    cues_padded[i] = feature.cues[i]
                scopes_padded = pad_sequences(scopes_padded,
                                              maxlen=param.max_len, value=0, padding="post",
                                              dtype="long", truncating="post").tolist()
                cues_padded = pad_sequences(cues_padded,
                                            maxlen=param.max_len, value=0, padding="post",
                                            dtype="long", truncating="post").tolist()
                scopes.append(scopes_padded)
                cues.append(cues_padded)

                master_mat.append(feature.master_mat)

            padding_masks = pad_sequences(padding_masks,
                                         maxlen=param.max_len, value=0, padding="post",
                                         dtype="long", truncating="post").tolist()
            subword_masks = pad_sequences(subword_masks,
                                         maxlen=param.max_len, value=0, padding="post",
                                         dtype="long", truncating="post").tolist()
            input_ids = pad_sequences(input_ids,
                                      maxlen=param.max_len, value=0, padding="post",
                                      dtype="long", truncating="post").tolist()

            input_ids = torch.LongTensor(input_ids)
            scopes = torch.LongTensor(scopes)
            cues = torch.LongTensor(cues)
            input_len = torch.LongTensor(input_len)
            padding_masks = torch.LongTensor(padding_masks)
            subword_masks = torch.LongTensor(subword_masks)
            master_mat = torch.stack(master_mat, 0)
            return TensorDataset(input_ids, padding_masks, scopes, input_len, cues, subword_masks, master_mat)
        else:
            raise ValueError(cue_or_scope, example_type)
    
    def create_pipeline_ds(self, feats: List[PipelineScopeFeature]):
        weak = []
        strict = []

    def create_aug_ds(self, features: List[AugFeature]) -> TensorDataset:
        pass
        input_ids = []
        padding_mask = []
        subword_mask = []
        scopes = []
        input_len = []
        cues = []
        target_ids = []
        target_padding_mask = []
        target_subword_mask = []
        for feature in features:
            num_cue = feature.num_cues
            if param.task == 'joint':
                if num_cue == 0:
                    num_cue = 1
            for cue_i in range(num_cue):
                input_ids.append(feature.input_ids[cue_i])
                padding_mask.append(feature.padding_mask[cue_i])
                scopes.append(feature.scopes[cue_i])
                subword_mask.append(feature.subword_mask[cue_i])
                input_len.append(feature.input_len[cue_i])
                cues.append(feature.cues[cue_i])
                target_ids.append(feature.target_ids[cue_i])
                target_padding_mask.append(feature.target_padding_mask[cue_i])
                target_subword_mask.append(feature.target_subword_mask[cue_i])

        input_ids = pad_sequences(input_ids,
                                  maxlen=param.max_len, value=self.tokenizer.pad_token_id, padding="post",
                                  dtype="long", truncating="post").tolist()
        target_ids = pad_sequences(target_ids,
                                   maxlen=param.max_len, value=self.tokenizer.pad_token_id, padding="post",
                                   dtype="long", truncating="post").tolist()
        padding_mask = pad_sequences(padding_mask,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()

        scopes = pad_sequences(scopes,
                               maxlen=param.max_len, value=0, padding="post",
                               dtype="long", truncating="post").tolist()
        cues = pad_sequences(cues,
                             maxlen=param.max_len, value=0, padding="post",
                             dtype="long", truncating="post").tolist()
        subword_mask = pad_sequences(subword_mask,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()

        target_padding_mask = pad_sequences(target_padding_mask,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()
        target_subword_mask = pad_sequences(target_subword_mask,
                                     maxlen=param.max_len, value=0, padding="post",
                                     dtype="long", truncating="post").tolist()

        input_ids = torch.LongTensor(input_ids)
        padding_mask = torch.LongTensor(padding_mask)
        input_len = torch.LongTensor(input_len)
        scopes = torch.LongTensor(scopes)
        cues = torch.LongTensor(cues)
        target_ids = torch.LongTensor(target_ids)
        target_padding_mask = torch.LongTensor(target_padding_mask)
        target_subword_mask = torch.LongTensor(target_subword_mask)
        return TensorDataset(input_ids, padding_mask, scopes, input_len, cues, target_ids, target_padding_mask, target_subword_mask)

    def generate_negative_scopes(self, data: TensorDataset):
        """
        tokenizer.additional_special_tokens_ids[-2] [-1]
        """
        new_input_ids = []
        y = []
        for row in data:
            scope = row[2]
            input_id = row[0].tolist()
            boundary = get_boundary(scope.unsqueeze(0))[0].tolist()
            start_pos = [i for i, x in enumerate(boundary) if x in [1, 3]]
            end_pos = [i for i, x in enumerate(boundary) if x in [2, 3]]
            cue_pos = [i for i, x in enumerate(scope) if x == 3]
            start_offsets = []
            end_offsets = []
            for sp in start_pos:
                temp = [0]
                # shift start to left
                if sp - 2 not in start_pos and sp - 2 not in end_pos:
                    temp.append(-2)
                if sp - 1 not in start_pos and sp - 1 not in end_pos:
                    temp.append(-1)
                if sp + 1 not in start_pos and sp + 1 not in end_pos:
                    temp.append(1)
                if sp + 2 not in start_pos and sp + 2 not in end_pos:
                    temp.append(2)
                start_offsets.append(temp)
            for ep in end_pos:
                temp = [0]
                # shift start to left
                if ep - 2 not in end_pos and ep - 2 not in end_pos:
                    temp.append(-2)
                if ep - 1 not in end_pos and ep - 1 not in end_pos:
                    temp.append(-1)
                if ep + 1 not in end_pos and ep + 1 not in end_pos:
                    temp.append(1)
                if ep + 2 not in end_pos and ep + 2 not in end_pos:
                    temp.append(2)
                end_offsets.append(temp)
            print()

        return new_input_ids, y
            


    def ex_to_bio(self, data: List[ScopeFeature]):
        """
        Label: 1: scope, 2: O, 3: cue
        BIO: 1: B, 2:I, 3:O, 4:cue
        """
        new_features = data
        for f_count, feat in enumerate(new_features):
            for c_count in range(feat.num_cues):
                cues = feat.cues[c_count]
                scopes = feat.scopes[c_count]
                temp_scope = []
                for i, e in enumerate(scopes):
                    if e == 2:
                        if cues[i] != 3:
                            temp_scope.append(6)
                        else:
                            temp_scope.append(e)
                    else:
                        temp_scope.append(e)
                if len(scopes) != 1:
                    scope_bioes = scope_to_bio(temp_scope)
                else:
                    scope_bioes = scopes
                new_features[f_count].scopes[c_count] = scope_bioes
        return new_features

    def cue_to_matrix(self, cues: Tensor, input_lens: Tensor):
        dataset_size = cues.size(0)
        all_cue_matrix = []
        for i in range(dataset_size):
            cue = cues[i].tolist()
            input_len = input_lens[i].tolist()
            cue_matrix = single_cue_to_matrix_pad(cue, input_len)
            all_cue_matrix.append(cue_matrix)
        all_cue_matrix = torch.stack(all_cue_matrix, 0)
        return all_cue_matrix
    

    def scope_to_matrix(self, scopes: Tensor, cues: Tensor, input_lens: Tensor):
        dataset_size = scopes.size(0)
        all_scope_matrix = []
        for i in range(dataset_size):
            scope = scopes[i].tolist()
            cue = cues[i].tolist()
            input_len = input_lens[i].tolist()
            scope_matrix = self.single_scope_to_matrix(scope, cue, input_len)
            all_scope_matrix.append(scope_matrix)
        all_scope_matrix = torch.stack(all_scope_matrix, 0)
        return all_scope_matrix
    
    def single_scope_to_matrix(self, scope: List, cue: List, input_len: int):
        temp_scope = []
        for j, e in enumerate(scope):
            if e == 2:
                if cue[j] != 3:
                    temp_scope.append(3)
                else:
                    temp_scope.append(e)
            else:
                temp_scope.append(e)
        if len(scope) != 1:
            assert len(scope) == len(cue)
            scope_matrix = single_scope_to_link_matrix_pad(temp_scope, cue, input_len)
        else:
            scope_matrix = torch.LongTensor(scope)
        return scope_matrix
    
    def multi_scopes_to_matrix(self, scopes: List, cues: List, input_lens: List, num_cues: int) -> np.ndarray:
        marked_scopes = []
        if num_cues == 0:
            num_cues = 1
        for i in range(num_cues):
            temp_scope = []
            for j, e in enumerate(scopes[i]):
                if cues[i][j] != 3:
                    temp_scope.append(3)
                else:
                    temp_scope.append(e)
            marked_scopes.append(temp_scope)
        mat_m = multi_scope_to_link_matrix_pad(marked_scopes, cues, input_lens)
        return mat_m

    def scope_add_cue(self, data: List[ScopeFeature]):
        new_features = data.copy()
        for f_count, feat in enumerate(new_features):
            for c_count in range(feat.num_cues):
                cues = feat.cues[c_count]
                scopes = feat.scopes[c_count]
                temp_scope = []
                for i, e in enumerate(scopes):
                    if cues[i] != 3:
                        temp_scope.append(1)
                    else:
                        temp_scope.append(e)
                new_features[f_count].scopes[c_count] = temp_scope
        return new_features
                

    def get_tokenizer(self, data: Tuple[InputExample], is_bert=False, do_lower_case=False, bert_path=None, non_cue_sents: List[str] = None, noembed=False):
        if is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=param.bert_path, do_lower_case=do_lower_case, cache_dir=param.bert_cache, add_prefix_space=True)
        else:
            if noembed:
                self.tokenizer = NaiveTokenizer(data)
            else:
                self.tokenizer = OtherTokenizer(
                    data, external_vocab=False, non_cue_sents=non_cue_sents)


class Dictionary(object):
    def __init__(self):
        self.token2id = {}
        self.id2token = []

    def add_word(self, word):
        if word not in self.token2id:
            self.id2token.append(word)
            self.token2id[word] = len(self.id2token) - 1
        return self.token2id[word]

    def __len__(self):
        return len(self.id2token)


class NaiveTokenizer(object):
    def __init__(self, data: Tuple[InputExample], non_cue_sents: List[str] = None):
        self.dictionary = Dictionary()
        self.data = data
        self.dictionary.add_word('<PAD>')
        for s in self.data:
            if isinstance(s.sent, str):
                split_sent = s.sent.split(' ')
                for word in split_sent:
                    self.dictionary.add_word(word)
            else:
                for word in s.sent:
                    self.dictionary.add_word(word)
        if non_cue_sents is not None:
            for sent in non_cue_sents:
                for word in sent:
                    self.dictionary.add_word(word)
        self.dictionary.add_word('<OOV>')
        self.dictionary.add_word('[CLS]')
        self.dictionary.add_word('[SEP]')

    def tokenize(self, text: Union[str, List]):
        if isinstance(text, list):
            return text
        elif isinstance(text, str):
            words = text.split()
            return words

    def convert_tokens_to_ids(self, tokens: Iterable):
        if isinstance(tokens, list):
            ids = []
            for token in tokens:
                try:
                    ids.append(self.dictionary.token2id[token])
                except KeyError:
                    ids.append(self.dictionary.token2id['<OOV>'])
            return ids
        elif isinstance(tokens, str):
            try:
                return self.dictionary.token2id[tokens]
            except KeyError:
                return self.dictionary.token2id['<OOV>']

    def decode(self, ids):
        token_list = [self.dictionary.id2token[tid] for tid in ids]
        return " ".join(token_list)


class OtherTokenizer(NaiveTokenizer):
    def __init__(self, data, emb=param.embedding, external_vocab=False, non_cue_sents=None):
        super(OtherTokenizer, self).__init__(data, non_cue_sents)
        if external_vocab is True:
            with open('reduced_fasttext_vocab.bin', 'rb') as f:
                vocab = _pickle.load(f)
            for i, (k, v) in enumerate(vocab.items()):
                self.dictionary.add_word(k)
        self.vector = gensim.models.KeyedVectors.load_word2vec_format(
            param.emb_cache, binary=True)
        self.embedding = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06),
                                           (len(self.dictionary.token2id), param.word_emb_dim))
        for w in self.dictionary.token2id:
            if w in self.vector:
                self.embedding[self.dictionary.token2id[w]] = self.vector[w]
            elif w.lower() in self.vector:
                self.embedding[self.dictionary.token2id[w]
                               ] = self.vector[w.lower()]
        del self.vector
        gc.collect()


if __name__ == "__main__":
    proc = Processor()
    sfu_data = proc.read_data(param.data_path['sfu'], 'sfu')
    proc.create_examples(sfu_data, 'joint', 'sfu', 'cue', 'sfu.pt')
    bio_a_data = proc.read_data(
        param.data_path['bioscope_abstracts'], 'bioscope')
    proc.create_examples(bio_a_data, 'joint', 'bioscope_a', 'cue', 'bioA.pt')
    bio_f_data = proc.read_data(param.data_path['bioscope_full'], 'bioscope')
    proc.create_examples(bio_f_data, 'joint', 'bioscope_f', 'cue', 'bioF.pt')
