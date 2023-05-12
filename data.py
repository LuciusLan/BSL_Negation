import os
import re
import html
import random
import glob
from typing import List, Tuple, T, Iterable, Union, NewType
from params import param
from brat_parser import get_entities_relations_attributes_groups
import nltk

#cue_id2label = {4: 'Affix', 1: 'Cue', 2: 'MultiWordCue', 3: 'O'}
#cue_label2id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
#cue_id2label = {1: 'Cue', 2: 'O'}
#cue_label2id = {0: 0, 1: 1, 2: 2}
cue_id2label = {0: '<PAD>', 1: 'B', 2: 'I', 3:'O'}
cue_label2id = {0: 0, 1: 1, 2: 2, 3:3}
scope_id2label = {0: '<PAD>', 1: 'I', 2:'O', 3: 'C', 4: 'B', 5: 'E', 6: 'S'}

class RawData():
    """
    Wrapper of data. For sfu and bioscope data, splitting of train-dev-test will also be done here.
    class var:
        cues (tuple[T]): cues data.
            (sentences, cue_labels, cue_sep, num_cues)
        scopes (tuple[T]): scopes data.
            (original_sent, sentences[n], cue_labels[n], scope_labels[n]). n=num_cues
        non_cue_sents (tuple[T]): sentences that does not contain negation.
    For detail refer to the local methods for each dataset in constructor
    """
    def __init__(self, file, dataset_name='sfu'):
        """
        params:
            file: The path of the data file.
            dataset_name: The name of the dataset to be preprocessed. Values supported: sfu, bioscope, sherlock.
            frac_no_cue_sents: The fraction of sentences to be included in the data object which have no negation/speculation cues.
        """
        if dataset_name == 'bioscope':
            self.cues, self.scopes, self.non_cue_sents = bioscope(file)
        elif dataset_name == 'sfu':
            sfu_cues = [[], [], [], []]
            sfu_scopes = [[], [], [], []]
            sfu_noncue_sents = []
            for dir_name in os.listdir(file):
                if '.' not in dir_name:
                    for f_name in os.listdir(os.path.join(file, dir_name)):
                        r_val = sfu_review(
                            os.path.join(file, dir_name, f_name))
                        sfu_cues = [a+b for a, b in zip(sfu_cues, r_val[0])]
                        sfu_scopes = [a+b for a,
                                      b in zip(sfu_scopes, r_val[1])]
                        sfu_noncue_sents.extend(r_val[2])
            self.cues = sfu_cues
            self.scopes = sfu_scopes
            self.non_cue_sents = sfu_noncue_sents
        elif dataset_name == 'sherlock':
            self.cues, self.scopes, self.non_cue_sents = sherlock(file)
            print()
        elif dataset_name == 'wiki':
            self.wiki = wiki_weasel(file)
        elif dataset_name == 'vet':
            list_file = glob.glob(file+'/*.ann')
            vetcompass_cues = [[], [], [], []]
            vetcompass_scopes = [[], [], [], []]
            for f in list_file:
                f_ann = f
                f_text = f.replace('negspec','sent')
                r_val = vetcompass(f_text,f_ann,frac_no_cue_sents=0)
                vetcompass_cues = [a+b for a,b in zip(vetcompass_cues, r_val[0])]
                vetcompass_scopes = [a+b for a,b in zip(vetcompass_scopes, r_val[1])]
            self.cues = vetcompass_cues
            self.scopes = vetcompass_scopes
        else:
            raise ValueError(
                "Supported Dataset types are:\n\tbioscope\n\tsfu\n\tsherlock\n\tvet")

def sherlock(f_path) -> Tuple[List, List, List]:
    """
    return: raw data format. (cues, scopes, non_cue_sents)
        cues (list[list[T]]): (sentences, cue_labels, cue_sep, num_cues)
            cue_sep is the seperation label of cues. 
            num_cues notes the number of cues in this sent.
        scopes (list[list[T]]): (original_sent, sentences[n], cue_labels[n], scope_labels[n])
            where n = number of cues in this sentence.
            Note that for senteces[n], the length is different for sent corresponding to
            an affix cue.
        non_cue_sents (list[T]): sentences that does not contain negation.
    """
    raw_data = open(f_path)
    sentence = []
    labels = []
    label = []
    scope_sents = []
    or_sents = []
    data_scope = []
    scope = []
    scope_cues = []
    data = []
    non_cue_data = []
    num_cue_list = []
    cue_sep = []

    for line in raw_data:
        label = []
        sentence = []
        tokens = line.strip().split()
        afsent = []
        noncue_sep = []
        if len(tokens) == 8:  # This line has no cues
            num_cues = 0
            sentence.append(tokens[3])
            label.append(3)  # Not a cue
            noncue_sep.append(0)
            for line in raw_data:
                tokens = line.strip().split()
                if len(tokens) == 0:
                    break
                else:
                    sentence.append(tokens[3])
                    label.append(3)
                    noncue_sep.append(0)
            non_cue_data.append([sentence, label, noncue_sep])

        else:  # The line has 1 or more cues
            num_cues = (len(tokens) - 7) // 3
            affix_num = -1
            # cue_count+=num_cues
            scope = [[] for i in range(num_cues)]
            # First list is the real labels, second list is to modify
            # if it is a multi-word cue.
            label = [[], []]
            # Generally not a cue, if it is will be set ahead.
            label[0].append(3)
            label[1].append(-1)  # Since not a cue, for now.
            aflabel = [[],[]]
            aflabel[0].append(3)
            aflabel[1].append(-1)
            cue_counter = -1
            prev_cue_c = -1
            for i in range(num_cues):
                if tokens[7 + 3 * i] != '_':  # Cue field is active
                    if tokens[8 + 3 * i] != '_':  # Check for affix
                        label[0][-1] = 4  # Affix
                        #affix_list.append(tokens[7 + 3 * i]) # pylint: disable=undefined-variable
                        if i != prev_cue_c:
                            cue_counter += 1
                        prev_cue_c = i
                        label[1][-1] = i  # Cue number
                        aflabel[0][-1] = 0
                        aflabel[1][-1] = i
                        # sentence.append(tokens[7+3*i])
                        # new_word = '##'+tokens[8+3*i]
                    else:
                        # Maybe a normal or multiword cue. The next few
                        # words will determine which.
                        label[0][-1] = 1
                        aflabel[0][-1] = 1
                        # Which cue field, for multiword cue altering.
                        if i != prev_cue_c:
                            cue_counter += 1
                        prev_cue_c = i
                        label[1][-1] = i
                        aflabel[1][-1] = i

                if tokens[8 + 3 * i] != '_':
                    scope[i].append(1)
                else:
                    scope[i].append(2)
            sentence.append(tokens[3])
            afsent.append(tokens[3])
            for line in raw_data:
                tokens = line.strip().split()
                if len(tokens) == 0:
                    break
                else:
                    #sentence.append(tokens[3])
                    token = tokens[3]
                    affix_flag = False
                    # Generally not a cue, if it is will be set ahead.
                    label[0].append(3)
                    label[1].append(-1)  # Since not a cue, for now.
                    aflabel[0].append(3)
                    aflabel[1].append(-1)
                    for i in range(num_cues):
                        if tokens[7 + 3 *
                                    i] != '_':  # Cue field is active
                            if tokens[8 + 3 *
                                        i] != '_':  # Check for affix
                                label[0][-1] = 4  # Affix
                                aflabel[0][-1] = 0
                                aflabel[0].append(3)
                                if i != prev_cue_c:
                                    cue_counter += 1
                                prev_cue_c = i
                                label[1][-1] = i  # Cue number
                                aflabel[1][-1] = i
                                aflabel[1].append(-1)
                                affix_flag = True
                                affix_num = i
                                token = [tokens[3], tokens[7 + 3 * i], tokens[8 + 3 * i]]
                            else:
                                # Maybe a normal or multiword cue. The
                                # next few words will determine which.
                                label[0][-1] = 1
                                aflabel[0][-1] = 1
                                # Which cue field, for multiword cue
                                # altering.
                                if i != prev_cue_c:
                                    cue_counter += 1
                                prev_cue_c = i
                                label[1][-1] = i
                                aflabel[1][-1] = i
                                
                        if tokens[8 + 3 * i] != '_':
                            # Detected scope
                            if tokens[7 + 3 * i] != '_' and i == affix_num:
                                # Check if it is affix cue
                                scope[i].append(1)
                                if param.sherlock_seperate_affix:
                                    scope[i].append(1)
                            else:
                                scope[i].append(1)
                        else:
                            scope[i].append(2)
                    if affix_flag is False:
                        sentence.append(token)
                        afsent.append(token)
                    else:
                        sentence.append(token[0])
                        afsent.append(token[1])
                        afsent.append('<AFF>'+token[2])
            for i in range(num_cues):
                indices = []
                for index, j in enumerate(label[1]):
                    if i == j:
                        indices.append(index)
                count = len(indices)
                if count > 1:
                    # Multi word cue
                    for j in indices:
                        label[0][j] = 2
            
            sent_scopes = []
            sent_cues = []
            or_sents.append(sentence)
            scope_sent = []
            for i in range(num_cues):
                sc = []

                if affix_num == -1:
                    # No affix cue in this sent
                    scope_sent.append(sentence)

                    for a, b in zip(label[0], label[1]):
                        if i == b:
                            sc.append(a)
                        else:
                            sc.append(3)
                else:
                    if affix_num == i and param.sherlock_seperate_affix:
                        # Detect affix cue
                        scope_sent.append(afsent)

                        for a, b in zip(aflabel[0], aflabel[1]):
                            if i == b:
                                sc.append(a)
                            else:
                                sc.append(3)
                    else:
                        scope_sent.append(sentence)

                        for a, b in zip(label[0], label[1]):
                            if i == b:
                                sc.append(a)
                            else:
                                sc.append(3)
                sent_scopes.append(scope[i])
                sent_cues.append(sc)
            data_scope.append(sent_scopes)
            scope_cues.append(sent_cues)
            scope_sents.append(scope_sent)
            labels.append(label[0])
            data.append(sentence)
            num_cue_list.append(num_cues)
            cue_sep.append([e+1 for e in label[1]])

    non_cue_sents = [i[0] for i in non_cue_data]
    non_cue_cues = [i[1] for i in non_cue_data]
    non_cue_sep = [i[2] for i in non_cue_data]
    non_cue_num = [0 for i in non_cue_data]
    """if param.mark_cue:
        for bi, block in enumerate(scope_cues):
            for ci, c in enumerate(block):
                for i, e in enumerate(c):
                    if e == 1 or e == 2 or e==3:
                        data_scope[bi][ci][i] = 3"""
    sherlock_cues = (data + non_cue_sents, labels + non_cue_cues, cue_sep+non_cue_sep, num_cue_list+non_cue_num)
    sherlock_scopes = (or_sents, scope_sents, scope_cues, data_scope)
    
    return [sherlock_cues, sherlock_scopes, non_cue_sents]

def bioscope(f_path) -> Tuple[List, List, List]:
    """
    return: raw data format. (cues, scopes, non_cue_sents)
        cues (list[list[T]]): (sentences, cue_labels, cue_sep, num_cues)
            cue_sep is the seperation label of cues. 
            num_cues notes the number of cues in this sent.
        scopes (list[list[T]]): (original_sent, sentences[n], cue_labels[n], scope_labels[n])
            where n = number of cues in this sentence.
        non_cue_sents: sentences that does not contain negation.
    """
    file = open(f_path, encoding='utf-8')
    sentences = []
    for s in file:
        sentences += re.split("(<.*?>)", html.unescape(s))
    cue_sentence = []
    cue_cues = []
    non_cue_data = []
    scope_cues = []
    scope_scopes = []
    scope_sentences = []
    scope_orsents = []
    sentence = []
    cue = {}
    scope = {}
    in_scope = []
    in_cue = []
    word_pos = 0
    c_idx = []
    s_idx = []
    in_sentence = 0
    num_cue_list = []
    cue_sep = []
    for token in sentences:
        if token == '':
            continue
        elif '<sentence' in token:
            in_sentence = 1
        elif '<cue' in token:
            if 'negation' in token:
                in_cue.append(
                    str(re.split('(ref=".*?")', token)[1][4:]))
                c_idx.append(
                    str(re.split('(ref=".*?")', token)[1][4:]))
                if c_idx[-1] not in cue.keys():
                    cue[c_idx[-1]] = []
        elif '</cue' in token:
            in_cue = in_cue[:-1]
        elif '<xcope' in token:
            # print(re.split('(id=".*?")',token)[1][3:])
            in_scope.append(str(re.split('(id=".*?")', token)[1][3:]))
            s_idx.append(str(re.split('(id=".*?")', token)[1][3:]))
            scope[s_idx[-1]] = []
        elif '</xcope' in token:
            in_scope = in_scope[:-1]
        elif '</sentence' in token:
            #print(cue, scope)
            if len(cue.keys()) == 0:
                # no cue in this sent
                non_cue_data.append([sentence, [3]*len(sentence), [0]*len(sentence)])
            else:
                cue_sentence.append(sentence)
                cue_cues.append([3]*len(sentence))
                cue_sep.append([0]*len(sentence))
                scope_sentence = []
                scope_subscope = []
                scope_subcues = []
                for count, i in enumerate(cue.keys()):
                    scope_sentence.append(sentence)
                    scope_subcues.append([3]*len(sentence))
                    if len(cue[i]) == 1:
                        cue_cues[-1][cue[i][0]] = 1
                        scope_subcues[-1][cue[i][0]] = 1
                        cue_sep[-1][cue[i][0]] = count + 1
                    else:
                        for c in cue[i]:
                            cue_cues[-1][c] = 2
                            scope_subcues[-1][c] = 2
                            cue_sep[-1][c] = count + 1
                    scope_subscope.append([2]*len(sentence))

                    if i in scope.keys():
                        for s in scope[i]:
                            scope_subscope[-1][s] = 1
                scope_orsents.append(sentence)
                scope_sentences.append(scope_sentence)
                scope_cues.append(scope_subcues)
                scope_scopes.append(scope_subscope)
            num_cue_list.append(len(cue.keys()))

            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_pos = 0
            in_sentence = 0
            c_idx = []
            s_idx = []
        elif '<' not in token:
            if in_sentence == 1:
                words = token.split()
                sentence += words
                if len(in_cue) != 0:
                    for i in in_cue:
                        cue[i].extend([word_pos + i for i in range(len(words))])
                elif len(in_scope) != 0:
                    for i in in_scope:
                        scope[i] += [word_pos + i for i in range(len(words))]
                word_pos += len(words)
    non_cue_sents = [i[0] for i in non_cue_data]
    non_cue_cues = [i[1] for i in non_cue_data]
    non_cue_sep = [i[2] for i in non_cue_data]
    non_cue_num = [0 for i in non_cue_data]
    """if param.mark_cue:
        for bi, block in enumerate(scope_cues):
            for ci, c in enumerate(block):
                for i, e in enumerate(c):
                    if e == 1 or e== 2:
                        scope_scopes[bi][ci][i] = 3"""
    return [(cue_sentence+non_cue_sents, cue_cues+non_cue_cues, cue_sep+non_cue_sep, num_cue_list+non_cue_num), 
            (scope_orsents, scope_sentences, scope_cues, scope_scopes),
            non_cue_sents]

def sfu_review(f_path) -> Tuple[List, List, List]:
    """
    return: raw data format. (cues, scopes, non_cue_sents)
        cues (list[list[T]]): (sentences, cue_labels)
        scopes (list[list[T]]): (original_sent, sentences[n], cue_labels[n], scope_labels[n])
            where n = number of cues in this sentence.
        non_cue_sents: sentences that does not contain negation.
    """
    file = open(f_path, encoding='utf-8')
    sentences = []
    for s in file:
        sentences += re.split("(<.*?>)", html.unescape(s))
    cue_sentence = []
    cue_cues = []
    scope_cues = []
    scope_scopes = []
    scope_sentence = []
    scope_sentences = []
    scope_orsents = []
    sentence = []
    cue = {}
    scope = {}
    in_scope = []
    in_cue = []
    word_pos = 0
    c_idx = []
    non_cue_data = []
    s_idx = []
    in_word = 0
    num_cue_list = []
    cue_sep = []
    for token in sentences:
        if token == '':
            continue
        elif token == '<W>':
            in_word = 1
        elif token == '</W>':
            in_word = 0
            word_pos += 1
        elif '<cue' in token:
            if 'negation' in token:
                in_cue.append(
                    int(re.split('(ID=".*?")', token)[1][4:-1]))
                c_idx.append(
                    int(re.split('(ID=".*?")', token)[1][4:-1]))
                if c_idx[-1] not in cue.keys():
                    cue[c_idx[-1]] = []
        elif '</cue' in token:
            in_cue = in_cue[:-1]
        elif '<xcope' in token:
            continue
        elif '</xcope' in token:
            in_scope = in_scope[:-1]
        elif '<ref' in token:
            in_scope.append([int(i) for i in re.split(
                '(SRC=".*?")', token)[1][5:-1].split(' ')])
            s_idx.append([int(i) for i in re.split(
                '(SRC=".*?")', token)[1][5:-1].split(' ')])
            for i in s_idx[-1]:
                scope[i] = []
        elif '</SENTENCE' in token:
            if len(cue.keys()) == 0:
                non_cue_data.append([sentence, [3]*len(sentence), [0]*len(sentence)])
            else:
                cue_sentence.append(sentence)
                cue_cues.append([3]*len(sentence))
                cue_sep.append([0]*len(sentence))
                scope_sentence = []
                scope_subscope = []
                scope_subcues = []
                for count, i in enumerate(cue.keys()):
                    scope_sentence.append(sentence)
                    scope_subcues.append([3]*len(sentence))
                    if len(cue[i]) == 1:
                        cue_cues[-1][cue[i][0]] = 1
                        scope_subcues[-1][cue[i][0]] = 1
                        cue_sep[-1][cue[i][0]] = count + 1
                    else:
                        for c in cue[i]:
                            cue_cues[-1][c] = 2
                            scope_subcues[-1][c] = 2
                            cue_sep[-1][c] = count + 1
                    scope_subscope.append([2]*len(sentence))
                    if i in scope.keys():
                        for s in scope[i]:
                            scope_subscope[-1][s] = 1
                scope_orsents.append(sentence)
                scope_sentences.append(scope_sentence)
                scope_cues.append(scope_subcues)
                scope_scopes.append(scope_subscope)
                num_cue_list.append(len(cue.keys()))
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_pos = 0
            in_word = 0
            c_idx = []
            s_idx = []
        elif '<' not in token:
            if in_word == 1:
                if len(in_cue) != 0:
                    for i in in_cue:
                        cue[i].append(word_pos)
                if len(in_scope) != 0:
                    for i in in_scope:
                        for j in i:
                            scope[j].append(word_pos)
                sentence.append(token)
    non_cue_sents = [i[0] for i in non_cue_data]
    non_cue_cues = [i[1] for i in non_cue_data]
    non_cue_sep = [i[2] for i in non_cue_data]
    non_cue_num = [0 for i in non_cue_data]
    """if param.mark_cue:
        for bi, block in enumerate(scope_cues):
            for ci, c in enumerate(block):
                for i, e in enumerate(c):
                    if e == 1 or e== 2:
                        scope_scopes[bi][ci][i] = 3"""

    return [(cue_sentence+non_cue_sents, cue_cues+non_cue_cues, cue_sep+non_cue_sep, num_cue_list+non_cue_num), 
            (scope_orsents, scope_sentences, scope_cues, scope_scopes), non_cue_sents]

def wiki_weasel(f_path):
    file = open(f_path, encoding='utf-8')
    sentences = []
    for s in file:
        sentences += re.split("(<.*?>)", html.unescape(s))
    cue_sentence = []
    cue_cues = []
    sentence = []
    cue = {}
    word_pos = 0
    total_cue_count = 0
    in_word = 0
    cue_flag = 0
    cue_count = 0
    num_cue_list = []
    cue_sep = []
    sent_count = 0
    for i, token in enumerate(sentences):
        if token == '':
            continue
        elif token.startswith('<Sentence') or token.startswith('<sentence'):
            in_word = 1
            sent_count += 1
        elif '<ccue' in token:
            cue_flag = 1
            cue_count += 1
            total_cue_count += 1
        elif '</ccue' in token:
            cue_flag = 0
        elif '</Sentence' in token or '</sentence' in token:
            if len(cue.keys()) == 0:
                cue_sentence.append(sentence)
                cue_cues.append([2]*len(sentence))
                cue_sep.append([0]*len(sentence))
                num_cue_list.append(0)
            else:
                cue_sentence.append(sentence)
                cue_cues.append([2]*len(sentence))
                cue_sep.append([0]*len(sentence))
                for count, i in enumerate(cue.keys()):
                    if len(cue[i]) == 1:
                        cue_cues[-1][cue[i][0]] = 1
                        cue_sep[-1][cue[i][0]] = count + 1
                    else:
                        for c in cue[i]:
                            cue_cues[-1][c] = 1
                            cue_sep[-1][c] = count + 1
                num_cue_list.append(len(cue.keys()))
            sentence = []
            cue = {}
            word_pos = 0
            in_word = 0
            cue_count = 0
        elif '<' not in token:
            if in_word == 1:
                words = token.split()
                sentence += words
                if cue_flag:
                    if cue_count-1 not in cue.keys():
                        cue[cue_count-1] = []
                    for _ in words:
                        cue[cue_count-1].extend([word_pos + c for c in range(len(words))])
                word_pos += len(words)
    if param.dataset_name == 'wiki2':
        return (cue_sentence[:8343], cue_cues[:8343], cue_sep[:8343], num_cue_list[:8343]),\
            (cue_sentence[8343:11111], cue_cues[8343:11111], cue_sep[8343:11111], num_cue_list[8343:11111]),\
            (cue_sentence[11111:], cue_cues[11111:], cue_sep[11111:], num_cue_list[11111:])
    elif param.dataset_name == 'wiki':
        return cue_sentence, cue_cues, cue_sep, num_cue_list
    else:
        raise NameError('dataset name to be either wiki or wiki2')

def vetcompass(f_text, f_ann, cue_sents_only=False, frac_no_cue_sents = 1.0):
    sentences = []
    cue_sentence = [] #sentence with cues
            
    cue_cues = [] #cues label
    cue_cuesep_ph = []
    num_cues_ph = []
    cue_only_data = [] #(sentences without cues, [3...3])
    scope_cues = [] # == cue_cues
    scope_scopes = [] # scope labels
    scope_ors_ph = []
    scope_sentence = [] # setence with scopes == cue_sentence
            
    #load sents
    s_entities, s_relations, s_attributes, s_groups = get_entities_relations_attributes_groups(f_text)

    #load annotations
    a_entities, a_relations, a_attributes, a_groups = get_entities_relations_attributes_groups(f_ann)

    #sentence {text, cue_labels, scope_labels}
    sentences = {}
    for id, entity in s_entities.items():
        text = entity.text
        spans = entity.span
        type = entity.type
        
        if sentences.get(text) == None:
            sentences[text] = {}
            sentences[text]['start'] = spans[0][0]
            sentences[text]['end'] = spans[0][1]
            sentences[text]['pos'] = [] 
            cur_pos = spans[0][0]
            rel_cur_pos = 0 
            tokens = nltk.word_tokenize(text)
            sentences[text]['tokens'] = tokens
            
            for i, tok in enumerate(tokens):
                cur_pos = spans[0][0] + text.find(tok, rel_cur_pos)
                item = [tok, cur_pos]
                rel_cur_pos = rel_cur_pos + len(tok)
                # cur_pos += len(tok) + 1
                sentences[text]['pos'].append(item)

            # for i,tok in enumerate(text.split()):
            #     # if i == 0:
            #     # 	item = [tok, cur_pos]
            #     # 	rel_cur_pos = rel_cur_pos + len(tok)
            #     # 	sentences[text]['pos'].append(item)
            #     # 	continue
            #     # rel_cur_pos = rel_cur_pos + len(tok)
            #     cur_pos = spans[0][0] + text.find(tok, rel_cur_pos)
            #     item = [tok, cur_pos]
            #     rel_cur_pos = rel_cur_pos + len(tok)
            #     # cur_pos += len(tok) + 1
            #     sentences[text]['pos'].append(item)

            sentences[text]['cue_labels'] = [3] * len(tokens)
            sentences[text]['scope_labels'] = [2] * len(tokens)
    # tag cue labels, scope labels
    for id, entity in a_entities.items():
        text = entity.text
        spans = entity.span
        type = entity.type
        if 'Neg' not in type:
            continue
        if len(text.split()) > 1:
            cue_label = 2
        else:
            cue_label = 1
        for span in spans:
            start = span[0]
            end = span[1]
            for key, value in sentences.items():
                if type == 'NegSignal':
                    if value['start'] <= start and value['end'] >= end:
                        for i,item in enumerate(value['pos']):
                            # if item[1] == start:
                            # 	sentences[key]['cue_labels'][i] = cue_label
                            if item[1] >= start and item[1] <= end:
                                sentences[key]['cue_labels'][i] = cue_label
                else:
                    if value['start'] <= start and value['end'] >= end:
                        for i,item in enumerate(value['pos']):
                            # if item[1] == start:
                            # 	sentences[key]['cue_labels'][i] = cue_label
                            if item[1] >= start and item[1] <= end:
                                sentences[key]['scope_labels'][i] = 1

    for key,value in sentences.items():
        if 1 not in value['cue_labels'] and 2 not in value['cue_labels']:
            cue_only_data.append((value['tokens'], value['cue_labels']))
        else:
            cue_sentence.append(value['tokens'])
            cue_cues.append(value['cue_labels'])
            cue_cuesep_ph.append(value['cue_labels'])
            num_cues_ph.append(1)
            scope_sentence.append(value['tokens'])
            scope_ors_ph.append([value['tokens']])
            scope_cues.append([value['cue_labels']])
            scope_scopes.append([value['scope_labels']])
        
    cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents*len(cue_only_data)))
    cue_only_sents = [i[0] for i in cue_only_samples]
    cue_only_cues = [i[1] for i in cue_only_samples]
    # print(sentences)
    # for key,value in sentences.items():
    #     print(key)
    #     print(value)
    #     print('==================================')
    return [(cue_sentence+cue_only_sents, cue_cues+cue_only_cues, cue_cuesep_ph, num_cues_ph),(scope_sentence, scope_ors_ph, scope_cues, scope_scopes)]


class SplitData():
    def __init__(self, cue, scope, non_cue_sents):
        if isinstance(cue, list):
            self.cues = self.combine_lists(cue)
        if isinstance(scope, list):
            self.scopes = self.combine_lists(scope)
        if isinstance(non_cue_sents, list):
            self.non_cue_sents = self.combine_lists(non_cue_sents)

    def combine_lists(self, input_: List):
        t = []
        for e in input_:
            t.extend(e)
        return t

class SplitMoreData():
    def __init__(self, cue, scope, non_cue_sents):
        if isinstance(cue, list):
            self.cues = self.combine_lists(cue)
        if isinstance(scope, list):
            self.scopes = self.combine_lists(scope)
        if isinstance(non_cue_sents, list):
            self.non_cue_sents = self.pack_(non_cue_sents)

    def pack_(self, input_):
        t = []
        for e in input_:
            t.extend(e)
        return t

    def combine_lists(self, input_: List[List]):
        tmp = input_[0]
        for i, e in enumerate(input_):
            if i == 0:
                continue
            for ii, elem in enumerate(e):
                tmp[ii].extend(elem)
        return tmp


if __name__ == "__main__":
    print()
