import os

class Param(object):
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = {
            'sfu': 'Data/SFU_Review_Corpus_Negation_Speculation',
            'bioscope_abstracts': 'Data/bioscope/abstracts.xml',
            'bioscope_full': 'Data/bioscope/full_papers.xml',
            'sherlock': {
                #'train': 'Data/starsem-st-2012-data/cd-sco/corpus/training/dtrain.txt',
                #'dev': 'Data/starsem-st-2012-data/cd-sco/corpus/dev/ddev.txt',
                'train': 'Data/starsem-st-2012-data/cd-sco/corpus/training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt',
                'dev': 'Data/starsem-st-2012-data/cd-sco/corpus/dev/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt',
                'test1': 'Data/starsem-st-2012-data/cd-sco/corpus/test-gold/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt',
                'test2': 'Data/starsem-st-2012-data/cd-sco/corpus/test-gold/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'
            },
            'wiki': {
                'train': 'Data/uncertainty/task1_train_wikipedia_rev2.xml',
                'test': 'Data/uncertainty/task1_weasel_eval.xml',
            },
            'wiki2': 'Data/uncertainty/wiki.xml',
            'vet': {
                'train': 'Data/vetcompass_subsets/train/negspec',
                'dev': 'Data/vetcompass_subsets/dev/negspec',
                'test': 'Data/vetcompass_subsets/test/negspec',
            }
        }
        self.split_path = {
            'sfu': {
                'cue': {
                    'train': 'split/train_cue_sfu.pt',
                    'dev': 'split/dev_cue_sfu.pt',
                    'test': 'split/test_cue_sfu.pt',
                },
                'scope': {
                    'train': 'split/train_scope_sfu.pt',
                    'dev': 'split/dev_scope_sfu.pt',
                    'test': 'split/test_scope_sfu.pt',
                },
                'joint_cue': {
                    'train': 'split/joint_train_cue_sfu.pt',
                    'dev': 'split/joint_dev_cue_sfu.pt',
                    'test': 'split/joint_test_cue_sfu.pt',
                },
                'joint_scope': {
                    'train': 'split/joint_train_scope_sfu.pt',
                    'dev': 'split/joint_dev_scope_sfu.pt',
                    'test': 'split/joint_test_scope_sfu.pt',
                }

            },
            'bioscope_abstracts': {
                'cue': {
                    'train': 'split/train_cue_bioA.pt',
                    'dev': 'split/dev_cue_bioA.pt',
                    'test': 'split/test_cue_bioA.pt',
                },
                'scope': {
                    'train': 'split/train_scope_bioA.pt',
                    'dev': 'split/dev_scope_bioA.pt',
                    'test': 'split/test_scope_bioA.pt',
                },
                'joint_cue': {
                    'train': 'split/joint_train_cue_bioA.pt',
                    'dev': 'split/joint_dev_cue_bioA.pt',
                    'test': 'split/joint_test_cue_bioA.pt',
                },
                'joint_scope': {
                    'train': 'split/joint_train_scope_bioA.pt',
                    'dev': 'split/joint_dev_scope_bioA.pt',
                    'test': 'split/joint_test_scope_bioA.pt',
                }
            },
            'bioscope_full': {
                'cue': {
                    'train': 'split/train_cue_bioF.pt',
                    'dev': 'split/dev_cue_bioF.pt',
                    'test': 'split/test_cue_bioF.pt',
                },
                'scope': {
                    'train': 'split/train_scope_bioF.pt',
                    'dev': 'split/dev_scope_bioF.pt',
                    'test': 'split/test_scope_bioF.pt',
                },
                'joint_cue': {
                    'train': 'split/joint_train_cue_bioF.pt',
                    'dev': 'split/joint_dev_cue_bioF.pt',
                    'test': 'split/joint_test_cue_bioF.pt',
                },
                'joint_scope': {
                    'train': 'split/joint_train_scope_bioF.pt',
                    'dev': 'split/joint_dev_scope_bioF.pt',
                    'test': 'split/joint_test_scope_bioF.pt',
                }
            }
        }
        self.seg_path = {
            'sfu': 'seg/sfu_seg.bin',
            'bioscope_abstracts': 'seg/bioscope_a_seg.bin',
            'bioscope_full': 'seg/bioscope_f_seg.bin',
            'sherlock': {
                'train': 'seg/train_seg_s.bin',
                'dev': 'seg/dev_seg_s.bin',
                'test': 'seg/test_seg_s.bin',
            },
            'sherlock_com': {
                'train': 'seg/train_seg_c.bin',
                'dev': 'seg/dev_seg_c.bin',
                'test': 'seg/test_seg_c.bin',
            }
        }
        self.split_and_save = False
        self.test_only = False # Assign True when only performing testing
        self.num_runs = 7
        self.dataset_name = 'bioscope_abstracts' # Available options: 'bioscope_full', 'bioscope_abstracts', 'sherlock', 'sfu', 'vet'
        self.task = 'scope' # Available options: 'cue', 'scope', 'pipeline', 'joint'
        self.predict_cuesep = False # Specify whether to predict the cue seperation
        self.model_name = f'{self.task}_roberta_bsl_alpha_02_{self.dataset_name}'

        self.embedding = 'BERT' # Available options: Word2Vec, FastText, GloVe, BERT\
        if self.embedding != 'BERT':
            if self.embedding == 'FastText':
                self.emb_cache = 'Dev/Vector/generated.bin'
        self.bert_path = 'roberta-base' #'bert-base-cased' #'xlm-roberta-large' 'facebook/bart-base''SpanBERT/spanbert-base-cased' 'roberta-base' 
        self.bert_cache = f'BERT/{self.bert_path}'
        self.is_bert = self.embedding == 'BERT'

        self.sherlock_seperate_affix = False
        self.sherlock_ex_combine_nt = False

        self.use_focal_loss = True
        self.ignore_multiword_cue = False
        self.cased = True
        self.max_len = 265
        self.batch_size = 8
        self.num_ep = 80
        self.lr = 5e-5
        self.early_stop_thres = 12
        
        self.gru_or_lstm = 'LSTM'
        self.scope_method = 'augment' # Available options: augment, replace
    
        self.word_emb_dim = 300
        self.lstm_emb_type = 'pre_emb'
        self.cue_emb_dim = 10
        self.position_emb_dim = 0
        self.hidden_dim = 200
        self.dropout = 0.1
        self.biaffine_hidden_dropout = 0.33
        self.label_dim = 3 # number of output labels [0:pad, 1:scope, 2:out, 3:cue]
        self.mark_cue = True # To mark cue in scope, instead of treaing as part of scope
        if self.mark_cue:
            self.label_dim += 1
        self.bio = False
        if self.bio:
            self.label_dim += 1
        self.max_num_cues = 4
        self.cue_matrix = False
        self.matrix = False
        self.fact = False # Factorized matrix 
        self.m_dir = 'd2'
        self.cue_mode = 'diag' # 'root' or 'diag'
        if self.m_dir == 'd1':
            self.label_dim += 1
        #if self.cue_mode == 'root':
        #    self.label_dim -= 1 
        self.augment_cue = 'surround' # surround, front, None
        if self.task == 'joint':
            self.augment_cue = False
            self.label_dim += 1
        self.encoder_attention = None # 'meta'
        self.decoder_attention = [] # 'multihead', 'label'
        self.num_attention_head = 5
        self.external_vocab = False
        self.use_crf = False
        if self.use_crf is True:
            self.label_dim += 2

        self.multi = False
        self.ignore_multi_negation = False
        self.aug_seq = False # True when using the generator setting (generate augmented sent)
        self.edu_sep = False

        self.temp = False
        self.boundary = True
        self.boundary_label_num = 4
        self.bound_only = False
        self.bsl_warmup = 4

param = Param()
