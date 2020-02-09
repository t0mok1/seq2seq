from collections import Counter
from pathlib import Path
import torch
import copy

def make_vocab(text, initial_vocab={}, vocabsize=0, freq=0):
    """
    text[list]]: 入力文のリスト
    initial_vocab[dict]: 辞書の初期値
    vocabsize[int]: 辞書のサイズ
    freq[int]: 登録する単語の最低頻度
    """
    vocab = copy.copy(initial_vocab)
    word_count = Counter()
    for line in text:
        for w in line.split():
            word_count[w] += 1
    if vocabsize > 0: # 辞書サイズが指定されている場合は上から順に登録
        for w in word_count.most_common(vocabsize):
            if w[1] < freq:
                break
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocabsize + len(initial_vocab):
                    break
    else: #辞書サイズが指定されていない場合は,すべての単語を辞書に登録
        for w in word_count.most_common():
            if w[1] < freq:
                break
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
        
    return vocab

def get_vocab(train_file, initial_vocab={'<pad>':0,'<unk>':1,'<sos>':2,'<eos>':3},\
              source_vocabsize=0, target_vocabsize=0, source_freq=0, target_freq=0):
    """
    train_file[str]: train.txtのファイルパス
    initial_vocab[dict]: 辞書の初期値
    source_voacb[int]: 翻訳元言語の辞書サイズ
    target_voacb[int]: 翻訳先言語の辞書サイズ
    source_freq[int]: 翻訳元言語の登録する単語の最低頻度
    target_freq[int]: 翻訳先言語の登録する単語の最低頻度
    """
    source_txt = []
    target_txt = []

    train_file = Path(train_file)
    with train_file.open('r', encoding='utf-8') as f:
        for line in f:
            data = line.split('|||')
            source_txt.append(data[1].strip())
            target_txt.append(data[0].strip())
    source_vocab = make_vocab(source_txt, initial_vocab, source_vocabsize, source_freq)
    target_vocab = make_vocab(target_txt, initial_vocab, target_vocabsize, target_freq)
    
    return source_vocab, target_vocab

            
def save_checkpoint(path, epoch, bleu, step_num, model_param, optim_param, is_best=False):
    
    state = {'epoch' : epoch,
             'bleu' : bleu,
             'step-num' : step_num,
             'model' : model_param,
             'optim' : optim_param
    }
    
    filename = path + 'model.pth.tar'
    
    if is_best:
        torch.save(state, path + 'best_checkpoint.pth.tar')
    else:
        torch.save(state, filename)
            
def load_checkpoint(path, model=None, optimizer=None):
    state = torch.load(path)
    if model is not None:
        model.load_state_dict(state['model'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])

def adjust_learning_rate(optimizer, shrink_factor):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
