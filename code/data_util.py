from transformers import BertTokenizer, RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import pdb
from collections import Counter
import queue
from tqdm import tqdm

debug = False


class DatasetReader():

    def __init__(self, max_len, BERT_name):
        self.max_len = max_len
        self.bertname = BERT_name

    def read_data(self,
                  filename,
                  aspect_only_by_rule=False,
                  mask_version=False,
                  return_origin=False,
                  filter=False,
                  id_filter_list=None,
                  is_sentihood=False,
                  is_test=False,
                  select_test_list=None
                  ):
        dj = data_loader(filename)
        if is_sentihood: label_map = {'positive':1, 'negative':0}
        else: label_map = {'positive': 2, 'neutral': 1, 'negative': 0}

        if filter:
            if is_test:
                pairs = [(int(did), int(tid), data['original_text'], term['term'], term['answers'],
                          label_map[term['polarity']])
                         for did, data in dj.items()
                         for tid, term in data['terms'].items() if
                         (int(did), int(tid)) in select_test_list]
            else:
                pairs = [(int(did), int(tid), data['original_text'], term['term'], term['answers'],
                          label_map[term['polarity']])
                         for did, data in dj.items()
                         for tid, term in data['terms'].items() if
                         (int(did), int(tid)) not in id_filter_list]
        else:
            pairs = [
                (int(did), int(tid), data['original_text'], term['term'], term['answers'], label_map[term['polarity']])
                for did, data in dj.items()
                for tid, term in data['terms'].items()
            ]
        dids = [did for (did, tid, a, t, an, b) in pairs]
        tids = [tid for (did, tid, a, t, an, b) in pairs]
        origin_sentences = [a for (did, tid, a, t, an, b) in pairs]
        terms = [t for (did, tid, a, t, an, b) in pairs]
        labels = [b for (did, tid, a, t, an, b) in pairs]

        print('Data Size is: {0}, It is mask version? {1}, label bias: {2}'.format(len(labels), mask_version,
                                                                                   Counter(labels)))
        if self.bertname == 'bert-base-uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            sentences = [
                '[CLS] ' + t + ' [SEP] ' + self.mask_tokens(s, an, tokenizer, t, mask_version, '[MASK]', '[CLS]',
                                                            '[SEP]') + ' [SEP] ' for (did, tid, s, t, an, b) in pairs]
        elif self.bertname == 'roberta-large':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
            sentences = []
            for did, tid, s, t, an, b in pairs:
                masked_sentence = self.mask_tokens(s, an, tokenizer, t, mask_version, '<mask>', '<s>', '</s>',
                                                   start_symbol='Ġ')
                if masked_sentence:
                    sentence = '<s> ' + t + ' </s> <s> ' + masked_sentence + ' </s> '
                else:
                    sentence = '<s> ' + '<mask>' + ' </s> <s> ' + s + ' </s> '
                sentences.append(sentence)

        encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)

        if return_origin:
            return dids, tids, encoding['input_ids'], encoding['attention_mask'], labels, [dids, tids, origin_sentences,
                                                                                           terms, labels]

        return dids, tids, encoding['input_ids'], encoding['attention_mask'], labels

    def mask_tokens(self, sentence, answers, tokenizer, term, to_mask, mask_token, start_token, end_token,
                    start_symbol=''):

        # if not to mask, return sentence itself.
        if not to_mask: return sentence

        tokens = tokenizer.tokenize(sentence)
        tokens = [token.strip(start_symbol) for token in tokens]
        tokenized_sentence = ' '.join(tokens)

        valid_answers = [ans for aid, ans in answers[0].items()
                         if aid not in 'rules'
                         and start_token not in ans
                         and end_token not in ans
                         and len(ans) > 0
                         ]

        if len(valid_answers) == 0:
            # if no answer, mask tokens between close split punctuations
            puncsidx = [i for i, token in enumerate(tokens) if token in set([',', '.', '?', '!', '--'])]
            termidx = [i for i, token in enumerate(tokens) if token == term]
            if len(termidx) == 0:
                if debug: print('no term matched. return whole sentence.')
                return None  # can not find term in sentence. Skip it.
            if len(puncsidx) == 0:
                if debug: print('Term matched but no punc. Mask till end of sentence.')
                return ' '.join([token for i, token in enumerate(tokens) if
                                 i > max(termidx)])  # mask words that are after the last matched term.
            maskposs = []
            for tid in termidx:
                # search to right for punc
                endpos = [pid for pid in puncsidx if pid > tid]
                if len(endpos) == 0:
                    endpos = len(tokens)
                else:
                    endpos = min(endpos)
                # search backward for punc
                stpos = [pid for pid in puncsidx if pid < tid]
                if len(stpos) == 0:
                    stpos = 0
                else:
                    stpos = max(stpos)
                maskposs.append((stpos, endpos))
            for stpos, endpos in maskposs:
                tokens = [mask_token if i >= stpos and i < endpos else token for i, token in enumerate(tokens)]
            if debug: print('no answer matched, but heuristically matched a span around term.')
            return ' '.join(tokens)

        answers = Counter(valid_answers)
        matched = None
        for ans, freq in answers.most_common():
            stpos = tokenized_sentence.find(ans)
            if stpos == -1:
                breakpoint()
            edpos = stpos + len(ans)
            if debug: print('answer whole string matched')
            return tokenized_sentence[:stpos] + ' ' + ' '.join([mask_token] * len(ans.split(' '))) + ' ' + sentence[
                                                                                                           edpos:]

        # mask individual words instead.
        for ans, freq in answers.most_common():
            anstoks = {a: 0 for a in ans.strip().split(' ')}
            if debug: print('answer per-term matched ')
            return ' '.join([mask_token if token in anstoks else token for token in tokens])

    # Deprecated
    def mask(self, sentence, answers, tokenizer, term, to_mask, mask_token, start_token, end_token):
        if not to_mask:
            return sentence
        valid_answers = [ans for aid, ans in answers[0].items()
                         if aid not in 'rules'
                         and start_token not in ans
                         and end_token not in ans
                         and len(ans) > 0
                         ]
        answers = Counter(valid_answers)
        matched = None
        for ans, freq in answers.most_common():
            matched = self.fuzzymatch(sentence, ans, mask_token)
            if matched:
                print(f'span found! term: {term} - answer:{ans} -sentence:{sentence}')
                return matched

        print(f'span not found from all answers!! term: {term} -sentence:{sentence}')
        return self.span_mask(sentence, term, mask_token)


def data_loader(filename):
    data = json.load(open(filename, 'r'))
    return data
