import asyncio
import heapq
import pickle
import re
import timeit
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from itertools import chain

import dill
import inflect
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import numpy as np
import networkx as nx

from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from os.path import join
# from thesaurus.thesaurus import Word
import scipy.io as sio

from constants import VOCAB_FREQUENCY, MATLAB_DIR, ROGET_DIR, THESAURUS_COM_DIR
# from thesaurus.thesaurus import fetch_list_of_words


class GeneralTextProcesser(object):
    '''
    A general text preprocesser borrowed from
    https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
    '''
    def tokenize(self,text):
        words = nltk.word_tokenize(text)
        return words

    def remove_non_ascii(self,words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self,words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self,words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self,words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word).split(' ')
                for word in new_word:
                    new_words.append(word)
                # new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    def stem_words(self,words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self,words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def vocab2vec(self, vocab, output_dir, output_name, word_emb, savefmt, type='glove', normalize=False, oov_handle='random'):
        """

        :param vocab: list of words(str)
        :param output_dir: location to put vocab_vec numpy array
        :param word_emb: word embedding file name
        :return: vocab_vec, a numpy array, order is the same with vocab, d*len(vocab)
        """
        print('preparing vocab vectors')
        vocab2pos = {word:i for i,word in enumerate(vocab)}

        if type == 'glove':
            with open(word_emb, 'r') as f:
                _, *vec = next(f).rstrip().split(' ')
                vec_dim = len(vec)
        elif type == 'word2vec':
            word_dictionary = KeyedVectors.load_word2vec_format(word_emb, binary=True)
            vec_dim = word_dictionary.vector_size

        len_vocab = len(vocab)
        vocab_vec_ind = np.ones(len_vocab,dtype=bool)
        vocab_vec = np.zeros((vec_dim,len_vocab))
        mean_emb_vec = np.zeros(vec_dim)

        if type == 'glove':
            with open(word_emb, 'r') as f:
                i = 0
                for line in tqdm(f):
                    word, *vec = line.rstrip().split(' ')
                    vec = np.array(vec, dtype=float)
                    mean_emb_vec += vec
                    if word in vocab2pos:
                        if normalize:
                            vec = vec / np.linalg.norm(vec)
                        vocab_vec[:, vocab2pos[word]] = vec
                        vocab_vec_ind[vocab2pos[word]] = False
                    i += 1
            mean_emb_vec = mean_emb_vec / i
        elif type == 'word2vec':
            for id,word in enumerate(vocab):
                if word in word_dictionary:
                    # todo: normalize word2vec if normalization is needed
                    vocab_vec[:,id] = word_dictionary[word]
                    vocab_vec_ind[id] = False
            mean_emb_vec = np.mean(word_dictionary.vectors,axis=0)

        if normalize:
            mean_emb_vec = mean_emb_vec / np.linalg.norm(mean_emb_vec)
        # handling OOV words in vocab2vec
        # TODO: find better ways to handle OOV
        n_oov = sum(vocab_vec_ind)
        if oov_handle == 'random':
            print('%d words in vocab, %d words not found in word embedding file, init them with random numbers' % (
            len_vocab, n_oov))
            # todo: normalize random vec if normalization is needed
            vocab_vec[:,vocab_vec_ind] = np.random.rand(vec_dim,n_oov)
        elif oov_handle == 'mean_emb_vec':
            print('%d words in vocab, %d words not found in word embedding file, init them with the mean vector' % (
                len_vocab, n_oov))
            vocab_vec[:, vocab_vec_ind] = np.repeat(mean_emb_vec[:,np.newaxis],n_oov,1)
        elif oov_handle == 'none':
            print('%d words in vocab, %d words not found in word embedding file, ignore them in the embedding' % (
                len_vocab, n_oov))
        print('saving vocab vector file')

        if oov_handle != 'none':
            word2vec = OrderedDict((word, vocab_vec[:, id]) for id, word in enumerate(vocab))
        else:
            word2vec = OrderedDict((vocab[id], vocab_vec[:, id]) for id, flag in enumerate(vocab_vec_ind) if flag == False)
            vocab_vec = vocab_vec[:, ~vocab_vec_ind]

        for fmt in savefmt:
            if fmt == 'mat':
                sio.savemat(join(output_dir,output_name+'.mat'),{output_name:vocab_vec})
            elif fmt == 'npy':
                np.save(join(output_dir,output_name+'.npy'),vocab_vec)
            elif fmt == 'pickle':
                with open(join(output_dir,output_name+'.pickle'), 'wb') as handle:
                    pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return word2vec,vocab_vec.T

    def normalize(self,words):
        # TODO: remove non ascii if you don't need ascii
        # words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        # words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        # caution: remove_stopwords may remove the whole sentence and return an empty list
        # words = self.remove_stopwords(words)
        return words


class VocabProcesser(object):

    def prepare_vocab(self,word_corpus,word_embedding,topk,num_vocab):

        lemmatizer = WordNetLemmatizer()
        lemmaed_count_1w = Counter()

        with open(word_corpus,'r') as f:
            for line in f:
                word,count = line.strip().split('\t')
                lemmaed_count_1w[lemmatizer.lemmatize(word)] += int(count)
        topk_vocab = heapq.nlargest(topk, lemmaed_count_1w, key=lemmaed_count_1w.get)

        topk_vocab_vec = OrderedDict.fromkeys(topk_vocab)
        n_non_empty = 0
        with open(word_embedding,'r') as f:
            for line in tqdm(f):
                word,*vec = line.rstrip().split(' ')
                if word in topk_vocab_vec:
                    topk_vocab_vec[word] = np.array(vec, dtype=float)
                    n_non_empty += 1
            print('Num of non empty vectors: ',n_non_empty)

        vocab_vec = np.zeros([300,num_vocab])
        vocab = []
        num = 0
        for k,v in iter(topk_vocab_vec.items()):
            if v is not None:
                vocab_vec[:,num] = v
                vocab.append(k)
                num += 1
            if num >= num_vocab:
                break
        np.save('vocab.npy',vocab)
        np.save('vocab_vec.npy',vocab_vec)

    def prepare_vocab_thesaurus_dict(self,syn_dict,ant_dict,vocab):
        vocab_syn_dict,vocab_ant_dict = {},{}
        set_vocab = set(vocab)
        vocab_id = {word:i for i,word in enumerate(vocab,1)}
        for i,word in enumerate(vocab,1):
            if word in syn_dict:
                int_syn_set = syn_dict[word] & set_vocab
                if int_syn_set:
                    i_syn = [vocab_id[int_word] for int_word in int_syn_set]
                    vocab_syn_dict[i] = i_syn
            if word in ant_dict:
                int_ant_set = ant_dict[word] & set_vocab
                if int_ant_set:
                    i_ant = [vocab_id[int_word] for int_word in int_ant_set]
                    vocab_ant_dict[i] = i_ant

        np.save('vocab_syn_dict.npy',vocab_syn_dict)
        np.save('vocab_ant_dict.npy',vocab_ant_dict)

    def lemmatize_frequent_words(self, file_dir,file_name, lemmatizer):
        lemmaed_word_count = Counter()
        with open(join(file_dir,file_name),'r') as f,\
             open(join(file_dir,'lemmaed_'+file_name),'w') as f_out:
            for line in f:
                word, count = line.strip().split('\t')
                lemmaed_word_count[lemmatizer.lemmatize(word)] += int(count)
            for word,count in lemmaed_word_count.most_common():
                f_out.write(word+'\t'+str(count)+'\n')


    def prepare_vocab_syn_component(self,syn_dict):
        edges = []
        for k,synonyms in iter(syn_dict.items()):
            for v in synonyms:
                edges.append((k,v))
        G = nx.Graph()
        G.add_edges_from(edges)
        components = set(frozenset(component) for component in nx.connected_components(G))
        np.save('vocab_syn_components.npy',components)

    def join_vocab(self,out_dir_name,*vocabs):
        joined_vocab = set()
        for vocab in vocabs:
            joined_vocab.update(vocab)
        np.save(join(out_dir_name,'vocab.npy'), list(joined_vocab))

    def select_vocab(self,vocab_dir,vocab,frequency_file,select_th):
        set_vocab = set(vocab)
        selected_vocab = set()
        with open(frequency_file, 'r') as f:
            i = 0
            for line in f:
                word, _ = line.strip().split('\t')
                if word in set_vocab:
                    selected_vocab.add(word)
                    i += 1
                if i >= select_th:
                    break
        np.save(join(vocab_dir,'selected_vocab.npy'), list(selected_vocab))

class ThesaurusPreProcesser(object):

    def construct_wn_thesauri(self,thesauri_dir):
        """
        1. T <--- extract all wordnet words
        2. for a word in T, find its wordnet synonyms and antonyms, save them in two sets
        3. for all words and their corresponding sets, save them in two dicts
        4. save two dicts as npy files
        if a word doesn't appear in the dict, it will return an empty set
        :param thesauri_dir: pos to save syn_dict and ant_dict
        :return: None
        """

        # key: word, value: syn_words(set)
        syn_dict = defaultdict(set)
        # key: word, value: ant_words(set)
        ant_dict = defaultdict(set)

        # get synonyms from wordnet
        # TODO: configure this in the future
        # since a word has multiple meanings we combine all the synonyms as its synonyms
        # this can be configured in the future
        # and all the antonyms are its antonyms
        for word in tqdm(wn.words()):
            synonyms, antonyms = [], []
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
                    antonyms.append([ant_lemma.name() for ant_lemma in lemma.antonyms()])
            # add word's synonyms
            syn_dict[word].update(synonyms)
            # remove itself in syn set
            if word in syn_dict[word]:
                syn_dict[word].remove(word)
            # add word's antonyms
            ant_dict[word].update(chain.from_iterable(antonyms))
            # remove itself in ant set
            if word in ant_dict[word]:
                ant_dict[word].remove(word)

            # since a word t may have multiple meanings, a same word w could be its synonym and antonym
            # for this case, remove w in t's synonym
            syn_dict[word] = syn_dict[word] - ant_dict[word]

            # CAUTION: original wordnet is asymmetric on synonyms and antonyms
            # make syn_dict symmetric
            for synonym in syn_dict[word]:
                syn_dict[synonym].add(word)
            # make ant_dict symmetric
            for antonym in ant_dict[word]:
                ant_dict[antonym].add(word)


        # save synonym/antonym dict
        np.save(join(thesauri_dir,'syn_dict.npy') , syn_dict)
        np.save(join(thesauri_dir,'ant_dict.npy') , ant_dict)

    def construct_wn_vocab(self,thesauri_dir):
        with open(join(thesauri_dir,'wn_vocab.txt'),'w') as f:
            for word in tqdm(wn.words()):
                f.write(word+'\n')


    def prepare_wn_common_thesauri(self, vocab_file_name, syn_dict, ant_dict, output_dir, len_common_syn, len_common_ant):
        # TODO: common_thesauri has no order
        common_syn_dict = defaultdict(set)
        common_ant_dict = defaultdict(set)
        n_syn,n_ant = 0,0
        with open(vocab_file_name, 'r') as f:
            for line in f:
                word,_ = line.strip().split('\t')
                if syn_dict[word] and n_syn < len_common_syn:
                    common_syn_dict[word].update(syn_dict[word])
                    n_syn += 1
                if ant_dict[word] and n_ant < len_common_ant:
                    common_ant_dict[word].update(ant_dict[word])
                    n_ant += 1
                if n_syn > len_common_syn and n_ant > len_common_ant : break
        np.save(join(output_dir, 'common_wn_syn_dict.npy'), common_syn_dict)
        np.save(join(output_dir, 'common_wn_ant_dict.npy'), common_ant_dict)
        np.save(join(output_dir, 'common_wn_vocab_keys.npy'),list(set(common_syn_dict.keys()) | set(common_ant_dict.keys())))

    # TODO: combine multiple thesaurus

    def search_vocab_in_wn(self, vocab, syn_dict_name,ant_dict_name):
        """
        given a vocab, list of string, ouput synonym and antonym dicts, where keys are words in vocab list
        values are words' synonyms/antonyms in wordnet, value type is set of string, if a word has no synonyms or antonyms, its
        value would be an empty set, if a word is not in dict, return an empty set
        :param vocab:
        :return:
        """
        syn_dict = defaultdict(set)
        ant_dict = defaultdict(set)
        for word in vocab:
            synonyms, antonyms = [], []
            # only need the most common synset
            for syn in wn.synsets(word)[:1]:
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
                    antonyms.append([ant_lemma.name() for ant_lemma in lemma.antonyms()])

            # add word's synonyms
            syn_dict[word].update(synonyms)
            # remove itself in syn set
            if word in syn_dict[word]:
                syn_dict[word].remove(word)
            # add word's antonyms
            ant_dict[word].update(chain.from_iterable(antonyms))
            # remove itself in ant set
            if word in ant_dict[word]:
                ant_dict[word].remove(word)

            # since a word t may have multiple meanings, a same word w could be its synonym and antonym
            # for this case, remove w in t's synonym
            # TODO: ant_dict should not contain syn
            syn_dict[word] = syn_dict[word] - ant_dict[word]

        # save synonym/antonym dict
        np.save(syn_dict_name, syn_dict)
        np.save(ant_dict_name, ant_dict)

    def combine_vocab_dict(self,vocab,task_vocab,syn_dicts,ant_dicts,syn_dict_name,ant_dict_name):
        syn_dict = defaultdict(set)
        ant_dict = defaultdict(set)
        for word in vocab:
            word_synonyms = set(syn_dicts[0][word])
            for i in range(1,len(syn_dicts)):
                word_synonyms |= syn_dicts[i][word]
            # we only need to clustering task specific vocab
            word_synonyms &= task_vocab

            word_antonyms = set(ant_dicts[0][word])
            for i in range(1,len(ant_dicts)):
                word_antonyms |= ant_dicts[i][word]
            word_antonyms &= task_vocab

            # word_synonyms -= word_antonyms
            if word_antonyms & word_synonyms:
                print('%s has an intersect in synonyms and antonyms')
                print('they are %s' %(' '.join(word_antonyms & word_synonyms)))

            if word_synonyms:
                syn_dict[word].update(word_synonyms)
            if word_antonyms:
                ant_dict[word].update(word_antonyms)
        # save synonym/antonym dict
        np.save(syn_dict_name, syn_dict)
        np.save(ant_dict_name, ant_dict)


class StanfordSentimentProcesser(object):

    def split_train_dev_test(self, sentiment_dir):
        '''
        split train dev and test from downloaded data
        :param sentiment_dir:
        :return:
        '''
        TRAIN_LABEL = '1'
        TEST_LABEL = '2'
        DEV_LABEL = '3'

        label2sen_id = defaultdict(list)
        with open(join(sentiment_dir,'datasetSplit.txt'), 'r') as f:
            f.readline()
            for line in f:
                id, label = line.strip().split(',')
                if label is TRAIN_LABEL:
                    label2sen_id[TRAIN_LABEL].append(id)
                elif label is TEST_LABEL:
                    label2sen_id[TEST_LABEL].append(id)
                elif label is DEV_LABEL:
                    label2sen_id[DEV_LABEL].append(id)

        phrase2id = {}
        with open(join(sentiment_dir,'dictionary.txt'), 'r') as f:
            for line in f:
                phrase, id = line.strip().split('|')
                phrase2id[phrase] = id

        id2sen_label = {}
        with open(join(sentiment_dir,'sentiment_labels.txt'), 'r') as f:
            f.readline()
            for line in f:
                id, v = line.strip().split('|')
                id2sen_label[id] = 0 if float(v) <= 0.5 else 1

        sen_id2sen = {}
        with open(join(sentiment_dir,'datasetSentences.txt'), 'r') as f:
            f.readline()
            for line in f:
                sen_id, sen = line.strip().split('\t')
                # 'Ã©' to 'é'
                sen = sen.encode('latin-1').decode('utf-8')
                sen = sen.replace('-LRB-', '(')
                sen = sen.replace('-RRB-', ')')
                sen_id2sen[sen_id] = sen

        with open(join(sentiment_dir,'train.txt'), 'w') as f:
            for sen_id in label2sen_id[TRAIN_LABEL]:
                f.write(sen_id2sen[sen_id] + '\t' + str(id2sen_label[phrase2id[sen_id2sen[sen_id]]]) + '\n')

        with open(join(sentiment_dir,'dev.txt'), 'w') as f:
            for sen_id in label2sen_id[DEV_LABEL]:
                f.write(sen_id2sen[sen_id] + '\t' + str(id2sen_label[phrase2id[sen_id2sen[sen_id]]]) + '\n')

        with open(join(sentiment_dir,'test.txt'), 'w') as f:
            for sen_id in label2sen_id[TEST_LABEL]:
                f.write(sen_id2sen[sen_id] + '\t' + str(id2sen_label[phrase2id[sen_id2sen[sen_id]]]) + '\n')

    def process_train_dev_test(self, read_fname, write_fname, text_processer):
        # train/dev/test.txt -> processed_train/dev/test.txt
        with open(read_fname, 'r') as f, open(write_fname, 'w') as f_out:
            print('processing ' + read_fname)

            for line in f:
                sen, score = line.strip().split('\t')

                words = text_processer.normalize(sen.split())
                words = text_processer.lemmatize_verbs(words)

                # skip empty words list
                if not words:
                    continue

                f_out.write(' '.join(words) + '\t' + score + '\n')

    def extract_vocab_from_processed_file(self,read_fname):
        # processed_train.txt -> vocab(set)
        vocab = set()
        with open(read_fname,'r') as f:
            for line in f:
                sen,_ = line.strip().split('\t')
                vocab.update(sen.split(' '))

        return vocab


class SimLexProcesser(object):

    def prepare_syn_ant_pair_vocab(self, simlex_dir):
        vocab = set()
        with open(join(simlex_dir,'SimLex-999.txt') ,'r') as f,\
             open(join(simlex_dir,'syn_word_pairs.txt') ,'w') as f_syn,\
             open(join(simlex_dir,'ant_word_pairs.txt') ,'w') as f_ant:
            f.readline()
            for line in f:
                word1,word2,pos,sim,*_ = line.strip().split('\t')
                if float(sim) > 8:
                    f_syn.write(word1+'\t'+word2+'\n')
                    vocab.update([word1,word2])
                elif float(sim) < 2:
                    f_ant.write(word1+'\t'+word2+'\n')
                    vocab.update([word1, word2])
        np.save(join(simlex_dir, 'selected_vocab.npy'), list(vocab))

class SimVerbProcesser(object):

    def prepare_syn_ant_pair_vocab(self,simverb_dir):
        vocab = set()
        with open(join(simverb_dir, 'SimVerb-3500.txt'), 'r') as f1, \
             open(join(simverb_dir, 'syn_word_pairs.txt'), 'w') as f_syn, \
             open(join(simverb_dir, 'ant_word_pairs.txt'), 'w') as f_ant:
            for line in f1:
                word1, word2, pos, sim, *_ = line.strip().split('\t')
                if float(sim) > 8:
                    f_syn.write(word1 + '\t' + word2 + '\n')
                    vocab.update([word1, word2])
                elif float(sim) < 2:
                    f_ant.write(word1 + '\t' + word2 + '\n')
                    vocab.update([word1, word2])

        np.save(join(simverb_dir, 'selected_vocab.npy'), list(vocab))
