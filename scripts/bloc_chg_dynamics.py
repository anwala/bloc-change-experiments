import json
import math
import random
import re
import sys
import statistics

from collections import Counter
from copy import deepcopy
from datetime import datetime
from datetime import timedelta

from bloc.util import cosine_sim
from bloc.util import get_bloc_variant_tf_matrix
from bloc.util import conv_tf_matrix_to_json_compliant
from bloc.util import fold_word

from scipy.stats import entropy

def get_normalized_entropy(labels):
    
    #credit: https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147
    #normalized entropy: https://search.r-project.org/CRAN/refmans/posterior/html/entropy.html
    #normalized entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)#Efficiency_(normalized_entropy)
    #normalized entropy: https://math.stackexchange.com/a/945172

    labels = list(labels)
    labels = Counter(labels)
    base = 2

    #method 1: entropy( list(labels.values()), base=len(labels) )
    #method 2 (entropy/max-entropy):
    entrpy = entropy(list(labels.values()), base=base)
    normalizer = math.log(len(labels), base)
    
    return entrpy if normalizer == 0 else entrpy/normalizer

def get_bloc_vocab_doc_len_linear_model():
    
    return {
        'action': {'slope': 0.2198, 'intercept': 0.5700, 'r_score': 0.1078},
        'content_syntactic': {'slope': 0.5292, 'intercept': 0.1428, 'r_score': 0.1778}
    }

def predict_vocab_and_doc_len(doc_len, bloc_alphabet):

    bloc_vdl_model = get_bloc_vocab_doc_len_linear_model()

    if( bloc_alphabet == 'action' ):
        slope = bloc_vdl_model['action']['slope']
        intercept = bloc_vdl_model['action']['intercept']
    else:
        slope = bloc_vdl_model['content_syntactic']['slope']
        intercept = bloc_vdl_model['content_syntactic']['intercept']
    
    if( doc_len == 0 ):
        return -1

    vocab_len = slope * math.log10(doc_len) + intercept

    return 10**vocab_len


def get_bloc_entropy(bloc_tweets, bloc_doc, bloc_alphabet, **kwargs):

    result = {'word_entropy': -1, 'char_entropy': -1}
    gen_bloc_params = {
        'ngram': 1, 
        'min_df': 1, 
        'tf_matrix_norm': '', 
        'keep_tf_matrix': True,
        'set_top_ngrams': True, 
        'top_ngrams_add_all_docs': False,
        'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'
    }
     
    fold_start_count = kwargs.get('fold_start_count', 3)
    bloc_variant = {'type': 'folded_words', 'fold_start_count': fold_start_count, 'count_applies_to_all_char': False}

    tf_matrices = get_bloc_variant_tf_matrix(
        [{'text': bloc_doc}], 
        ngram=gen_bloc_params['ngram'], 
        min_df=gen_bloc_params['min_df'], 
        tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
        keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
        set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
        top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
        bloc_variant=bloc_variant,
        token_pattern=gen_bloc_params['token_pattern']
    )
    
    if( 'vocab' not in tf_matrices ):
        return result

    tf_matrices = conv_tf_matrix_to_json_compliant(tf_matrices)
    result['vocab_len'] = len(tf_matrices['vocab'])
    result['top_bloc_words'] = tf_matrices['top_ngrams']['per_doc'][0]

    normalizer = math.log(result['vocab_len'], 2)
    result['word_entropy'] = entropy(tf_matrices['tf_matrix'][0]['tf_vector'], base=2)
    result['word_entropy'] = result['word_entropy'] if normalizer == 0 else result['word_entropy']/normalizer 
    result['char_entropy'] = get_normalized_entropy(re.sub(r'([ |()])', '', bloc_doc))

    return result

def get_vocab_and_doc_len(bloc_doc, bloc_alphabet, **kwargs):

    #print('bloc_doc:', bloc_doc)
    result = {'vocab_len': 0, 'doc_len': 0}
    gen_bloc_params = {
        'ngram': 1, 
        'min_df': 1, 
        'tf_matrix_norm': '', 
        'keep_tf_matrix': True,
        'set_top_ngrams': True, 
        'top_ngrams_add_all_docs': False,
        'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'
    }
     
    fold_start_count = kwargs.get('fold_start_count', 3)
    bloc_variant = {'type': 'folded_words', 'fold_start_count': fold_start_count, 'count_applies_to_all_char': False}

    tf_matrices = get_bloc_variant_tf_matrix(
        [{'text': bloc_doc}], 
        ngram=gen_bloc_params['ngram'], 
        min_df=gen_bloc_params['min_df'], 
        tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
        keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
        set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
        top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
        bloc_variant=bloc_variant,
        token_pattern=gen_bloc_params['token_pattern']
    )
    
    if( 'vocab' not in tf_matrices ):
        return result

   
    result['vocab_len'] = len(tf_matrices['vocab'])
    result['doc_len'] = int(tf_matrices['tf_matrix'][0]['tf_vector'].sum())
    result['predicted_vocab_len'] = predict_vocab_and_doc_len(result['doc_len'], bloc_alphabet)
    result['prediction_error'] = abs(result['predicted_vocab_len'] - result['vocab_len'])/result['vocab_len']
    result['top_bloc_words'] = tf_matrices['top_ngrams']['per_doc'][0]
    
    return result


def slide_time_window_over_bloc_tweets_v0(bloc_tweets, alphabet, window_size_seconds=3600, window_mv_rate_seconds=1800, **kwargs):
    
    def count_word_switches(bloc_words, fold_start_count):

        if( len(bloc_words) == 0 ):
            return 0

        fmt_words = [fold_word(w, fold_start_count=fold_start_count) for w in bloc_words]
        return len([ True for i in range(1, len(fmt_words)) if fmt_words[i-1] != fmt_words[i] ])
        

    if( len(bloc_tweets) == 0 ):
        return {}

    gen_bloc_params = {
        'ngram': 1, 
        'min_df': 1, 
        'tf_matrix_norm': '', 
        'keep_tf_matrix': True,
        'set_top_ngrams': False, 
        'top_ngrams_add_all_docs': False,
        'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'
    }
    fold_start_count = kwargs.get('fold_start_count', 3)

    time_bins = {}
    start_datetime = datetime.strptime(bloc_tweets[0]['created_at'], '%a %b %d %H:%M:%S %z %Y')
    end_datetime = start_datetime + timedelta(seconds=window_size_seconds)
    terminal_datetime = datetime.strptime(bloc_tweets[-1]['created_at'], '%a %b %d %H:%M:%S %z %Y')

    change_dynamics = {'change_dynamics': [], 'avg_change_dynamics': {}, 'change_dynamics_score': 0}
    bloc_variant = {'type': 'folded_words', 'fold_start_count': fold_start_count, 'count_applies_to_all_char': False}

    #slide a time window from the bloc_tweets[0]['created_at'] to bloc_tweets[-1]['created_at']
    while( True ):

        time_bins[ (start_datetime, end_datetime) ] = []
        start_datetime = start_datetime + timedelta(seconds=window_mv_rate_seconds)
        end_datetime = start_datetime + timedelta(seconds=window_size_seconds)
        
        if( start_datetime > terminal_datetime ):
            time_bins[ (start_datetime, end_datetime) ] = []
            break

    #populate time bin with tweets that fall into time bins
    for i in range( len(bloc_tweets) ):
        tweet_time = datetime.strptime(bloc_tweets[i]['created_at'], '%a %b %d %H:%M:%S %z %Y')
        for date_rng, tweets in time_bins.items():
            start_datetime, end_datetime = date_rng
            if( tweet_time >= start_datetime and tweet_time <= end_datetime ):
                tweets.append(i)

    dist = None
    prev_doc = ''
    prev_tweet_indices = []
    for date_rng, tweet_indices in time_bins.items():

        if( len(tweet_indices) == 0 ):
            continue

        if( set(prev_tweet_indices) == set(tweet_indices) ):
            continue

        doc = ''
        for indx in tweet_indices:
            if( 'bloc' not in bloc_tweets[indx] ):
                continue
            if( 'bloc_sequences_short' not in bloc_tweets[indx]['bloc'] ):
                continue
            doc = doc + bloc_tweets[indx]['bloc']['bloc_sequences_short'].get(alphabet, '')
        
        if( doc == '' ):
            continue
        prev_tweet_indices = tweet_indices

        tf_matrices = get_bloc_variant_tf_matrix(
            [{'text': doc}], 
            ngram=gen_bloc_params['ngram'], 
            min_df=gen_bloc_params['min_df'], 
            tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
            keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
            set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
            top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
            bloc_variant=bloc_variant,
            token_pattern=gen_bloc_params['token_pattern']
        )
        
        '''
        if( prev_doc != '' ):
            cur_prev_tf_mat = get_bloc_variant_tf_matrix(
                [{'text': prev_doc}, {'text': doc}], 
                ngram=gen_bloc_params['ngram'], 
                min_df=gen_bloc_params['min_df'], 
                tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
                keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
                set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
                top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
                bloc_variant=bloc_variant,
                token_pattern=gen_bloc_params['token_pattern']
            )
            cur_prev_tf_mat = conv_tf_matrix_to_json_compliant(cur_prev_tf_mat)
            dist = 1 - cosine_sim( [cur_prev_tf_mat['tf_matrix'][0]['tf_vector']], [cur_prev_tf_mat['tf_matrix'][1]['tf_vector']] )
        '''

        all_doc_words = []
        all_doc_pause_words = []
        for w in re.split(r'([□⚀⚁⚂⚃⚄⚅.()])', doc):
            w = w.strip()
            if( w == '' ):
                continue
            if( w in '□⚀⚁⚂⚃⚄⚅.' ):
                all_doc_pause_words.append(w)
            else:
                all_doc_words.append(w)

        
        #max_word_len = max([ len(w) for w in all_doc_words ])
        num_word_switches = count_word_switches(all_doc_words, fold_start_count=bloc_variant['fold_start_count'])
        unique_pauses = len([ True for p in tf_matrices['vocab'] if p in '□⚀⚁⚂⚃⚄⚅.' ])
        unique_words = len(tf_matrices['vocab']) - unique_pauses

        #normalize - start
        unique_words = 0 if len(all_doc_words) < 2 else unique_words/len(all_doc_words)
        #unique_pauses = 0 if len(all_doc_pause_words) < 2 else unique_pauses/len(all_doc_pause_words)
        num_word_switches = 0 if len(all_doc_words) < 2 else num_word_switches/(len(all_doc_words) - 1)
        #normalize - end

        '''
        print(tf_matrices['vocab'])
        print(all_doc_words, all_doc_pause_words)
        if( dist is not None ):
            print('dist:', round(dist, 5))
        print('unique_words:', unique_words)
        print('unique_pauses:', unique_pauses)
        #print('max_word_len:', max_word_len)
        print('num_word_switches:', num_word_switches)
        print()
        '''

        change_dynamics['change_dynamics'].append({
            #'window_dist': dist,
            #'max_word_len': max_word_len,
            #'unique_pauses': unique_pauses,
            'num_word_switches': num_word_switches,
            'unique_words': unique_words
        })
        for metric, value in change_dynamics['change_dynamics'][-1].items():
            if( value is None ):
                continue
            change_dynamics['avg_change_dynamics'].setdefault(metric, {'sum': 0, 'total': 0})
            change_dynamics['avg_change_dynamics'][metric]['sum'] += value
            change_dynamics['avg_change_dynamics'][metric]['total'] += 1

        prev_doc = '' if doc == '' else doc


    for metric, value in change_dynamics['avg_change_dynamics'].items():
        change_dynamics['avg_change_dynamics'][metric] = change_dynamics['avg_change_dynamics'][metric]['sum']/change_dynamics['avg_change_dynamics'][metric]['total']
        change_dynamics['change_dynamics_score'] += change_dynamics['avg_change_dynamics'][metric]

    change_dynamics['change_dynamics_score'] = change_dynamics['change_dynamics_score']/len(change_dynamics['avg_change_dynamics'])
    change_dynamics['change_dynamics_score'] = change_dynamics['change_dynamics_score']#*len(change_dynamics['change_dynamics'])

    return change_dynamics

def slide_time_window_over_bloc_tweets_v1(bloc_tweets, alphabet, window_size_seconds=3600, **kwargs):
    
    def cosine_dist(fst_vect, sec_vect):

        fst_comb_vect = []
        sec_comb_vect = []
        combined_vocab = set( fst_vect['vocab'] + sec_vect['vocab'] )
        
        for v in combined_vocab:
            
            try:
                indx = fst_vect['vocab'].index(v)
                fst_comb_vect.append( fst_vect['tf_vector'][indx] )
            except:
                fst_comb_vect.append( 0 )

            try:
                indx = sec_vect['vocab'].index(v)
                sec_comb_vect.append( sec_vect['tf_vector'][indx] )
            except:
                sec_comb_vect.append( 0 )
        
        return 1 - cosine_sim( [fst_comb_vect], [sec_comb_vect] )

    def get_doc_and_pause_words(vocab):
        
        all_doc_words = []
        all_doc_pause_words = []
        for w in vocab:

            w = w.strip()
            if( w == '' ):
                continue
            if( w in '□⚀⚁⚂⚃⚄⚅.' ):
                all_doc_pause_words.append(w)
            else:
                all_doc_words.append(w)

        return all_doc_words, all_doc_pause_words

    if( len(bloc_tweets) == 0 ):
        return {}

    gen_bloc_params = {
        'ngram': 1, 
        'min_df': 1, 
        'tf_matrix_norm': '', 
        'keep_tf_matrix': True,
        'set_top_ngrams': True, 
        'top_ngrams_add_all_docs': False,
        'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'
    }
    
    min_df_rate = kwargs.get('min_df_rate', 0.1)
    fold_start_count = kwargs.get('fold_start_count', 3)
    result = {'unique_words': [], 'unique_pause_words': [], 'unique_dates': set(), 'avg_cosine_dist': []}
    bloc_variant = {'type': 'folded_words', 'fold_start_count': fold_start_count, 'count_applies_to_all_char': False}

    time_bins = {}
    start_datetime = datetime.strptime(bloc_tweets[0]['created_at'], '%a %b %d %H:%M:%S %z %Y')
    terminal_datetime = datetime.strptime(bloc_tweets[-1]['created_at'], '%a %b %d %H:%M:%S %z %Y')


    #slide a time window from the bloc_tweets[0]['created_at'] to bloc_tweets[-1]['created_at']
    while( True ):

        end_datetime = start_datetime + timedelta(seconds=window_size_seconds)
        time_bins[ (start_datetime, end_datetime) ] = []
        start_datetime = end_datetime + timedelta(seconds=1)
        
        if( start_datetime > terminal_datetime ):
            time_bins[ (start_datetime, end_datetime) ] = []
            break


    #populate time bin with tweets that fall into time bins
    for i in range( len(bloc_tweets) ):
        tweet_time = datetime.strptime(bloc_tweets[i]['created_at'], '%a %b %d %H:%M:%S %z %Y')
        for date_rng, tweets in time_bins.items():
            start_datetime, end_datetime = date_rng
            if( tweet_time >= start_datetime and tweet_time <= end_datetime ):
                tweets.append(i)


    vocab_dist = {}
    #vect_history = []
    prev_tweet_indices = []
    for date_rng, tweet_indices in time_bins.items():

        if( len(tweet_indices) == 0 ):
            continue

        if( set(prev_tweet_indices) == set(tweet_indices) ):
            print('set(prev_tweet_indices) == set(tweet_indices)')
            continue
        

        doc = ''
        for indx in tweet_indices:
            if( 'bloc' not in bloc_tweets[indx] ):
                continue
            if( 'bloc_sequences_short' not in bloc_tweets[indx]['bloc'] ):
                continue
            doc = doc + bloc_tweets[indx]['bloc']['bloc_sequences_short'].get(alphabet, '')
        
        if( doc == '' ):
            continue


        tf_vect = get_bloc_variant_tf_matrix(
            [{'text': doc}], 
            ngram=gen_bloc_params['ngram'], 
            min_df=gen_bloc_params['min_df'], 
            tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
            keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
            set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
            top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
            bloc_variant=bloc_variant,
            token_pattern=gen_bloc_params['token_pattern']
        )
        
        if( len(tf_vect) == 0 ):
            continue


        tf_vect = conv_tf_matrix_to_json_compliant(tf_vect)
        #vect_history.append({ 'vocab': tf_vect['vocab'], 'tf_vector': tf_vect['tf_matrix'][0]['tf_vector'] })
        for v in tf_vect['vocab']:
            vocab_dist.setdefault(v, 0)
            vocab_dist[v] += 1
        
        all_doc_words, all_doc_pause_words = get_doc_and_pause_words(tf_vect['vocab'])
        result['unique_dates'].add(date_rng[0].strftime('%Y-%m-%d'))
        result['unique_dates'].add(date_rng[1].strftime('%Y-%m-%d'))
        result['unique_words'].append( len(all_doc_words) )
        result['unique_pause_words'].append( len(all_doc_pause_words) )

    
    for metric in ['unique_words', 'unique_pause_words']:
        count = len(result[metric])
        if( count == 0 ):
            result[metric] = {'mean': -1, 'pstdev': -1, 'count': -1}
        else:
            result[metric] = { 'mean': statistics.mean(result[metric]), 'pstdev': statistics.pstdev(result[metric]), 'count': count }


    #random.shuffle(vect_history)
    #result['avg_cosine_dist'] = 0 if len(vect_history) < 2 else statistics.mean([ cosine_dist(vect_history[i-1], vect_history[i]) for i in range(1, len(vect_history)) ])
    result['unique_dates'] = len(result['unique_dates'])
    
    #fmt vocab - start
    '''
    result['temporal_vocab_dist'] = {}

    for voc, freq in vocab_dist.items():
        df_rate = freq/result['unique_words']['count']
        if( df_rate >= min_df_rate ):
            result['temporal_vocab_dist'][voc] = {'doc_freq': freq, 'doc_rate': df_rate}

    result['temporal_vocab_dist'] = sorted( result['temporal_vocab_dist'].items(), key=lambda x: x[1]['doc_freq'], reverse=True )
    '''
    result['total_unique_words'] = len(vocab_dist)
    #fmt vocab - end

    return result

def slide_time_window_over_bloc_tweets_v2(bloc_tweets, bloc_doc, bloc_alphabet, window_size_seconds=3600, **kwargs):#1h: 3600, 1d: 86400
    
    def cosine_dist(fst_vect, sec_vect):

        fst_comb_vect = []
        sec_comb_vect = []
        combined_vocab = set( fst_vect['vocab'] + sec_vect['vocab'] )
        
        for v in combined_vocab:
            
            try:
                indx = fst_vect['vocab'].index(v)
                fst_comb_vect.append( fst_vect['tf_vector'][indx] )
            except:
                fst_comb_vect.append( 0 )

            try:
                indx = sec_vect['vocab'].index(v)
                sec_comb_vect.append( sec_vect['tf_vector'][indx] )
            except:
                sec_comb_vect.append( 0 )
        
        return 1 - cosine_sim( [fst_comb_vect], [sec_comb_vect] )

    def get_doc_and_pause_words(vocab):
        
        all_doc_words = []
        all_doc_pause_words = []
        for w in vocab:

            w = w.strip()
            if( w == '' ):
                continue
            if( w in '□⚀⚁⚂⚃⚄⚅.' ):
                all_doc_pause_words.append(w)
            else:
                all_doc_words.append(w)

        return all_doc_words, all_doc_pause_words

    #print('\nslide_time_window_over_bloc_tweets_v2()')
    if( len(bloc_tweets) == 0 ):
        return {}
    
    #print('\tbloc_tweets:', len(bloc_tweets))
    gen_bloc_params = {
        'ngram': 1, 
        'min_df': 1, 
        'tf_matrix_norm': '', 
        'keep_tf_matrix': True,
        'set_top_ngrams': True, 
        'top_ngrams_add_all_docs': False,
        'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'
    }
    
    result = {'avg_cosine_dist': {'sum': 0, 'count': 0}}
    #min_df_rate = kwargs.get('min_df_rate', 0.1)
    fold_start_count = kwargs.get('fold_start_count', 3)
    window_size_seconds = kwargs.get('window_size_seconds', 3600)

    
    bloc_variant = {'type': 'folded_words', 'fold_start_count': fold_start_count, 'count_applies_to_all_char': False}

    time_bins = {}
    start_datetime = datetime.strptime(bloc_tweets[0]['created_at'], '%a %b %d %H:%M:%S %z %Y')
    terminal_datetime = datetime.strptime(bloc_tweets[-1]['created_at'], '%a %b %d %H:%M:%S %z %Y')
    

    #slide a time window from the bloc_tweets[0]['created_at'] to bloc_tweets[-1]['created_at']
    while( True ):

        end_datetime = start_datetime + timedelta(seconds=window_size_seconds)
        time_bins[ (start_datetime, end_datetime) ] = []
        start_datetime = end_datetime + timedelta(seconds=1)
        
        if( start_datetime > terminal_datetime ):
            time_bins[ (start_datetime, end_datetime) ] = []
            break

    #print('\ttime_bins:', len(time_bins))
    #populate time bin with tweets that fall into time bins
    for i in range( len(bloc_tweets) ):
        tweet_time = datetime.strptime(bloc_tweets[i]['created_at'], '%a %b %d %H:%M:%S %z %Y')
        for date_rng, tweets in time_bins.items():
            start_datetime, end_datetime = date_rng
            if( tweet_time >= start_datetime and tweet_time <= end_datetime ):
                tweets.append(i)


    prev_doc = ''
    prev_tweet_indices = []
    for date_rng, tweet_indices in time_bins.items():

        if( len(tweet_indices) == 0 ):
            continue

        if( set(prev_tweet_indices) == set(tweet_indices) ):
            #print('set(prev_tweet_indices) == set(tweet_indices)')
            continue
        
        #print('date_rng:', date_rng)
        doc = ''
        for indx in tweet_indices:
            if( 'bloc' not in bloc_tweets[indx] ):
                continue
            if( 'bloc_sequences_short' not in bloc_tweets[indx]['bloc'] ):
                continue
            doc = doc + bloc_tweets[indx]['bloc']['bloc_sequences_short'].get(bloc_alphabet, '')
        
        if( doc == '' ):
            continue


        if( prev_doc != '' ):

            cur_prev_tf_mat = get_bloc_variant_tf_matrix(
                [{'text': prev_doc}, {'text': doc}], 
                ngram=gen_bloc_params['ngram'], 
                min_df=gen_bloc_params['min_df'], 
                tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
                keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
                set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
                top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
                bloc_variant=bloc_variant,
                token_pattern=gen_bloc_params['token_pattern']
            )

            cur_prev_tf_mat = conv_tf_matrix_to_json_compliant(cur_prev_tf_mat)
            dist = 1 - cosine_sim( [cur_prev_tf_mat['tf_matrix'][0]['tf_vector']], [cur_prev_tf_mat['tf_matrix'][1]['tf_vector']] )
            
            result['avg_cosine_dist']['sum'] += dist
            result['avg_cosine_dist']['count'] += 1

        prev_doc = doc

    result['avg_cosine_dist'] = -1 if result['avg_cosine_dist']['count'] == 0 else result['avg_cosine_dist']['sum']/result['avg_cosine_dist']['count']

    return result

def slide_time_window_over_bloc_tweets_v3(bloc_tweets, bloc_doc, bloc_alphabet, **kwargs):
    
    if( len(bloc_tweets) == 0 ):
        return {}

    print(f'\nslide_time_window_over_bloc_tweets_v3(): {bloc_alphabet}')

    time_bins = {}
    result = {'avg_cosine_dist': {'sum': 0, 'count': 0}}

    study_dates = kwargs.get('study_dates', [])
    fold_start_count = kwargs.get('fold_start_count', 3)
    study_dates_ranges = kwargs.get('study_dates_ranges', [])

    gen_bloc_params = {
        'ngram': 1, 
        'min_df': 1, 
        'tf_matrix_norm': '', 
        'keep_tf_matrix': True,
        'set_top_ngrams': True, 
        'top_ngrams_add_all_docs': False,
        'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'
    }
    
    result = {'avg_cosine_dist': {'sum': 0, 'count': 0}}
    bloc_variant = {'type': 'folded_words', 'fold_start_count': fold_start_count, 'count_applies_to_all_char': False}

    #bin tweets before (range 1) and after (range 2) study date
    for i in range( len(bloc_tweets) ):
        
        tweet_time = datetime.strptime(bloc_tweets[i]['created_at'], '%a %b %d %H:%M:%S %z %Y')
        for j in range(len(study_dates_ranges)):
            
            time_bins.setdefault( (study_dates_ranges[j][0], study_dates_ranges[j][1]), {'lhs_twt_indices': [], 'rhs_twt_indices': [], 'user': bloc_tweets[i]['user']} )
            
            #date range: study_dates_ranges[j][0]...study_dates[j]...study_dates_ranges[j][1]
            #range 1: [ study_dates_ranges[j][0], study_dates[j] ]
            #range 2: [ study_dates[j], study_dates_ranges[j][1] ]

            plain_tweet_time = tweet_time.strftime('%Y-%m-%d %H:%M:%S')
            plain_tweet_time = datetime.strptime(plain_tweet_time, '%Y-%m-%d %H:%M:%S')

            if( plain_tweet_time >= study_dates_ranges[j][0] and plain_tweet_time <= study_dates[j] ):
                #tweet is in range 1
                time_bins[(study_dates_ranges[j][0], study_dates_ranges[j][1])]['lhs_twt_indices'].append(i)

            elif( plain_tweet_time >= study_dates[j] and plain_tweet_time <= study_dates_ranges[j][1] ):
                #tweet is in range 2
                time_bins[(study_dates_ranges[j][0], study_dates_ranges[j][1])]['rhs_twt_indices'].append(i)
            

    #compute different 
    for date_rng, tweet_indices in time_bins.items():
        
        if( len(tweet_indices['lhs_twt_indices']) == 0 or len(tweet_indices['rhs_twt_indices']) == 0 ):
            continue

        docs = {'lhs_twt_indices': '', 'rhs_twt_indices': ''}
        for lhs_rhs in ['lhs_twt_indices', 'rhs_twt_indices']:
            for indx in tweet_indices[lhs_rhs]:
                if( 'bloc' not in bloc_tweets[indx] ):
                    continue
                if( 'bloc_sequences_short' not in bloc_tweets[indx]['bloc'] ):
                    continue
                docs[lhs_rhs] = docs[lhs_rhs] + bloc_tweets[indx]['bloc']['bloc_sequences_short'].get(bloc_alphabet, '')
        

        tf_mat = get_bloc_variant_tf_matrix(
            [{'text': docs['lhs_twt_indices']}, {'text': docs['rhs_twt_indices']}], 
            ngram=gen_bloc_params['ngram'], 
            min_df=gen_bloc_params['min_df'], 
            tf_matrix_norm=gen_bloc_params['tf_matrix_norm'], 
            keep_tf_matrix=gen_bloc_params['keep_tf_matrix'],
            set_top_ngrams=gen_bloc_params['set_top_ngrams'], 
            top_ngrams_add_all_docs=gen_bloc_params['top_ngrams_add_all_docs'],
            bloc_variant=bloc_variant,
            token_pattern=gen_bloc_params['token_pattern']
        )

        tf_mat = conv_tf_matrix_to_json_compliant(tf_mat)
        dist = 1 - cosine_sim( [tf_mat['tf_matrix'][0]['tf_vector']], [tf_mat['tf_matrix'][1]['tf_vector']] )

        result['avg_cosine_dist']['sum'] += dist
        result['avg_cosine_dist']['count'] += 1
        
        '''
        print('\t', date_rng)
        print('\t', tweet_indices['user']['screen_name'], len(tweet_indices['lhs_twt_indices']), len(tweet_indices['rhs_twt_indices']))
        print('\t', docs['lhs_twt_indices'])
        print('\t', docs['rhs_twt_indices'])
        print()
        '''
    
    result['avg_cosine_dist'] = -1 if result['avg_cosine_dist']['count'] == 0 else result['avg_cosine_dist']['sum']/result['avg_cosine_dist']['count']

    return result
