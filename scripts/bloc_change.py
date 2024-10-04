import logging
import argparse
import csv
import gzip
import json
import math
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
#import networkx as nx
import numpy as np
import os
#import seaborn as sns
import random
import requests
import statistics
import sys
import time

from copy import deepcopy
from collections import Counter
from datetime import datetime
from datetime import timedelta
from itertools import product

from bloc.generator import add_bloc_sequences
from bloc.subcommands import bloc_change_usr_self_cmp
from bloc.util import conv_tf_matrix_to_json_compliant
from bloc.util import dumpJsonToFile
from bloc.util import five_number_summary
from bloc.util import genericErrorInfo
from bloc.util import get_default_symbols
from bloc.util import getDictFromJson

from info_ops_tk.create_datasets import get_driver_per_day_tweets
from info_ops_tk.create_datasets import get_driver_post_date_dist
from info_ops_tk.create_datasets import get_driver_post_stats
from info_ops_tk.util import get_bloc_lite_twt_frm_full_twt
from info_ops_tk.util import parallelTask
from sumgram.sumgram import get_top_sumgrams

from itertools import combinations
from random import randint, shuffle
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from bloc_chg_dynamics import slide_time_window_over_bloc_tweets_v2
from bloc_chg_dynamics import slide_time_window_over_bloc_tweets_v3
from bloc_chg_dynamics import get_bloc_entropy
from bloc_chg_dynamics import get_vocab_and_doc_len
from bloc_chg_dynamics import get_bloc_vocab_doc_len_linear_model

#logging.basicConfig(format='', level=logging.INFO)
#logger = logging.getLogger(__name__)
plt.rcParams.update({"font.family": 'monospace'})

def get_generic_args():

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), description='BLOC change experiments')
    parser.add_argument('tweets_files', nargs='+', default='', help='Filename/Path containing tweets to process. If filename (see --tweets-path)')

    parser.add_argument('--add-pauses', action='store_true', help='BLOC generator (for words not bigram) use pause as separate words in vocabulary True/268e. False is default')
    parser.add_argument('--bloc-model', default='word', choices=['bigram', 'word'], help='BLOC tokenization method.')

    parser.add_argument('--no-merge', action='store_true', help='Do not merge dataset variants (e.g., "bot-A" and "bot-B") of sources into single class (e.g, "bot").')
    parser.add_argument('-o', '--outpath', default=os.getcwd() + '/Output/', help='Output path')
    parser.add_argument('-t', '--task', default='bot_human_cosine_sim_dist', choices=['bot_human_cosine_sim_dist', 'info_ops_study', 'midterm_2022_study', 'diversity_study', 'change_segments_study','change_ref_dates_study', 'gen_vocab_doc_len_model', 'cosine_sim_dist'], help='Task to run')
    parser.add_argument('--tweets-path', default='/scratch/anwala/IU/BLOC/bloc-intro-paper/bot-detection/retraining_data/', help='The path to extract tweets for --tweets-files.')
    #parser.add_argument('--tweets-path-info-ops', default='/scratch/anwala/IU/BLOC/InfoOps/YYYY_MM', help='Drivers: the path to extract tweets for --tweets-files.')
    #/Users/alexander_nwala/data/coordination-detection-info-ops/YYYY_MM

    #for midterm 2022
    parser.add_argument('--start-time', default='', help='For midterm 2022 tweets, start time created_at datetimes (YYYY-MM-DDTHH:MM:SS) tweets to collect')
    parser.add_argument('--end-time', default='', help='For midterm 2022 tweets, end time created_at datetimes (YYYY-MM-DDTHH:MM:SS) tweets to collect')

    #max parameters
    parser.add_argument('-m', '--max-tweets', type=int, default=-1, help='Maximum number of tweets per user to consider')
    parser.add_argument('-n', '--min-tweets', type=int, default=20, help='Mininum number of tweets per user to consider')
    parser.add_argument('-u', '--max-users', type=int, default=-1, help='Maximum number of users per class to process')
    
    #BLOC parameters
    parser.add_argument('--bc-blank-mark', type=int, default=60, help="add_bloc_sequences()'s blank_mark")
    parser.add_argument('--bc-bloc-alphabets', nargs='+', default=['action', 'content_syntactic'], choices=['action', 'content_syntactic', 'content_semantic_entity', 'content_syntactic_with_pauses', 'change', 'action_content_syntactic'], help='add_bloc_sequences()\'s BLOC alphabets to extract')
    parser.add_argument('--bc-days-segment-count', type=int, default=-1, help="add_bloc_sequences()'s days_segment_count, if > 0, segmentation_type is set to day_of_year_bin")
    parser.add_argument('--bc-fold-start-count', type=int, default=10, help="add_bloc_sequences()'s change alphabet fold_start_count")
    parser.add_argument('--bc-gen-rt-content', action='store_true', help="add_bloc_sequences()'s gen_rt_content")
    parser.add_argument('--bc-keep-bloc-segments', action='store_true', help="add_bloc_sequences()'s keep_bloc_segments")
    parser.add_argument('--bc-keep-tweets', action='store_true', help="add_bloc_sequences()'s keep_tweets")
    parser.add_argument('--bc-minute-mark', type=int, default=5, help="add_bloc_sequences()'s minute_mark")
    parser.add_argument('--bc-segmentation-type', default='week_number', choices=['week_number', 'day_of_year', 'yyyy-mm-dd', 'segment_on_pauses'], help="add_bloc_sequences()'s segmentation_type")
    parser.add_argument('--bc-segment-on-pauses', type=int, default=3600, help='Segment tweets (BLOC strings) using pauses >= value (value in seconds)')
    parser.add_argument('--bc-sort-action-words', action='store_true', help="add_bloc_sequences()'s sort_action_words")
    parser.add_argument('--bc-tweet-order', default='sorted', choices=['sorted', 'reverse'], help="add_bloc_sequences()'s tweet_order")
    
    #change/vector parameters
    parser.add_argument('--account-class', default='', help='Class labels (e.g., bot or cyborgs or humans) of accounts')
    parser.add_argument('--account-src', default='', help='Origin of accounts')


    #For --task=change_segments_study
    parser.add_argument('--window-size-seconds', type=int, default=3600, help='For --task change-segments-study, segment tweets (BLOC strings) into bins of tweets (BLOC strings) generated within value-specified')
    #For --task=change_ref_dates_study 
    parser.add_argument('--study-dates', default='', help='For --task=change_ref_dates_study, Dates (comma-separaed YYYY-MM-DD) for which to compute change around.')
    parser.add_argument('--study-dates-seconds-offset', type=int, default=604800, help='For --task=change_ref_dates_study, Time (seconds) offset to add to study_dates to produce range (start date -- study_date -- end date) of dates to compute change.')
    
    #action
    parser.add_argument('--action-change-mean', type=float, help='For BLOC action: Empirical mean cosine similarity across BLOC segments.')
    parser.add_argument('--action-change-stddev', type=float, help='For BLOC action: Empirical standard deviation across BLOC segments.')
    parser.add_argument('--action-change-zscore-threshold', type=float, default=-1.3, help='For BLOC action: Number of standard deviations (z-score) a similarity value has to exceed to be considered significant.')
    #content
    parser.add_argument('--content-syntactic-change-mean', type=float, help='For BLOC content_syntactic: Empirical mean cosine similarity across BLOC segments.')
    parser.add_argument('--content-syntactic-change-stddev', type=float, help='For BLOC content_syntactic: Empirical standard deviation across BLOC segments.')
    parser.add_argument('--content-syntactic-change-zscore-threshold', type=float, default=-1.3, help='For BLOC content_syntactic: Number of standard deviations (z-score) a similarity value has to exceed to be considered significant.')
    
    parser.add_argument('--fold-start-count', type=int, default=4, help='For word models, value marks maximum threshold words must reach before truncation.')
    parser.add_argument('--keep-tf-matrix', action='store_true', help='Keep or do not keep tf_matrix. Default is False.')
    parser.add_argument('--ngram', type=int, default=1, help='n-gram for tokenization BLOC string.')
    parser.add_argument('--tf-matrix-norm', default='', choices=['', 'l1', 'max'], help='Norm to use for normalizing TF matrix (see sklearn.preprocessing.normalize(). Blank means tf_matrix_normalized is not needed.')
    
    return parser

def get_bloc_for_tweets(tweets_files, tweets_path, gen_bloc_params, **kwargs):

    def get_user_id_class_map(f):

        user_id_class_map = {}
        all_classes = set()

        try:
            with open(f) as fd:
                
                rd = csv.reader(fd, delimiter='\t')
                for user_id, user_class in rd:
                    user_id_class_map[user_id] = user_class
                    all_classes.add(user_class)
        except:
            genericErrorInfo()

        return user_id_class_map, all_classes

    def is_all_class_full(all_users, max_users):

        for c in all_users:
            if( len(all_users[c]) != max_users ):
                return False

        return True

    print('\nget_bloc_for_tweets():')
    max_users = kwargs.get('max_users', 300)
    max_tweets = kwargs.get('max_tweets', -1)
    min_tweets = kwargs.get('min_tweets', 20)

    user_class = ''
    payload = {}
    all_bloc_symbols = get_default_symbols()
    
    for f in tweets_files:

        f = tweets_path + f + '/tweets.jsons.gz' if f.find('tweets') == -1 else f
        cf = '/'.join( f.split('/')[:-1] ) + '/userIds.txt'
        src = f.split('/')[-2]
        
        print('tweets_path:', tweets_path)
        print('tweets file:', f)
        print('src:', src)
        
        if( os.path.exists(f) is False ):
            print('\ttweets file doesn\'t exist, returning')
            continue

        user_id_class_map, all_classes = get_user_id_class_map( cf )
        print('all_classes:', all_classes)
        #print(set(list(user_id_class_map.values())))
        #sys.exit(0)

        if( len(user_id_class_map) == 0 ):
            print('\tuser_id_class_map is empty, will use human class')
            all_classes = ['human']
            user_class = 'human'
            #continue
        
        users_tweets = {}
        for c in all_classes:
            users_tweets[c] = []

        encoding = None
        if( src.find('stock') != -1 ):
            encoding = 'windows-1252'

        
        with gzip.open(f, 'rt', encoding=encoding) as infile:
            for line in infile:
                try:

                    line = line.split('\t')
                    '''
                        line[0]: user_id
                        line[1]: tweets
                    '''
                    if( len(line) != 2 ):
                        continue
                    
                    #user_class = user_id_class_map.get(line[0], '') if user_class == '' else user_class
                    user_class = user_id_class_map[ line[0] ]

                    if( user_class == '' ):
                        continue
                    
                    tweets = getDictFromJson( line[1] )
                    if( len(tweets) < min_tweets ):
                        continue
                    
                    if( is_all_class_full(users_tweets, max_users) ):
                        print('full class breaking')
                        break

                    count = len( users_tweets[user_class] )
                    if( count == max_users ):
                        continue

                    if( isinstance(max_tweets, tuple) ):
                        max_range = len(tweets) if max_tweets[1] > len(tweets) else max_tweets[1]
                        tweets = tweets[: randint(max_tweets[0], max_range) ]
                    elif( max_tweets > 0 ):
                        tweets = tweets[:max_tweets]
                    
                    bloc_payload = add_bloc_sequences(tweets, all_bloc_symbols=all_bloc_symbols, **gen_bloc_params)
                    
                    if( count % 100 == 0 ):
                        print( f'\t{count} of {max_users} {user_class} users' )

                    users_tweets[user_class].append( bloc_payload )
                except:
                    genericErrorInfo()


        payload[src] = users_tweets

    return payload

def obsolete_calc_change_rate(sim_vals, change_mean, change_stddev, change_zscore_threshold):

    sim_count = len(sim_vals)
    if( sim_count == 0 or change_mean is None and change_stddev is None or change_zscore_threshold is None ):
        return None

    change_rate = 0
    for sm in sim_vals:

        zscore_sim = (sm - change_mean)/change_stddev
        if( abs(zscore_sim) < change_zscore_threshold ):
            continue

        change_rate += 1
        
    return change_rate/sim_count

def draw_ccdfdist(dist, color, **kwargs):
        
    title = kwargs.pop('title', '')
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')

    ax = sns.ecdfplot(data=dist, complementary=True, color=color, **kwargs)
    
    if( title != '' ):
        ax.set_title( title )

    if( xlabel != '' ):
        ax.set_xlabel( xlabel )

    if( ylabel != '' ):
        ax.set_ylabel( ylabel )


def print_tweet_messages_for_segment(tweets, segmentation_type, seg_num):
    
    for t in tweets:
        if( t['bloc'][segmentation_type] == seg_num ):
            full_text = ' '.join(t['full_text'].strip().split())
            print( '\t{}: {}'.format(t['tweet_time'], full_text) )

def study_info_ops_drivers_change_context_v0(bloc_collection, compute_change_alphabet, segmentation_type, class_to_study=['driver']):

    def print_change_report(user_bloc, segmentation_type, alph):

        #if( alph == 'content_syntactic' ):
        #    return

        for sm in user_bloc['user_change_report']['self_sim'].get(alph, []):

            changed_flag = sm.get('changed', False)

            if( changed_flag is False ):
                continue

            fst_key = sm['fst_doc_seg_id']
            sec_key = sm['sec_doc_seg_id']

            fst_doc = user_bloc['bloc_segments']['segments'][fst_key][alph]
            sec_doc = user_bloc['bloc_segments']['segments'][sec_key][alph]

            print(f'changed: {changed_flag}')
            print(f'{fst_key}: {fst_doc}')
            print('skipping tweet content')
            #print_tweet_messages_for_segment( user_bloc['tweets'], segmentation_type, seg_num=fst_key )
            print(f'{sec_key}: {sec_doc}')
            print('skipping tweet content')
            #print_tweet_messages_for_segment( user_bloc['tweets'], segmentation_type, seg_num=sec_key )
            
            if( alph == 'action' ):
                print('content')
                fst_doc = user_bloc['bloc_segments']['segments'][fst_key]['content_syntactic']
                sec_doc = user_bloc['bloc_segments']['segments'][sec_key]['content_syntactic']
                print(f'{fst_key}: {fst_doc}')
                print(f'{sec_key}: {sec_doc}')
                
            print()

    all_drivers = [t['screen_name'] for t in bloc_collection]
    print('\nstudy_info_ops_drivers_change_context()')
    print('all drivers:')
    print(all_drivers)
    for i in range(len(bloc_collection)):

        user_bloc = bloc_collection[i]
        if( user_bloc['class'] not in class_to_study ):
            continue
        
        if( 'user_change_report' not in user_bloc ):
            continue

        change_rate = user_bloc['user_change_report']['change_rates'].get(compute_change_alphabet, None)
        change_profile = user_bloc['user_change_report']['avg_change_profile'].get(compute_change_alphabet, {})
        if( change_rate is None ):
            continue        

        change_study_threshold = 0.45 if compute_change_alphabet == 'action' else 0
        if( change_rate >= change_study_threshold ):
            print('**change**')
            print(user_bloc['screen_name'])
            segments = list(user_bloc['bloc_segments']['segments'].keys())
            segments.sort()
            print('class/segment count/change_rate:', user_bloc['bloc_segments']['segment_count'], change_rate)
            print(change_profile)
            #print('change_dates:', change_dates)

            print('segments:', segments)
            print( user_bloc['bloc'][compute_change_alphabet] )
            print( user_bloc['created_at_utc'] )
            #print(json.dumps(user_bloc, ensure_ascii=False))
            print_change_report( user_bloc, segmentation_type, compute_change_alphabet )
            #sys.exit(0)
            
            print()
            print()

        #print(json.dumps(user_bloc, ensure_ascii=False))
        #sys.exit(0)


def study_info_ops_drivers_change_context_v1(u_bloc, bloc_alphabet, out):
    
    params = {
        'top_sumgram_count': 20,
        'add_stopwords': ['rt', 'http', 'https', 'amp', 't.co'],
        'no_rank_sentences': True,
        'title': 'Top sumgrams'
    }
    segments = {}
    fst_sumgram_doc_lst = []
    sec_sumgram_doc_lst = []

    all_user_mentions = []
    all_user_hashtags = []

    for i in range( len(u_bloc['tweets']) ):
        
        t = u_bloc['tweets'][i]
        seg_num = t['bloc'][ t['bloc']['segmentation_type'] ]

        for m in t['entities']['user_mentions']:
            all_user_mentions.append(m['screen_name'].lower())

        for h in t['entities']['hashtags']:
            all_user_hashtags.append(h['text'].lower())
        
        segments.setdefault(seg_num, [])
        segments[seg_num].append(i)

        fst_sumgram_doc_lst.append({ 'id': len(fst_sumgram_doc_lst), 'text': t['full_text'] })
        sec_sumgram_doc_lst.append({ 'id': len(sec_sumgram_doc_lst), 'text': t['full_text'] })


    if( len(fst_sumgram_doc_lst) > 0 ):
        sumgrams = get_top_sumgrams(fst_sumgram_doc_lst, n=1, params=params)
        out('First: top sumgrams\n')
        for i in range(len(sumgrams.get('top_sumgrams', []))):
            t = sumgrams['top_sumgrams'][i]
            out('\t{}: {}, {:.4f}\n'.format(i+1, t['ngram'], t['term_rate']))

        sumgrams = get_top_sumgrams(sec_sumgram_doc_lst, n=2, params=params)
        out('Second: top sumgrams\n')
        for i in range(len(sumgrams.get('top_sumgrams', []))):
            t = sumgrams['top_sumgrams'][i]
            out('\t{}: {}, {:.4f}\n'.format(i+1, t['ngram'], t['term_rate']))
        out('\n')

    
    total_mentions = len(all_user_mentions)
    all_user_mentions  = Counter(all_user_mentions)
    all_user_mentions = sorted(all_user_mentions.items(), key=lambda k: k[1], reverse=True )[:20]
    if( total_mentions > 0 ):
        out('Top mentions\n')
        for i in range(len(all_user_mentions)):
            m = all_user_mentions[i]
            out( '\t{:>3}. {:>4}, {:5.4f}: {}\n'.format(i+1, m[1], m[1]/total_mentions, m[0]) )

    total_hashtags = len(all_user_hashtags)
    all_user_hashtags  = Counter(all_user_hashtags)
    all_user_hashtags = sorted(all_user_hashtags.items(), key=lambda k: k[1], reverse=True )[:20]
    
    if( total_hashtags > 0 ):
        out('Top hashtags\n')
        for i in range(len(all_user_hashtags)):
            h = all_user_hashtags[i]
            out( '\t{:>3}. {:>4}, {:5.4f}: {}\n'.format(i+1, h[1], h[1]/total_hashtags, h[0]) )

    for seg, tweet_indices in segments.items():

        bloc_doc = ''
        tweets_doc = ''

        for indx in tweet_indices:
            twt = u_bloc['tweets'][indx]
            bloc_doc = bloc_doc + twt['bloc']['bloc_sequences_short'][bloc_alphabet]
            rt_uid = twt.get('retweeted_status', {}).get('user', {}).get('id', None)
            tweets_doc = tweets_doc + '\t\t{}: {} tid: {}, rt_uid: {}\n'.format(twt['tweet_time'], ' '.join(twt['full_text'].split()), twt['id'], rt_uid )

        out(f'\t{bloc_doc}\n')
        out(f'{tweets_doc}\n')
        out('\n')

def study_info_ops_drivers_cochange(drivers_change_dates, bloc_collection, store_path, legend_title, **kwargs):
    
    def draw_info_ops_network(G, color='red'):
        
        pos = nx.kamada_kawai_layout(G)  #, seed=7 positions for all nodes - seed for reproducibility
        
        edge_weights = []
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        for ky, val in edge_labels.items():
            edge_labels[ky] = '{:.2f}'.format(val)
            edge_weights.append(val)

        # nodes
        node_sizes = [v for v in dict(G.degree).values()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=color, alpha=0.3, linewidths=0)

        # edges
        #edge_weights = 0.5
        nx.draw_networkx_edges(G, pos, alpha=0.05, width=edge_weights)

        # node labels
        #nx.draw_networkx_labels(G, pos, font_size=5)

        #remove edge labels
        print('Not drawing edge labels')
        edge_labels = {}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=4, bbox=dict(alpha=0))

        #ax = plt.gca()
        #plt.axis("off")
        #plt.tight_layout()

    count_of_drivers_with_change = len(drivers_change_dates)
    indices = list(range(count_of_drivers_with_change))     
    pairs = combinations(indices, 2)

    G = nx.Graph()
    for fst_u_indx, sec_u_indx in pairs:
        
        intersection = drivers_change_dates[fst_u_indx]['change_dates'] & drivers_change_dates[sec_u_indx]['change_dates']
        union = drivers_change_dates[fst_u_indx]['change_dates'] | drivers_change_dates[sec_u_indx]['change_dates']
        
        fst_chng_dates_len = len(drivers_change_dates[fst_u_indx]['change_dates'])
        sec_chng_dates_len = len(drivers_change_dates[sec_u_indx]['change_dates'])

        min_size = min(fst_chng_dates_len, sec_chng_dates_len)
        overlap = len(intersection)/min_size
        jaccard = len(intersection)/len(union)

        fst_bloc_col_indx = drivers_change_dates[fst_u_indx]['bloc_col_indx']
        sec_bloc_col_indx = drivers_change_dates[sec_u_indx]['bloc_col_indx']

        #if( overlap > 0 ):
        if( jaccard > 0.2 ):

            '''
            if( kwargs['agent'] == 'driver' ):
                print(drivers_change_dates[fst_u_indx]['change_dates'])
                print(drivers_change_dates[sec_u_indx]['change_dates'])
                #print(drivers_change_dates[fst_u_indx]['change_bloc'])
                #print(drivers_change_dates[sec_u_indx]['change_bloc'])
                #print('j:', jaccard, 'o:', overlap)
                #print(bloc_collection[fst_bloc_col_indx].keys())
                print()
                #sys.exit(0)
            '''

            fst_node = '{}\n(cd: {}, cr: {:.2f})'.format( bloc_collection[fst_bloc_col_indx]['screen_name'], fst_chng_dates_len, drivers_change_dates[fst_u_indx]['change_rate'] )
            sec_node = '{}\n(cd: {}, cr: {:.2f})'.format( bloc_collection[sec_bloc_col_indx]['screen_name'], sec_chng_dates_len, drivers_change_dates[sec_u_indx]['change_rate'] )
            G.add_edge(fst_node, sec_node, weight=overlap)

    campaign = kwargs['info_ops_dets']
    color = 'red' if kwargs['agent'] == 'driver' else 'blue'
    if( len(G.nodes()) != 0 ): 
        draw_info_ops_network(G, color=color)

    all_change_dates = list(campaign['all_change_dates'])
    all_change_dates.sort()
    all_change_dates = '' if len(all_change_dates) == 0 else f'{all_change_dates[0]} -- {all_change_dates[-1]} ({len(all_change_dates)} days)'

    info_ops_stats = 'total_{}: {:,} (changed: {:,}; {})\nall_change_dates: {}\ncampaign_date_range: {}'.format(kwargs['agent'], campaign['total'], count_of_drivers_with_change, '{:.2f}'.format(count_of_drivers_with_change/campaign['total']), all_change_dates, campaign['campaign_timerange'])
    handles = [ mlines.Line2D([], [], color=color, label='{}'.format(info_ops_stats)) ]
    out_file = '{}{}_{}_cochange_graph.png'.format(store_path, legend_title, kwargs['agent'])

    plt.legend( handles=handles, title=legend_title, fontsize=5 )
    plt.savefig(out_file, dpi=300)
    print('wrote:', out_file)
    plt.clf()


def calc_cosine_sim_dist(bloc_collection, bloc_model, args, legend_title='', compute_change_alphabet='action', **kwargs):

    def draw_change_dist(chrt_dist, styles, chart_type, store_path, legend_title='', stat_sig='', **kwargs ):
        
        print('\ndraw_change_dist()')
        if( len(chrt_dist) == 0 ):
            print('len(chrt_dist) == 0, returning')
            return

        counter = 0
        lg_lines = []

        all_classes = list(chrt_dist.keys())
        all_classes.sort()
        dist_slug = '_'.join(all_classes)
        slug_prefix = '' if legend_title == '' else f'{legend_title}_'
        
        #driver_count = ''
        compute_change_alphabet = kwargs.get('compute_change_alphabet', '')

        for usr_class in all_classes:

            usr_class_raw_dist_outfile = f'{store_path}{compute_change_alphabet}_{slug_prefix}{dist_slug}_{chart_type}_{usr_class}.txt'
            #'''
            outfile = open(usr_class_raw_dist_outfile, 'w')
            for v in chrt_dist[usr_class]:
                outfile.write(f'{v}\n')

            print(f'writing: {usr_class_raw_dist_outfile}')
            outfile.close()
            #'''
            #print(f'skipped writing: {usr_class_raw_dist_outfile}')
            
            draw_ccdfdist( chrt_dist[usr_class], color=styles[counter]['color'], linestyle=styles[counter]['linestyle'], xlabel=styles[counter].get('xlabel', ''), ylabel=styles[counter].get('ylabel', ''), title=styles[counter].get('title', '') + stat_sig )
            lg_lines.append( mlines.Line2D([], [], color=styles[counter]['color'], label='{} ({:,})'.format(usr_class.capitalize(), len(chrt_dist[usr_class])), linestyle=styles[counter]['linestyle']) )
            #driver_count = ' ({:,}) '.format(len(chrt_dist[usr_class])) if usr_class == 'driver' else ' '
            counter += 1
            

        if( len(lg_lines) != 0 ):
            
            plt.legend( handles=lg_lines, title=legend_title )#prop={"size": 14},
            plt.savefig(f'{store_path}{slug_prefix}{chart_type}_{dist_slug}.png', dpi=300)
            plt.clf()
            print(f'saved {store_path}{slug_prefix}{chart_type}_{dist_slug}.png')


        '''
        drivers_all_dates_change_occurred = kwargs.get('drivers_all_dates_change_occurred', [])
        if( len(drivers_all_dates_change_occurred) != 0 ):
            drivers_all_dates_change_occurred = sorted( drivers_all_dates_change_occurred.items(), key=lambda x: x[0] )
            dataset = [ driver_count[1] for driver_count in drivers_all_dates_change_occurred ]
        
            plt.plot(dataset, '-r', linewidth=0.7)
            plt.title(f'Distribution of number of drivers{driver_count}changing over time.')
            plt.ylabel('Number of drivers')
            plt.xlabel('Days')
            plt.savefig(f'{store_path}{slug_prefix}{dist_slug}_driver_change_timeline.png', dpi=300)
            plt.clf()
        '''
            
    def draw_change_profile_dist(chrt_dist, styles, chart_type, store_path, legend_title='', stat_sig='', **kwargs ):
        
        if( len(chrt_dist) == 0 ):
            return

        draw_plot_flag = kwargs.get('draw_plot', False)
        lg_lines = kwargs.get('lg_lines', [])
        counter = 0

        all_classes = list(chrt_dist.keys())
        all_classes.sort()
        dist_slug = '_'.join(all_classes)
        slug_prefix = '' if legend_title == '' else f'{legend_title}_'
        
        driver_count = ''

        for usr_class in all_classes:
            
            draw_ccdfdist( chrt_dist[usr_class], color=styles[counter]['color'], linestyle=styles[counter]['linestyle'], xlabel=styles[counter].get('xlabel', ''), ylabel=styles[counter].get('ylabel', ''), title=styles[counter].get('title', '') + stat_sig )
            lg_lines.append( mlines.Line2D([], [], color=styles[counter]['color'], label='{}: {}'.format(styles[counter]['change_profile'], usr_class.capitalize()), linestyle=styles[counter]['linestyle']) )
            driver_count = ' ({:,}) '.format(len(chrt_dist[usr_class])) if usr_class == 'driver' else ' '
            counter += 1


        if( draw_plot_flag ):
            plt.legend( handles=lg_lines, title=legend_title )#prop={"size": 14},
            plt.savefig(f'{store_path}{slug_prefix}{chart_type}_{dist_slug}.png', dpi=300)
            plt.clf()
            print(f'saved {store_path}{slug_prefix}{chart_type}_{dist_slug}.png')

    def test_for_stats_sig(chrt_dist):

        stat_sig = ''
        pair_dist = [ chrt_dist[usr_class] for usr_class in chrt_dist ]
        
        if( len(pair_dist) == 2 ):
            try:
                ks_stat = ks_2samp(pair_dist[0], pair_dist[1])
                stat_sig = '\n(KS-test, p < 0.01)' if ks_stat.pvalue < 0.01 else ''
                stat_sig = '\n(KS-test, p < 0.05)' if ks_stat.pvalue < 0.05 and stat_sig == '' else stat_sig
            except:
                genericErrorInfo()

        return stat_sig

    def explore_change_dist(change_dist):

        for usr_class, class_dist in change_dist.items():
            for user_change in class_dist:
                print(user_change)
                break
            break

    def get_dates_change_occurred(change_dist, user_bloc, bloc_alph):

        change_dates = []
        change_bloc = []
        for cd in change_dist:
            
            if( 'changed' not in cd ):
                continue

            fst_seg_key = cd['fst_doc_seg_id']
            sec_seg_key = cd['sec_doc_seg_id']

            change_dates += list(user_bloc['bloc_segments']['segments_details'][fst_seg_key]['local_dates'].keys()) + list(user_bloc['bloc_segments']['segments_details'][sec_seg_key]['local_dates'].keys())
            change_bloc.append( [user_bloc['bloc_segments']['segments'][fst_seg_key][bloc_alph], user_bloc['bloc_segments']['segments'][sec_seg_key][bloc_alph]] )
        
        change_dates = set(change_dates)
        return {
            'change_dates': change_dates,
            'change_bloc': change_bloc
        }

    segmentation_type = kwargs.get('segmentation_type', '')
    print('\ncalc_cosine_sim_dist():')
    print(f'\nrun_task: {args.task}')
    print(f'\tbloc alphabets: {args.bc_bloc_alphabets}')
    print(f'\tword_token_pattern:', bloc_model['token_pattern'])
    print(f'\tcompute_change_alphabet: {compute_change_alphabet}')
    print(f'\tsegmentation_type: {segmentation_type}')

    sim_dist = {}
    pause_profile_dist = {}
    word_profile_dist = {}
    activity_profile_dist = {}

    change_dist = {}
    styles = [
        {'color': 'green', 'linestyle': 'dotted'},
        {'color': 'red', 'linestyle': '-', 'xlabel': 'Cosine similarity', 'ylabel': 'CCDF', 'title': f'Distribution of cosine similarity between\nBLOC {compute_change_alphabet} words (adjacent weeks).'}
    ]

    info_ops_dets = {
        'driver': {'total': 0, 'all_change_dates': set(), 'campaign_timerange': kwargs.get('campaign_timerange', ''), 'dates_change_occurred': [], 'dates_change_dist': {}, 'dates_change_dist_total': 0},
        'control': {'total': 0, 'all_change_dates': set(), 'campaign_timerange': kwargs.get('campaign_timerange', ''), 'dates_change_occurred': [], 'dates_change_dist': {}, 'dates_change_dist_total': 0},
        'human': {'total': 0, 'all_change_dates': set(), 'campaign_timerange': kwargs.get('campaign_timerange', ''), 'dates_change_occurred': [], 'dates_change_dist': {}, 'dates_change_dist_total': 0},
        'bot': {'total': 0, 'all_change_dates': set(), 'campaign_timerange': kwargs.get('campaign_timerange', ''), 'dates_change_occurred': [], 'dates_change_dist': {}, 'dates_change_dist_total': 0}
    }
    
    for i in range(len(bloc_collection)):
        
        user_bloc = bloc_collection[i]
        user_change_report = bloc_change_usr_self_cmp(user_bloc, bloc_model, bloc_model['bloc_alphabets'], change_mean=args.change_mean, change_stddev=args.change_stddev, change_zscore_threshold=args.change_zscore_threshold)
        
        if( len(user_change_report['self_sim'].get(compute_change_alphabet, [])) == 0 ):
            continue
        
        user_bloc['user_change_report'] = user_change_report


        
        '''
        if( len(user_change_report['self_sim'][compute_change_alphabet]) > 3 and [1 for sm in user_change_report['self_sim'][compute_change_alphabet] if 'changed' in sm].count(1) != 0 ):
            dumpJsonToFile('tmp_bloc.json', user_bloc)
            dumpJsonToFile('tmp_change.json', user_change_report)
            sys.exit(0)
        '''

        #co-change graph - start
        '''
        user_class = user_bloc['class']

        #if( user_class in ['driver', 'control'] ):
        info_ops_dets[user_class]['total'] += 1
        change_rp = get_dates_change_occurred( user_change_report['self_sim'][compute_change_alphabet], user_bloc, compute_change_alphabet )

        #change_rp example: {'change_dates': {'2016-05-07', '2016-05-16', '2016-05-08', '2017-02-19', '2016-05-15', '2017-02-20'}, 'change_bloc': [['T⚀T⚁rr⚁r□rrr⚀rrrr⚁rrr', '⚃r'], ['⚃r', '⚁Trr⚁Trr⚁T'], ['⚁Trr⚁Trr⚁T', '⚄rr'], ['⚄rr', '⚁rrrrrr⚁r□r□rrrr⚁rrrrrrrrr□r□r⚀rr□r⚀r□r⚀r⚀rr⚁rrrrr⚀rrr⚀rr⚁rrr□rrr⚀rrr□r⚀r□rr']]}

        if( len(change_rp['change_dates']) != 0 ):
            info_ops_dets[user_class]['dates_change_occurred'].append({ 'bloc_col_indx': i, 'change_dates': change_rp['change_dates'], 'change_rate': user_change_report['change_rates'][compute_change_alphabet], 'change_bloc': change_rp['change_bloc'] })
            info_ops_dets[user_class]['all_change_dates'] = info_ops_dets[user_class]['all_change_dates'] | change_rp['change_dates']

            for chng_yyyy_mm_dd in change_rp['change_dates']:
                info_ops_dets[user_class]['dates_change_dist'].setdefault(chng_yyyy_mm_dd, 0)
                info_ops_dets[user_class]['dates_change_dist'][chng_yyyy_mm_dd] += 1
                info_ops_dets[user_class]['dates_change_dist_total'] += 1
        '''
        #co-change graph - end

        sim_vals = [ s['sim'] for s in user_change_report['self_sim'][compute_change_alphabet] ]
        usr_class = user_bloc['class']#user_bloc['src']
        
        sim_dist.setdefault(usr_class, [])
        sim_dist[usr_class] += sim_vals

        change_rate = user_change_report['change_rates'][compute_change_alphabet]
        if( change_rate is not None ):
            change_dist.setdefault(usr_class, [])
            change_dist[usr_class].append(change_rate)

            pause_profile_dist.setdefault(usr_class, [])
            pause_profile_dist[usr_class] += [ s['change_profile']['pause'] for s in user_change_report['self_sim'][compute_change_alphabet] ]

            word_profile_dist.setdefault(usr_class, [])
            word_profile_dist[usr_class] += [ s['change_profile']['word'] for s in user_change_report['self_sim'][compute_change_alphabet] ]

            activity_profile_dist.setdefault(usr_class, [])
            activity_profile_dist[usr_class] += [ s['change_profile']['activity'] for s in user_change_report['self_sim'][compute_change_alphabet] ]
        

    os.makedirs( f'./empirical-dists/{compute_change_alphabet}', exist_ok=True )
    os.makedirs( f'./change-dists/{compute_change_alphabet}', exist_ok=True )

    if( [args.change_mean, args.change_stddev].count(None) == 2 ):
        
        print('\tEmpirical stats:')
        
        for usr_class, class_dist in sim_dist.items():
            sum_stats = five_number_summary(sim_dist[usr_class])
            sum_stats['user_class'] = usr_class

            dumpJsonToFile(f'./empirical-dists/{compute_change_alphabet}/emp_cosine_sim_dist_{usr_class}.json', sum_stats )
            print(f'\twrote ./empirical-dists/{compute_change_alphabet}/emp_cosine_sim_dist_{usr_class}.json')
        
        stat_sig = test_for_stats_sig(sim_dist)
        draw_change_dist(sim_dist, styles, 'cosine_sim_dist', store_path=f'./empirical-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig)
        print()
        
    else:

        all_classes = len(change_dist.keys())
        style_indx = -1 if all_classes > 1 else 0

        styles[style_indx]['xlabel'] = 'Change rate'
        styles[style_indx]['ylabel'] = 'CCDF'
        styles[style_indx]['title'] = f'Distribution of BLOC {compute_change_alphabet} change rate.'
        stat_sig = test_for_stats_sig(change_dist)
        
        draw_change_dist(change_dist, styles, 'change_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, compute_change_alphabet=compute_change_alphabet)
        print()

        styles = [
            {'color': 'red', 'linestyle': '-', 'change_profile': 'p'},
            {'color': 'red', 'linestyle': 'dotted', 'change_profile': 'p'},
            {'color': 'green', 'linestyle': '-', 'change_profile': 'w'},
            {'color': 'green', 'linestyle': 'dotted', 'change_profile': 'w'}
        ]
        
        if( style_indx == -1 ):
            styles.append({'color': 'cyan', 'linestyle': '-', 'change_profile': 'a'})
            styles.append({'color': 'cyan', 'linestyle': 'dotted', 'change_profile': 'a', 'xlabel': 'Pause(p)/Word(w)/Activity(a) change', 'ylabel': 'CCDF', 'title': f'Distribution of change profiles for \nBLOC {compute_change_alphabet} strings from adjacent weeks.'})
        else:
            styles.append({'color': 'cyan', 'linestyle': '-', 'change_profile': 'a', 'xlabel': 'Pause(p)/Word(w)/Activity(a) change', 'ylabel': 'CCDF', 'title': f'Distribution of change profiles for \nBLOC {compute_change_alphabet} strings from adjacent weeks.'})
            styles.append({'color': 'cyan', 'linestyle': 'dotted', 'change_profile': 'a'})

        lg_lines = []
        #draw_change_profile_dist(pause_profile_dist, [styles[0], styles[1]], 'change_profile_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, lg_lines=lg_lines)
        #draw_change_profile_dist(word_profile_dist, [styles[2], styles[3]], 'change_profile_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, lg_lines=lg_lines)
        #draw_change_profile_dist(activity_profile_dist, [styles[4], styles[5]], 'change_profile_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, lg_lines=lg_lines, draw_plot=True)

        study_info_ops_drivers_change_context(bloc_collection, compute_change_alphabet, segmentation_type, class_to_study=['driver'])

        '''
        study_info_ops_drivers_cochange(info_ops_dets[agent]['dates_change_occurred'], bloc_collection, store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, info_ops_dets=info_ops_dets[agent], agent=agent)
        
        info_ops_dets[agent]['dates_change_dist'] = sorted( info_ops_dets[agent]['dates_change_dist'].items(), key=lambda x: x[1], reverse=True )
        info_ops_dets[agent]['dates_change_dist'] = [ 
            {'date': date_freq_tup[0], 'freq': date_freq_tup[1], 'rate': date_freq_tup[1]/info_ops_dets[agent]['dates_change_dist_total']} for date_freq_tup in info_ops_dets[agent]['dates_change_dist'] ]
        dist_dist_fname = '{}{}_{}_dist_change_dates.json'.format(f'./change-dists/{compute_change_alphabet}/', legend_title, agent)
        
        dumpJsonToFile(dist_dist_fname, info_ops_dets[agent]['dates_change_dist'])
        print('Saved:', dist_dist_fname)
        '''

def apk(y_true, y_pred, k=0):
  
  #credit: https://archive.ph/UmQOm/again#selection-1397.16-1397.21
  if k != 0:
    y_true_loc = y_true[:k]
    y_pred_loc = y_pred[:k]
    

  ori_len = len(y_true_loc)

  correct_predictions = 0
  running_sum = 0

  for i, yp_item in enumerate(y_pred_loc):
    rank = i+1 # our rank starts at 1
    #print(f'rank: {rank}, yp_item:', yp_item, 'vs', y_true_loc)

    if yp_item in y_true_loc:
      correct_predictions += 1
      running_sum += correct_predictions/rank
      y_true_loc.remove(yp_item)
    
    #print('\tcorrect_predictions:', correct_predictions)
    #print('\tcorrect_predictions/rank:', correct_predictions/rank)
    #print('\trunning_sum:', running_sum)
    #print()
  
  return running_sum/ori_len

def get_eval_metric_frm_conf_mat(TP, TN, FP, FN):
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    precision = TP/(TP+FP) if TP + FP != 0 else 0
    recall = TP/(TP+FN) if TP + FN != 0 else 0
    f1 = 2*((precision * recall)/(precision + recall)) if precision + recall != 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': TP + TN + FP + FN
    }

def calc_auc(actual, pred, pos_label=1):

    actual = np.array(actual)
    pred = np.array(pred)
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=pos_label)
    
    print( 'fpr:', fpr )
    print( 'tpr:', tpr )
    print( 'auc:', metrics.auc(fpr, tpr) )

def diversity_study(args, all_user_blocs, change_dynamics_method, **kwargs):

    all_user_blocs = add_change_dynamics_to_bloc(all_user_blocs, change_dynamics_method=change_dynamics_method, **kwargs)
    all_user_blocs = [ u for u in all_user_blocs if u['more_details']['total_tweets'] != 0 ]
    all_user_blocs = sorted( all_user_blocs, key=lambda x: x['change_dynamics']['change_dynamics_score'], reverse=True )
    total_users = len(all_user_blocs)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    class_maps = {
        'bot': 1,
        'human': 0,
        'driver': 1,
        'control': 0,
        'midterm': 1
    }

    actual = []
    prediction = []
    prec_k_val = 10

    for i in range(total_users):

        dbloc = all_user_blocs[i] 
        prediction.append(class_maps[dbloc['class']])
        actual_pred = 1 if i < total_users/2 else 0
        actual.append( actual_pred )
    
        print( 'pred class:', dbloc['class'] )
        print( '{} of {}: {} {}, tid: {}'.format(i, total_users, dbloc['user_id'], dbloc['screen_name'], 'x' ) )
        if( i < prec_k_val ):
            
            TP = TP + 1 if class_maps[dbloc['class']] == actual_pred and actual_pred == 1 else TP
            TN = TN + 1 if class_maps[dbloc['class']] == actual_pred and actual_pred == 0 else TN

            FP = FP + 1 if class_maps[dbloc['class']] != actual_pred and actual_pred == 0 else FP
            FN = FN + 1 if class_maps[dbloc['class']] != actual_pred and actual_pred == 1 else FN
            

            print(f'\t{i}')
        #if( i < 1000 or i > total_users - 1000 ):
        #    print('status_code: {}\n'.format( is_user_active(dbloc['user_id'])) )

        print('total_tweets: {}'.format(dbloc['more_details']['total_tweets']))
        print('first_tweet_created_at_local_time: {}'.format(dbloc['more_details']['first_tweet_created_at_local_time']))
        print('last_tweet_created_at_local_time: {}'.format(dbloc['more_details']['last_tweet_created_at_local_time']))
        print('datediff: {}'.format(datetime.strptime(dbloc['more_details']['last_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S') -  datetime.strptime(dbloc['more_details']['first_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S')))
        
        
        for alph in dbloc['bloc']:
            
            print(f'top BLOC {alph} words')
            top_bloc_words = dbloc['change_dynamics'][alph].pop('top_bloc_words', [])
            
            for j in range(len(top_bloc_words)):
                w = top_bloc_words[j]
                print( '\t{}. {}: {:.2f}'.format(j+1, w['term'], w['term_rate']) )
                if( w['term_rate'] < 0.01 ):
                    break

            print(f'BLOC {alph} words')
            print('\t', dbloc['bloc'][alph])
            print()

        print(json.dumps(dbloc['change_dynamics'], ensure_ascii=False))

        '''
        for alph, bloc_doc in dbloc['bloc'].items():
            
            print(f'\t{bloc_doc}\n')
            cust_highlight = 'h.' if dbloc['user_id'] in manual_highlight_user_ids else ''
            if( ('highlight' in dbloc or cust_highlight != '') and alph == 'action' ):
                study_info_ops_drivers_change_context_v1(dbloc, alph, outfile.write)
                stash_usr_tweets(dbloc['tweets'], '{}/saved-tweets/{}_{}_{}_tweets.jsonl.gz'.format(storage_path, cust_highlight+start_time.split('T')[0], i+1, dbloc['user_id']))
            print('\n')
        '''
                       
        print('\n')
        print('\n')

    print('prediction:')
    print(prediction)
    print('actual:')
    print(actual)

    calc_auc(deepcopy(actual), deepcopy(prediction))

    #print(f'precision@{prec_k_val}')
    #print(get_eval_metric_frm_conf_mat(TP, TN, FP, FN))
    
    print(f'Avg. Prec@{10}:', apk(actual, prediction, 10))
    print(f'Avg. Prec@{20}:', apk(actual, prediction, 20))
    print(f'Avg. Prec@{30}:', apk(actual, prediction, 30))
    print(f'Avg. Prec@{40}:', apk(actual, prediction, 40))
    print(f'Avg. Prec@{50}:', apk(actual, prediction, 50))
    print(f'Avg. Prec@{100}:', apk(actual, prediction, 100))

def change_ref_dates_study(args, bloc_model, gen_bloc_params):
    
    params = vars(args)
    study_dates_ranges = [ (d[0].strftime('%Y-%m-%d') + ' 00:00', d[1].strftime('%Y-%m-%d') + ' 23:59') for d in args.study_dates_ranges ]

    info_ops_accnts = get_bloc_for_info_ops_users(args, bloc_model, gen_bloc_params, date_ranges=study_dates_ranges)

    print('\nchange_ref_dates_study():')
    print('\targs.study_dates:', args.study_dates)
    print('\targs.study_dates_ranges:', args.study_dates_ranges)

    '''
        print('contro_all_user_blocs:')
        for u in info_ops_accnts['contro_all_user_blocs']:
            print(u['screen_name'], u['user_id'], len(u['tweets']))
            #break
        print()

        print('driver_all_user_blocs:')
        for u in info_ops_accnts['driver_all_user_blocs']:
            print(u['screen_name'], u['user_id'], len(u['tweets']))
            #break
        print()
    '''

    shuffle(info_ops_accnts['contro_all_user_blocs'])
    info_ops_accnts['contro_all_user_blocs'] = info_ops_accnts['contro_all_user_blocs'][:len(info_ops_accnts['driver_all_user_blocs'])]
    diversity_study( args, info_ops_accnts['driver_all_user_blocs'] + info_ops_accnts['contro_all_user_blocs'], change_dynamics_method=slide_time_window_over_bloc_tweets_v3, **params )


def info_ops_study(args, bloc_model, gen_bloc_params):
    
    print('\ninfo_ops_study()')
    print('\ttweets_file:', args.tweets_file)
    print('\toutpath:', args.outpath)
    
    #driver_file = '2020_12/armenia_202012/driver_tweets.csv.gz'
    #driver_file = '2019_01/bangladesh_201901_1/driver_tweets.csv.gz'
    #driver_file = '2019_06/catalonia_201906_1/driver_tweets.csv.gz'
    #driver_file = '2019_08/china_082019_1/driver_tweets.csv.gz'
    #driver_file = '2019_08/china_082019_2/driver_tweets.csv.gz'
    #driver_file = '2021_12/CNCC_0621_YYYY/CNCC_0621_2021/driver_tweets.csv.gz'
    #driver_file = '2021_12/CNHU_0621_YYYY/CNHU_0621_2020/driver_tweets.csv.gz'
    #driver_file = '2021_12/CNHU_0621_YYYY/CNHU_0621_2021/driver_tweets.csv.gz'
    #driver_file = '2020_08/cuba_082020/driver_tweets.csv.gz'
    #driver_file = '2019_08/ecuador_082019_1/driver_tweets.csv.gz'
    #driver_file = '2019_08/egypt_uae_082019_1/driver_tweets.csv.gz'
    #driver_file = '2020_03/ghana_nigeria_032020/driver_tweets.csv.gz'
    #driver_file = '2018_10/ira/driver_tweets.csv.gz'
    #driver_file = '2019_01/russia_201901_1/driver_tweets.csv.gz'
    #driver_file = '2020_12/IRA_202012/driver_tweets.csv.gz'
    #driver_file = '2020_12/GRU_202012/driver_tweets.csv.gz'
    #driver_file = '2020_09/ira_092020/driver_tweets.csv.gz'
    #driver_file = '2018_10/iranian/driver_tweets.csv.gz'
    #driver_file = '2019_01/iran_201901_X/iran_201901_1/driver_tweets.csv.gz'
    #driver_file = '2019_06/iran_201906_1/driver_tweets.csv.gz'
    #driver_file = '2019_06/iran_201906_2/driver_tweets.csv.gz'
    #driver_file = '2019_06/iran_201906_3/driver_tweets.csv.gz'
    #driver_file = '2020_09/iran_092020/driver_tweets.csv.gz'
    #driver_file = '2020_12/iran_202012/driver_tweets.csv.gz'
    #driver_file = '2021_12/MX_0621_YYYY/MX_0621_2019/driver_tweets.csv.gz'
    #driver_file = '2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz'
    #driver_file = '2020_08/qatar_082020/driver_tweets.csv.gz'
    #driver_file = '2019_08/spain_082019_1/driver_tweets.csv.gz'
    #driver_file = '2020_09/thailand_092020/driver_tweets.csv.gz'
    #driver_file = '2019_08/egypt_uae_082019_1/driver_tweets.csv.gz'
    #driver_file = '2021_12/uganda_0621_YYYY/uganda_0621_2019/driver_tweets.csv.gz'
    #driver_file = '2021_12/uganda_0621_YYYY/uganda_0621_2020/driver_tweets.csv.gz'
    #driver_file = '2019_01/venezuela_201901_1/driver_tweets.csv.gz'
    #driver_file = '2019_01/venezuela_201901_2/driver_tweets.csv.gz'
    #driver_file = '2019_06/venezuela_201906_1/driver_tweets.csv.gz'
    #driver_file = '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2020/driver_tweets.csv.gz'
    #driver_file = '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2021/driver_tweets.csv.gz'
    
    all_tweets_files = ['2020_12/armenia_202012/driver_tweets.csv.gz', '2019_01/bangladesh_201901_1/driver_tweets.csv.gz', '2019_06/catalonia_201906_1/driver_tweets.csv.gz', '2019_08/china_082019_1/driver_tweets.csv.gz', '2019_08/china_082019_2/driver_tweets.csv.gz', '2021_12/CNCC_0621_YYYY/CNCC_0621_2021/driver_tweets.csv.gz', '2021_12/CNHU_0621_YYYY/CNHU_0621_2020/driver_tweets.csv.gz', '2021_12/CNHU_0621_YYYY/CNHU_0621_2021/driver_tweets.csv.gz', '2020_08/cuba_082020/driver_tweets.csv.gz', '2019_08/ecuador_082019_1/driver_tweets.csv.gz', '2019_08/egypt_uae_082019_1/driver_tweets.csv.gz', '2020_03/ghana_nigeria_032020/driver_tweets.csv.gz', '2018_10/ira/driver_tweets.csv.gz', '2019_01/russia_201901_1/driver_tweets.csv.gz', '2020_12/IRA_202012/driver_tweets.csv.gz', '2020_12/GRU_202012/driver_tweets.csv.gz', '2020_09/ira_092020/driver_tweets.csv.gz', '2018_10/iranian/driver_tweets.csv.gz', '2019_01/iran_201901_X/iran_201901_1/driver_tweets.csv.gz', '2019_06/iran_201906_1/driver_tweets.csv.gz', '2019_06/iran_201906_2/driver_tweets.csv.gz', '2019_06/iran_201906_3/driver_tweets.csv.gz', '2020_09/iran_092020/driver_tweets.csv.gz', '2020_12/iran_202012/driver_tweets.csv.gz', '2021_12/MX_0621_YYYY/MX_0621_2019/driver_tweets.csv.gz', '2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz', '2020_08/qatar_082020/driver_tweets.csv.gz', '2019_08/spain_082019_1/driver_tweets.csv.gz', '2020_09/thailand_092020/driver_tweets.csv.gz', '2019_08/egypt_uae_082019_1/driver_tweets.csv.gz', '2021_12/uganda_0621_YYYY/uganda_0621_2019/driver_tweets.csv.gz', '2021_12/uganda_0621_YYYY/uganda_0621_2020/driver_tweets.csv.gz', '2019_01/venezuela_201901_1/driver_tweets.csv.gz', '2019_01/venezuela_201901_2/driver_tweets.csv.gz', '2019_06/venezuela_201906_1/driver_tweets.csv.gz', '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2020/driver_tweets.csv.gz', '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2021/driver_tweets.csv.gz']
    driver_file = '2020_03/ghana_nigeria_032020/driver_tweets.csv.gz' if args.tweets_file.strip() == '' else args.tweets_file
        
    campaign = driver_file.split('/')[-2]
    args.min_user_count = 999999999#10, 
    print(f'args.min_user_count coerced: {args.min_user_count}\n'*20)

    args.tweets_path = args.tweets_path.strip()
    args.tweets_path = args.tweets_path if ( args.tweets_path.endswith('/') or args.tweets_path == '' ) else args.tweets_path.strip() + '/'

    payload = get_driver_post_stats( f'{args.tweets_path}{driver_file}', start_datetime='', end_datetime='' )
    driver_post_dates_dets = get_driver_post_date_dist( payload['driver_posts_stats_yyyy_mm_dd'], payload['total_posts'] )
    info_ops_study_drivers_vs_control_users(driver_post_dates_dets, driver_file, campaign, args, bloc_model, gen_bloc_params)

def gen_bloc_for_user(users, gen_bloc_params, accnt_details):

    job_lst = []
    all_bloc_symbols = get_default_symbols()
    
    for uid, tweets in users.items():
    
        #to_print_out = to_print_out if len(job_lst) % 10 == 0 else ''
        job_lst.append({
            'func': add_bloc_sequences,
            'args': {'tweets': tweets, 'all_bloc_symbols': all_bloc_symbols, **gen_bloc_params},
            'print': '',
            'misc': None
        })
    
    all_usr_blocs = parallelTask(job_lst, threadCount=5)
    #all_user_blocs = [ res['output'] for res in all_usr_blocs ]
    all_user_blocs = []
    for ou in all_usr_blocs:
        ou['output'].update({
            'tweet_count': len(ou['input']['args']['tweets']),
            **accnt_details
        })
        all_user_blocs.append( ou['output'] )

    print('\ngen_bloc_for_user(): DONE add_bloc_sequences() for', len(job_lst), 'users')
    return all_user_blocs

def calc_change_dynamics_frm_tweets_v0(u_tweets, bloc_alphabets):
        
    change_dynamics = {}
    
    for alph in bloc_alphabets:
        change_dynamics[alph] = slide_time_window_over_bloc_tweets( u_tweets, alph )

    return change_dynamics

def calc_change_dynamics_frm_tweets(u_bloc, u_tweets, bloc_alphabets, **kwargs):
    
    #kwargs['change_dynamics_method']: get_bloc_entropy(), slide_time_window_over_bloc_tweets_v2(), slide_time_window_over_bloc_tweets_v3()
    
    change_dynamics = {}
    
    for alph in bloc_alphabets:
        
        if( alph not in u_bloc ):
            continue

        #change_dynamics[alph] = get_vocab_and_doc_len( u_bloc[alph], bloc_alphabet=alph, **kwargs )
        change_dynamics[alph] = kwargs['change_dynamics_method']( u_tweets, u_bloc[alph], bloc_alphabet=alph, **kwargs )


    return change_dynamics


def add_change_dynamics_to_bloc(all_user_bloc, change_dynamics_method, bloc_alphabets=['action', 'content_syntactic'], **kwargs):

    job_lst = []
    total_users = len(all_user_bloc)
    for i in range(total_users):
        
        dbloc = all_user_bloc[i]
        dbloc['change_dynamics'] = {}
        dbloc.setdefault('tweet_count', len(dbloc.get('tweets', [])))
        
        job_lst.append({
            'func': calc_change_dynamics_frm_tweets,
            'args': {'u_bloc': dbloc['bloc'], 'u_tweets': dbloc['tweets'], 'bloc_alphabets': bloc_alphabets, 'change_dynamics_method': change_dynamics_method, **kwargs},
            'print': f'\tcalc_change_dynamics_frm_tweets(): {i} of {total_users}' if i % 1000 == 0 else '',
            'misc': None
        })


    all_bloc_chg_metrics = {}
    human_bloc_metric = {}
    all_usr_change_dynamics = parallelTask(job_lst, threadCount=5)

    for i in range(total_users):
        
        all_user_bloc[i]['change_dynamics'] = all_usr_change_dynamics[i]['output']

        for alph in all_user_bloc[i]['change_dynamics']:

            all_bloc_chg_metrics.setdefault(alph, {'word_entropy': [], 'char_entropy': [], 'avg_cosine_dist': [] })
            human_bloc_metric.setdefault(alph, {'word_entropy': [], 'char_entropy': [], 'avg_cosine_dist': [] })
            for metric in ['word_entropy', 'char_entropy', 'avg_cosine_dist']:
                
                if( metric not in all_user_bloc[i]['change_dynamics'][alph] ):
                    continue

                metric_value = all_user_bloc[i]['change_dynamics'][alph][metric]
                if( metric_value == -1 ):
                    continue

                all_bloc_chg_metrics[alph][metric].append(metric_value)
                if( all_user_bloc[i]['class'] == 'human' ):
                    human_bloc_metric[alph][metric].append(metric_value)
    

    print('all_bloc_chg_metrics summary:')
    human_bloc_chng_metrics = False
    for alph, ent_payload in all_bloc_chg_metrics.items():
        print(f'\t{alph}:')
        for ent_metric in ent_payload:
            
            if( len(ent_payload[ent_metric]) == 0 ):
                continue
            
            mean = statistics.mean( ent_payload[ent_metric] )
            stdev = statistics.stdev( ent_payload[ent_metric] )
            ent_payload[ent_metric] = {'mean': mean, 'stdev': stdev}
            
            print(f'\t\t{ent_metric}: mean: {mean}, stdev: {stdev}')
            if( len(human_bloc_metric[alph][ent_metric]) != 0 ):
                human_bloc_chng_metrics = True
                mean = statistics.mean( human_bloc_metric[alph][ent_metric] )
                stdev = statistics.stdev( human_bloc_metric[alph][ent_metric] )
                human_bloc_metric[alph][ent_metric] = {'mean': mean, 'stdev': stdev}


    #all_bloc_chg_metrics = human_bloc_metric if human_bloc_chng_metrics is True else all_bloc_chg_metrics
    
    #'''
    print('coercing all_bloc_chg_metrics\n'*10)
    
    if( change_dynamics_method == get_bloc_entropy ):

        all_bloc_chg_metrics = {'action': {'word_entropy': {'mean': 0.7055596029364635, 'stdev': 0.0617149037895202}, 'char_entropy': {'mean': 0.8121314403565987, 'stdev': 0.09383841610741146}, 'avg_cosine_dist': []}, 'content_syntactic': {'word_entropy': {'mean': 0.659116834555024, 'stdev': 0.1270876848391208}, 'char_entropy': {'mean': 0.7441871223887485, 'stdev': 0.11845363421814638}, 'avg_cosine_dist': []}}
                               
    elif( change_dynamics_method == slide_time_window_over_bloc_tweets_v2 ):
        
        all_bloc_chg_metrics = {'action': {'word_entropy': [], 'char_entropy': [], 'avg_cosine_dist': {'mean': 0.46947384320607616, 'stdev': 0.11527244540217742}}, 'content_syntactic': {'word_entropy': [], 'char_entropy': [], 'avg_cosine_dist': {'mean': 0.6536864422297024, 'stdev': 0.17527634862050348}}}
    
    elif( change_dynamics_method == slide_time_window_over_bloc_tweets_v3 ):
        
        pass

    else:
        print(f'{change_dynamics_method} not implemented, exiting.')
        sys.exit(0)

    #'''

    print('all_bloc_chg_metrics:')
    print('\t', all_bloc_chg_metrics)

    '''
    #to get empirical bloc_chg_metrics, run this block with sys.exit(0) at end: time python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/bot-detection/retraining_data --task diversity_study --outpath=./ kevin_feedback gilani-17 midterm-2018 varol-icwsm verified zoher-organization (also run with --task=change_segments_study)
    #then set all_bloc_chg_metrics to human_bloc_metric in if( change_dynamics_method == get_bloc_entropy ): condition
    print('human_bloc_metric:')
    print('\t', human_bloc_metric)
    sys.exit(0)
    '''
    
    for i in range(total_users):
        dbloc = all_user_bloc[i]
        all_user_bloc[i]['change_dynamics']['change_dynamics_score'] = get_chng_dynamics_score(all_user_bloc[i]['change_dynamics'], all_bloc_chg_metrics=all_bloc_chg_metrics, method=change_dynamics_method)
    
    return all_user_bloc

def add_change_dynamics_to_bloc_v1(all_user_bloc, bloc_alphabets=['action', 'content_syntactic']):

    job_lst = []
    total_users = len(all_user_bloc)
    for i in range(total_users):
        
        dbloc = all_user_bloc[i]
        dbloc['change_dynamics'] = {}
        dbloc.setdefault('tweet_count', len(dbloc.get('tweets', [])))

        #calc_change_dynamics_frm_tweets(dbloc['tweets'], bloc_alphabets)
        #calc_change_dynamics_frm_tweets(d['bloc'], bloc_alphabets)
        job_lst.append({
            'func': calc_change_dynamics_frm_tweets,
            #'args': {'u_tweets': dbloc['tweets'], 'bloc_alphabets': bloc_alphabets},
            'args': {'u_bloc': dbloc['bloc'], 'bloc_alphabets': bloc_alphabets},
            'print': f'\tcalc_change_dynamics_frm_tweets(): {i} of {total_users}' if i % 1000 == 0 else '',
            'misc': None
        })

    all_bloc_chg_metrics = {}
    human_bloc_entropy = {}
    all_usr_change_dynamics = parallelTask(job_lst, threadCount=5)

    for i in range(total_users):
        
        all_user_bloc[i]['change_dynamics'] = all_usr_change_dynamics[i]['output']
        all_user_bloc[i]['change_dynamics']['change_dynamics_score'] = get_chng_dynamics_score(all_user_bloc[i]['change_dynamics'], method=get_vocab_and_doc_len)
    
    return all_user_bloc

def obsolete_draw_driver_control_change_dynamics(driver, control, **kwargs):

    def drawer(accnts, bloc_alphabet, plt, **kwargs):

        color = kwargs.get('color', 'black')
        marker = kwargs.get('marker', '*')
        facecolors = kwargs.get('facecolors', color)
        pnt_size = kwargs.get('size', 5)
        
        result = {
            'max_plot_x': -1,
            'min_plot_x': 100000
        }

        X = []
        Y = []
        for d in accnts:
            
            if( len(d['change_dynamics']) == 0 ):
                continue

            unique_dates = d['change_dynamics'][bloc_alphabet]['unique_dates']
            total_unique_words = d['change_dynamics'][bloc_alphabet]['total_unique_words']
            
            if( unique_dates == 0 or total_unique_words == 0 ):
                continue

            #X.append( unique_dates )
            #Y.append( total_unique_words ) 

            X.append( math.log10(unique_dates) )
            Y.append( math.log10(total_unique_words) ) 

            result['max_plot_x'] = X[-1] if X[-1] > result['max_plot_x'] else result['max_plot_x']
            result['min_plot_x'] = X[-1] if X[-1] < result['min_plot_x'] else result['min_plot_x']

        scat = plt.scatter(X, Y, alpha=0.5, s=pnt_size, edgecolors=color, marker=marker, facecolors=facecolors)
        plt.title(kwargs.get('title', ''))
        plt.xlabel(kwargs.get('xlabel', ''))
        plt.ylabel(kwargs.get('ylabel', ''))

        return result

    plt.figure(num=1, figsize=(14, 10), dpi=300)
    plt.subplot(211)
    campaign = kwargs.get('campaign', 'campaign')
    
    min_x = 100000
    max_x = -1


    fst_result = drawer(control, 'action', plt, size=5, color='blue', marker='.', title='action', xlabel='Total posting days', ylabel='Total BLOC words')
    min_x = min(fst_result['min_plot_x'], min_x)
    max_x = max(fst_result['max_plot_x'], max_x)

    sec_result = drawer(driver, 'action', plt, color='red', marker='+', edgecolors='white', title='action', xlabel='Total posting days', ylabel='Total BLOC words')
    min_x = min(sec_result['min_plot_x'], min_x)
    max_x = max(sec_result['max_plot_x'], max_x)


    plt.subplot(212)
    drawer(control, 'content_syntactic', plt, size=5, color='blue', marker='.', title='action', xlabel='Total posting days', ylabel='Total BLOC words')
    min_x = min(fst_result['min_plot_x'], min_x)
    max_x = max(fst_result['max_plot_x'], max_x)

    drawer(driver, 'content_syntactic', plt, color='red', marker='+', edgecolors='white', title='content', xlabel='Total posting days', ylabel='Total BLOC words')
    min_x = min(sec_result['min_plot_x'], min_x)
    max_x = max(sec_result['max_plot_x'], max_x)

    
    print('saving', f'{campaign}_time_vocab.png')
    plt.savefig(f'{campaign}_time_vocab.png')

def draw_driver_control_change_dynamics_v0(driver, control, **kwargs):

    def tick_fmt(x, y):
        return '{:,.2f}'.format(10**x)

    def drawer(accnts, bloc_alphabet, plt, **kwargs):

        color = kwargs.get('color', 'black')
        marker = kwargs.get('marker', '*')
        facecolors = kwargs.get('facecolors', color)
        pnt_size = kwargs.get('size', 5)
        do_linear_regression = kwargs.get('do_linear_regression', False)

        title = kwargs.get('title', '')
        xlabel = kwargs.get('xlabel', '')
        ylabel = kwargs.get('ylabel', '')

        result = {
            'slope': -1,
            'intercept': -1,
            'max_plot_x': -1,
            'min_plot_x': 100000
        }

        X = []
        Y = []

        X_highlight = []
        Y_highlight = []
        for i in range( len(accnts) ):
            
            d = accnts[i]
            if( len(d['change_dynamics']) == 0 ):
                continue

            accnt_name = d['screen_name']
            change_dynamics_score = d['change_dynamics']['change_dynamics_score']
            doc_len = d['change_dynamics'][bloc_alphabet]['doc_len']
            vocab_len = d['change_dynamics'][bloc_alphabet]['vocab_len']
            
            if( doc_len == 0 or vocab_len == 0 ):
                continue

            #X.append( doc_len )
            #Y.append( vocab_len )

            X.append( math.log10(doc_len) )
            Y.append( math.log10(vocab_len) ) 

            if( 'highlight' in d ):
                print(f'highlight {accnt_name}: {i} {doc_len} {vocab_len}, change_dynamics_score: {change_dynamics_score}')
                X_highlight.append( math.log10(doc_len) )
                Y_highlight.append( math.log10(vocab_len) ) 

            result['max_plot_x'] = X[-1] if X[-1] > result['max_plot_x'] else result['max_plot_x']
            result['min_plot_x'] = X[-1] if X[-1] < result['min_plot_x'] else result['min_plot_x']

        scat = plt.scatter(X, Y, alpha=0.5, s=pnt_size, edgecolors=color, marker=marker, facecolors=facecolors)
        plt.scatter(X_highlight, Y_highlight, alpha=1, s=pnt_size, marker=marker, facecolors='black')
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(tick.FuncFormatter(tick_fmt))
        ax.yaxis.set_major_formatter(tick.FuncFormatter(tick_fmt))

        if( len(X) != 0 and do_linear_regression is True ):
        
            X = np.array(X).reshape(-1, 1)
            Y = np.array(Y).reshape(-1, 1)
            reg = LinearRegression().fit( X, Y )
            
            #print('reg.coef_:', reg.coef_)
            #print('intercept:', reg.intercept_)

            result['slope'] = reg.coef_[0][0]
            result['intercept'] = reg.intercept_[0]
            #plt.scatter([0], [result['intercept']], alpha=1, s=pnt_size, edgecolors='red', marker='x')
            result['r_score'] = reg.score(X, Y)
            title = '{} (line: {:.4f}x + {:.4f}, r^2: {:.4f})'.format(title, result['slope'], result['intercept'], result['r_score'])

        title = title.strip()
        xlabel = xlabel.strip()
        ylabel = ylabel.strip()

        if( title != '' ):
            plt.title(title)
        if( xlabel != '' ):
            plt.xlabel(xlabel)
        if( ylabel != '' ):
            plt.ylabel(ylabel)

        return {} if len(accnts) == 0 else result

    def highlight_drivers(drivers, selected_drivers, front=True):

        if( front is True ):
            for i in range(0, selected_drivers):
                drivers[i]['highlight'] = True
        else:
            for i in range(len(drivers) - 1, len(drivers) - selected_drivers - 1, -1):
                drivers[i]['highlight'] = True

    def run_drawer(plt, driver, control, bloc_alphabet, line, **kwargs):

        slug = kwargs.get('slug', '')
        print('bloc_alphabet:', bloc_alphabet)

        min_x = 0#100000
        max_x = -1

        fst_result = drawer(control, bloc_alphabet, plt, do_linear_regression=True, size=5, color='blue', marker='.', title=slug + bloc_alphabet.replace('_', ' '), xlabel='Document length', ylabel='Vocabulary length')
        #min_x = min(fst_result.get('min_plot_x', min_x), min_x)
        max_x = max(fst_result.get('max_plot_x', max_x), max_x)

        sec_result = drawer(driver, bloc_alphabet, plt, color='red', marker='+', edgecolors='white', title='', xlabel='Document length', ylabel='Vocabulary length')
        #min_x = min(sec_result.get('min_plot_x', min_x), min_x)
        max_x = max(sec_result.get('max_plot_x', max_x), max_x)
        print('\tmin_x/max_x:', min_x, max_x)
        max_x += 0.05

        fst_result = line if len(fst_result) == 0 else fst_result
        
        if( len(fst_result) != 0 ):
            plt.plot([min_x, max_x], [fst_result['slope'] * min_x + fst_result['intercept'], fst_result['slope'] * max_x + fst_result['intercept']])
            print('\tmin_y/max_y:', fst_result['slope'] * min_x + fst_result['intercept'], fst_result['slope'] * max_x + fst_result['intercept'])
            print('\tslope/intercept:', fst_result['slope'], fst_result['intercept'], fst_result['r_score'])
            print()
        

    plt.figure(num=1, figsize=(10, 10), dpi=300)
    

    campaign = kwargs.get('campaign', 'campaign')
    action_line = kwargs.get('line', {}).get('action', {})
    content_line = kwargs.get('line', {}).get('content_syntactic', {})
    outfilename = kwargs.get('outfilename', f'{campaign}_heaps_plot.png')

    driver_highlight = int(0.05*len(driver))
    plt.subplot(211)
    highlight_drivers(driver, driver_highlight, front=True)
    highlight_drivers(driver, driver_highlight, front=False)
    run_drawer(plt, driver, control, 'action', line=action_line, slug=f'{campaign}: ')
    
    plt.subplot(212)
    run_drawer(plt, driver, control, 'content_syntactic', line=content_line, slug=f'{campaign}: ')
    

    '''
    plt.subplot(212)
    fst_result = drawer(control, 'content_syntactic', plt, do_linear_regression=True, size=5, color='blue', marker='.', title='action', xlabel='Document length', ylabel='Vocabulary length')
    min_x = min(fst_result['min_plot_x'], min_x)
    max_x = max(fst_result['max_plot_x'], max_x)

    drawer(driver, 'content_syntactic', plt, color='red', marker='+', edgecolors='white', title='content', xlabel='Document length', ylabel='Vocabulary length')
    min_x = min(sec_result['min_plot_x'], min_x)
    max_x = max(sec_result['max_plot_x'], max_x)

    print('min_x/max_x:', min_x, max_x)
    print('min_y/max_y:', fst_result['slope'] * min_x + fst_result['intercept'], fst_result['slope'] * max_x + fst_result['intercept'])
    print('slope/intercept:', fst_result['slope'], fst_result['intercept'])
    print()
    plt.plot([min_x, max_x], [fst_result['slope'] * min_x + fst_result['intercept'], fst_result['slope'] * max_x + fst_result['intercept']])
    '''
    print(f'saving: {outfilename}')

    plt.tight_layout()
    plt.savefig(outfilename)


def stash_usr_tweets(tweets, outfilename):

    outfile = gzip.open(outfilename, 'wt')
    for t in tweets:
        del t['bloc']
        outfile.write(json.dumps(t, ensure_ascii=False) + '\n')

    outfile.close()
    print(f'\nwrote {outfilename}')

def rehydrate_tweet(twt_id, user_agent='', token='abcde'):

    #grandparent: https://github.com/JustAnotherArchivist/snscrape/issues/996
    #parent: https://github.com/JustAnotherArchivist/snscrape/issues/996#issuecomment-1615937362
    url = "https://cdn.syndication.twimg.com/tweet-result"
    querystring = {"id": twt_id, "lang":"en", 'token': token}

    ''' 
        How to get token
        1. visit https://platform.twitter.com/embed/Tweet.html?id=1288498682971795463
        2. Inspect network traffic in developer tools to find URI of GET request which has token
    '''

    payload = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/116.0" if user_agent == '' else user_agent,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://platform.twitter.com",
        "Connection": "keep-alive",
        "Referer": "https://platform.twitter.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "TE": "trailers"
    }
    
    try:
        response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
        return json.loads(response.text)
    except:
        genericErrorInfo(f"Problem text:\n {response.text}\n")

    return {}

def is_user_active(tid):

    res = rehydrate_tweet(tid, token=str(random.randint(1, 10000000))+str(random.randint(1, 10000000)) )
    if( len(res) == 0 ):
        return {
            'active': False,
            'status_message': 'Request error'
        }

    tombstone = res.get('tombstone', {}).get('text', {}).get('text', '')
    
    return {
        'active': True if tombstone == '' else False,
        'status_message': tombstone
    }


def get_midterm_2022_tweets(in_filename, start_time, end_time, max_tweets=-1, min_tweets=50):

    def fix_midterm_tweet(tweet):

        tweet['created_at'] = tweet['created_at'].replace('  ', ' +0000 ')
        tweet['tweet_time'] = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%dT%H:%M:%S')

        #
        new_urls = []
        for i in range(len(tweet['entities']['urls'])):
            u = tweet['entities']['urls'][i]
            if( 'expanded_url' in u ):
                new_urls.append(u)
        tweet['entities']['urls'] = new_urls

        if( 'retweeted_status' in tweet ):
            
            rt_new_urls = []
            for i in range(len(tweet['retweeted_status']['entities']['urls'])):
                u = tweet['retweeted_status']['entities']['urls'][i]
                if( 'expanded_url' in u ):
                    rt_new_urls.append(u)
            tweet['retweeted_status']['entities']['urls'] = rt_new_urls

        return tweet

    print('\nget_midterm_2022_tweets():')
    users_tweets = {}
    dedup_tweets = {}
    tweet_count = -1
    infile = gzip.open(in_filename, 'rb')

    has_collected = False
    for line in infile:

        tweet_count += 1
        if( tweet_count == max_tweets ):
            print(f'\treached limit, breaking {tweet_count} = {max_tweets}')
            break

        twt = json.loads(line)
        twt = fix_midterm_tweet(twt)

        if( twt['tweet_time'] >= start_time and twt['tweet_time'] <= end_time ):
            has_collected = True
        elif( has_collected is True ):
            print('\tout of time range, breaking:', twt['tweet_time'], '\n')
            break
        else:
            if( tweet_count % 50000 == 0 ):
                print('skipping out of range, len(users_tweets):', len(users_tweets), f'tweets: {tweet_count}', 'created_at:', twt['created_at'])
            continue

        uid = twt['user']['id']
        
        users_tweets.setdefault(uid, [])
        dedup_tweets.setdefault(uid, set())

        if( twt['id'] not in dedup_tweets[uid] ):
            users_tweets[uid].append( twt )
            dedup_tweets[uid].add( twt['id'] )

        if( len(users_tweets) % 50000 == 0 ):
            print('len(users_tweets):', len(users_tweets), f'tweets: {tweet_count}', 'created_at:', twt['created_at'])
    
    infile.close()

    final_tweets = {}
    for uid, u_tweets in users_tweets.items():
        if( len(u_tweets) >= min_tweets ):
            final_tweets[uid] = u_tweets

    return final_tweets

def midterm_2022_study(args, bloc_model, gen_bloc_params):

    def get_midterm_2022_tweets_stats_v0(in_filename):

        tweet_count = -1
        users_tweets = {}
        infile = gzip.open(in_filename, 'rb')

        for line in infile:

            tweet_count += 1
            twt = json.loads(line)
            
            uid = twt['user']['id']
            users_tweets.setdefault(uid, 0)
            users_tweets[uid] += 1
            
            if( len(users_tweets) % 100000 == 0 ):
                print('get_midterm_2022_tweets_stats(): users:', len(users_tweets), f'tweets: {tweet_count}', 'created_at:', twt['created_at'])
        
        infile.close()

        print('total_user:', len(users_tweets))
        users_tweets = sorted(users_tweets.items(), key=lambda x: x[1], reverse=True)
        for uid, tweet_count in users_tweets.items():
            print('\t', uid, tweet_count)

        return stats

    print('\nmidterm_2022_study()')
    print('\ttweets_file:', args.tweets_file)
    print('\ttweets_path:', args.tweets_path)
    print('\toutpath:', args.outpath)
    in_filename = f'{args.tweets_path}{args.tweets_file}'
    manual_highlight_user_ids = ['2648502412', '593813785', '1512267118313721857', '1309853695157231616', '1260116792095186944', '1556023047156424704', '1553369360659632128', '1553381635399290880', '1328329017158725633', '1300550310998544390', '1490290225574252545', '1304305110735097856', '1284944226644561920', '1499339754483044353', '1490275200172269568', '1284031731104972800', '3053550018', '724327951370579968', '32920518', '1151118460665159680', '904351845769895937']
    
    storage_path = f'{args.outpath}/stable-results/midterm-2022/'
    os.makedirs(f'{storage_path}/saved-tweets/', exist_ok=True)
    
    min_tweets = 100
    max_tweets = -1
    start_time = '2022-12-01T00:00:00'
    end_time = '2022-12-31T11:59:59'

    outfilename = f'{storage_path}midterm_2022_chng_' + '{}_{}.txt'.format(start_time.split('T')[0], end_time.split('T')[0])
    print('\toutfilename:', outfilename)
    outfile = open(outfilename, 'w')

    outfile.write(f'{start_time} -- {end_time}\n')
    users_tweets = get_midterm_2022_tweets(in_filename, start_time=start_time, end_time=end_time, max_tweets=max_tweets, min_tweets=min_tweets)
    print('final len(users_tweets):', len(users_tweets), f'max_tweets: {max_tweets}')

    print('generating BLOC for midterm - start')
    midterm_all_user_blocs = gen_bloc_for_user( users_tweets, gen_bloc_params, accnt_details={'src': 'midterm_2022', 'class': 'mixed'} )
    print('generating BLOC for midterm - end')
    print('len(midterm_all_user_blocs):', len(midterm_all_user_blocs))
    midterm_all_user_blocs = add_change_dynamics_to_bloc(midterm_all_user_blocs)
    
    #Rom 6,7
    bloc_vdl_model = get_bloc_vocab_doc_len_linear_model()
    midterm_all_user_blocs = sorted( midterm_all_user_blocs, key=lambda x: x['change_dynamics']['change_dynamics_score'], reverse=True )
    draw_driver_control_change_dynamics_v0(campaign='midterm_2022', driver=midterm_all_user_blocs, control=[], bloc_vdl_model=bloc_vdl_model, line=bloc_vdl_model, outfilename=f'{outfilename}'.replace('.txt', '.png'))
    total_users = len(midterm_all_user_blocs)
    
    for i in range(total_users):

        dbloc = midterm_all_user_blocs[i] 

        outfile.write('{} of {}: {} {}, tid: {}\n'.format(i+1, total_users, dbloc['user_id'], dbloc['screen_name'], users_tweets[dbloc['user_id']][0]['id'] ))
        
        #if( i < 1000 or i > total_users - 1000 ):
        #    outfile.write('status_code: {}\n'.format( is_user_active(dbloc['user_id'])) )

        outfile.write('total_tweets: {}\n'.format(dbloc['more_details']['total_tweets']))
        outfile.write('first_tweet_created_at_local_time: {}\n'.format(dbloc['more_details']['first_tweet_created_at_local_time']))
        outfile.write('last_tweet_created_at_local_time: {}\n'.format(dbloc['more_details']['last_tweet_created_at_local_time']))
        outfile.write('datediff: {}\n'.format(datetime.strptime(dbloc['more_details']['last_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S') -  datetime.strptime(dbloc['more_details']['first_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S')))
        
        for alph in dbloc['bloc']:
            
            outfile.write(f'top BLOC {alph} words\n')
            top_bloc_words = dbloc['change_dynamics'][alph].pop('top_bloc_words', [])
            
            for j in range(len(top_bloc_words)):
                w = top_bloc_words[j]
                outfile.write( '\t{}. {}: {:.2f}\n'.format(j+1, w['term'], w['term_rate']) )
                if( w['term_rate'] < 0.01 ):
                    break

        outfile.write(json.dumps(dbloc['change_dynamics'], ensure_ascii=False) + '\n')

        for alph, bloc_doc in dbloc['bloc'].items():
            
            outfile.write(f'\t{bloc_doc}\n')
            cust_highlight = 'h.' if dbloc['user_id'] in manual_highlight_user_ids else ''
            if( ('highlight' in dbloc or cust_highlight != '') and alph == 'action' ):
                study_info_ops_drivers_change_context_v1(dbloc, alph, outfile.write)
                stash_usr_tweets(dbloc['tweets'], '{}/saved-tweets/{}_{}_{}_tweets.jsonl.gz'.format(storage_path, cust_highlight+start_time.split('T')[0], i+1, dbloc['user_id']))
            outfile.write('\n')
                       
        outfile.write('\n')
        outfile.write('\n')  

    print(f'wrote: {outfilename}')
    outfile.close()

def get_all_per_user_tweets(all_tweets, max_tweets, min_tweets):

    users_tweets = {}
    final_users_tweets = {}
    
    for twt in all_tweets:
        uid = twt['user']['id']
        users_tweets.setdefault(uid, [])
        users_tweets[uid].append( twt )

    for uid in users_tweets:
        if( len(users_tweets[uid]) > max_tweets ):
            users_tweets[uid] = sorted( users_tweets[uid], key=lambda k: k['tweet_time'] )
            users_tweets[uid] = users_tweets[uid][:max_tweets]

    for uid in users_tweets:
        if( len(users_tweets[uid]) < min_tweets ):
            continue

        final_users_tweets[uid] = users_tweets[uid]

    return final_users_tweets

def get_info_ops_drivers_control_users_tweets(driver_post_dates_dets, driver_file, campaign, args, bloc_model, gen_bloc_params):

    maximum_tweets_per_driver_and_control = 100000
    driver_control_users_file = get_driver_control_filename_mk_tweet_path(tweets_path=f'{args.tweets_path}{driver_file}')
        
    #get all tweets posted by driver and control for dates driver posted with hashtags (driver_post_dates_dets)
    driver_all_user_posts = get_driver_per_day_tweets( f'{args.tweets_path}{driver_file}', driver_post_dates_dets, start_datetime='', end_datetime='' )
    contro_all_user_posts = get_driver_control_per_day_tweets(driver_control_users_file, driver_post_dates_dets, start_datetime='', end_datetime='')

    driver_all_user_blocs = gen_bloc_for_user( get_all_per_user_tweets(driver_all_user_posts['driver_posts'], maximum_tweets_per_driver_and_control, args.min_tweets), gen_bloc_params, accnt_details={'src': campaign, 'class': 'driver'} )
    contro_all_user_blocs = gen_bloc_for_user( get_all_per_user_tweets(contro_all_user_posts['control_driver_posts'], maximum_tweets_per_driver_and_control, args.min_tweets), gen_bloc_params, accnt_details={'src': campaign, 'class': 'control'} )

    return {
        'driver_all_user_blocs': driver_all_user_blocs,
        'contro_all_user_blocs': contro_all_user_blocs
    }

def info_ops_study_drivers_vs_control_users(driver_post_dates_dets, driver_file, campaign, args, bloc_model, gen_bloc_params):

    storage_path = f'{args.outpath}/stable-results/{campaign}/'
    os.makedirs(f'{storage_path}/saved-tweets/', exist_ok=True)
    outfilename = f'{storage_path}/{campaign}_change_report.txt'
    outfile = open(outfilename, 'w')

    outfile.write('\ninfo_ops_study_drivers_vs_control_users():')
    #driver_post_dates: dates drivers posted with hashtags
    unfiltered_driver_post_dates = driver_post_dates_dets['post_dates']
    yyyy_cm_driver_count_map = driver_post_dates_dets['yyyy_cm_driver_count_map']

    
    stop_yyyy = ''
    driver_post_dates = []
    driver_post_dates_dict = {}


    #find the year with at least args.min_user_count drivers - start
    for d in unfiltered_driver_post_dates:
        
        yyyy = d.split('-')[0]
        if( stop_yyyy != '' and stop_yyyy != yyyy ):
            outfile.write(f'\treached new year ({yyyy}), breaking.')
            break

        driver_post_dates.append(d)
        driver_post_dates_dict[d] = True
        
        if( yyyy_cm_driver_count_map[yyyy] >= args.min_user_count ):
            outfile.write(f'\t{yyyy}, drivers:', yyyy_cm_driver_count_map[yyyy])
            stop_yyyy = yyyy
        
    driver_post_dates.sort()
    #find the year with at least args.min_user_count drivers - end


    maximum_tweets_per_driver_and_control = 100000
    driver_control_users_file = get_driver_control_filename_mk_tweet_path(tweets_path=f'{args.tweets_path}{driver_file}')
    
    outfile.write('\tgetting full timeline driver & control tweets from {} to {}\n'.format(driver_post_dates[0], driver_post_dates[-1]))
    #get all tweets posted by driver and control for dates driver posted with hashtags (driver_post_dates_dict)
    

    #experiment - start
    driver_all_user_posts = get_driver_per_day_tweets( f'{args.tweets_path}{driver_file}', driver_post_dates_dict, start_datetime='', end_datetime='' )
    #contro_all_user_posts = get_driver_control_per_day_tweets(driver_control_users_file, driver_post_dates_dict, start_datetime='', end_datetime='')

    driver_all_user_blocs = gen_bloc_for_user( get_all_per_user_tweets(driver_all_user_posts['driver_posts'], maximum_tweets_per_driver_and_control, args.min_tweets), gen_bloc_params, accnt_details={'src': campaign, 'class': 'driver'} )
    #contro_all_user_blocs = gen_bloc_for_user( get_all_per_user_tweets(contro_all_user_posts['control_driver_posts'], maximum_tweets_per_driver_and_control), gen_bloc_params, accnt_details={'src': campaign, 'class': 'control'} )
    
    driver_all_user_blocs = add_change_dynamics_to_bloc(driver_all_user_blocs)
    #contro_all_user_blocs = add_change_dynamics_to_bloc(contro_all_user_blocs)

    
    #'''
    bloc_vdl_model = get_bloc_vocab_doc_len_linear_model()
    driver_all_user_blocs = sorted( driver_all_user_blocs, key=lambda x: x['change_dynamics']['change_dynamics_score'], reverse=True )
    draw_driver_control_change_dynamics_v0(campaign=campaign, driver=driver_all_user_blocs, control=[], bloc_vdl_model=bloc_vdl_model, line=bloc_vdl_model, outfilename=f'{storage_path}{campaign}_heaps_plt.png')
    total_users = len(driver_all_user_blocs)
    
    for i in range(total_users):
        
        dbloc = driver_all_user_blocs[i]   
        #action_top_bloc_words = dbloc['change_dynamics']['action'].pop('top_bloc_words')
        #content_top_bloc_words = dbloc['change_dynamics']['content_syntactic'].pop('top_bloc_words')
        #dbloc: dict_keys(['bloc', 'tweets', 'bloc_segments', 'created_at_utc', 'screen_name', 'user_id', 'bloc_symbols_version', 'more_details', 'tweet_count', 'src', 'class', 'change_dynamics'])

        outfile.write('{} of {}: {}\n'.format(i+1, total_users, dbloc['screen_name']))
        outfile.write('user_id: {}\n'.format(dbloc['user_id']))
        outfile.write('total_tweets: {}\n'.format(dbloc['more_details']['total_tweets']))
        outfile.write('first_tweet_created_at_local_time: {}\n'.format(dbloc['more_details']['first_tweet_created_at_local_time']))
        outfile.write('last_tweet_created_at_local_time: {}\n'.format(dbloc['more_details']['last_tweet_created_at_local_time']))
        outfile.write('datediff: {}\n'.format(datetime.strptime(dbloc['more_details']['last_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S') -  datetime.strptime(dbloc['more_details']['first_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S')))
        
        for alph in dbloc['bloc']:
            
            outfile.write(f'top BLOC {alph} words\n')
            top_bloc_words = dbloc['change_dynamics'][alph].pop('top_bloc_words', [])
            
            for j in range(len(top_bloc_words)):
                w = top_bloc_words[j]
                outfile.write( '\t{}. {}: {:.2f}\n'.format(j+1, w['term'], w['term_rate']) )
                if( w['term_rate'] < 0.01 ):
                    break

        outfile.write(json.dumps(dbloc['change_dynamics'], ensure_ascii=False) + '\n')

        for alph, bloc_doc in dbloc['bloc'].items():
            
            outfile.write(f'\t{bloc_doc}\n')
            if( 'highlight' in dbloc and alph == 'action' ):
                study_info_ops_drivers_change_context_v1(dbloc, alph, outfile.write)
            outfile.write('\n')
            
        outfile.write('\n')
        outfile.write('\n')

        if( 'highlight' in dbloc ):
            stash_usr_tweets(dbloc['tweets'], '{}/saved-tweets/{}_{}_tweets.jsonl.gz'.format(storage_path, i+1, dbloc['screen_name']))
    
    #experiment - end
    print()
    print()
    print(f'wrote: {outfilename}')
    outfile.close()
    
    
def get_driver_control_per_day_tweets(driver_control_users_file, driver_post_dates_dict, start_datetime, end_datetime):
    
    all_drivers = set()
    all_tweets = []
    dv_ctrl_yyyy_mm_post_stats = {}

    driver_control_tweets = driver_control_users_file.replace('control_driver_users.csv', 'control_driver_tweets.jsonl.gz')
    print('\ndriver_control_tweets:', driver_control_tweets)
    with gzip.open(driver_control_tweets, 'rb') as f:
        for tweet in f:
            
            tweet = getDictFromJson(tweet)
            tweet = get_bloc_lite_twt_frm_full_twt(tweet)

            user_id = tweet['user']['id']
            tweet_id = tweet['id']
            tweet_time = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d %H:%M:%S')

            if( start_datetime != '' and end_datetime != '' ):
                if( tweet_time < start_datetime or tweet_time > end_datetime ):
                    continue

            yyyy_mm_dd = tweet_time.split(' ')[0]
            all_tweets_len = len(all_tweets)
            
            if( yyyy_mm_dd not in driver_post_dates_dict['post_dates'] ):
                continue

            all_drivers.add(user_id)
            tweet['tweet_time'] = tweet_time
            dv_ctrl_yyyy_mm_post_stats.setdefault( yyyy_mm_dd, {'tweet_ids': []} )
            dv_ctrl_yyyy_mm_post_stats[ yyyy_mm_dd ]['tweet_ids'].append( (tweet_id, all_tweets_len) )

            
            all_tweets.append( tweet )
    print('\tdone')
    return {
        'yyyy_mm_dd_post_stats': dv_ctrl_yyyy_mm_post_stats,
        'total_control': len(all_drivers),
        'control_driver_posts': all_tweets
    }

def get_driver_control_filename_mk_tweet_path(tweets_path):
    tweets_path = tweets_path.replace('driver_tweets.csv.gz', '')
    tweets_path = f'{tweets_path}DriversControl/'
    print('\ttweets_path:', tweets_path)

    os.makedirs( tweets_path, exist_ok=True )
    driver_control_users_file = f'{tweets_path}control_driver_users.csv'
    return driver_control_users_file




def rename_cols(all_datasets, src_map):

    print('\trename_cols():')

    for old_src, new_src in src_map:
        
        if( old_src not in all_datasets ):
            continue

        print(f'\trenaming {old_src} with {new_src}')
        all_datasets[new_src] = all_datasets.pop(old_src)

def merge_srcs(all_datasets, merge_lst, max_users):
    
    if( len(merge_lst) == 0 ):
        return
    
    print('\nmerge_srcs():')
    for i in range( len(merge_lst) ):
        
        src = merge_lst[i]['src']
        new_class = merge_lst[i]['new_class']

        if( src not in all_datasets ):
            continue

        print( '\tsrc:', src )
        print( '\tmerging all {} to {}'.format(merge_lst[i]['old_classes'], new_class) )
        print()

        new_cols = []
        new_size = max_users//len(merge_lst[i]['old_classes'])
        for c in merge_lst[i]['old_classes']:

            if( c not in all_datasets[src] ):
                continue

            new_cols += all_datasets[src][c][:new_size]
            del all_datasets[src][c]
        
        all_datasets[src][new_class] = new_cols

def add_more_details(all_datasets):

    for src, payload in all_datasets.items():
        for classs, users in payload.items():
            for i in range( len(users) ):

                users[i]['src'] = src
                users[i]['class'] = classs

                if( 'action_post_ref_time' in users[i]['bloc'] ):
                    del users[i]['bloc']['action_post_ref_time']

def flatten_dataset_shuffle(all_datasets):

    print('\nflatten_dataset_shuffle()')
    print('pre flatten report')

    min_class_count = {}
    users_per_class = {}
    for src, payload in all_datasets.items():
        print(f'\tsrc: {src}')
        for classs, users in payload.items():
            print(f'\t\tclass: {classs}', len(users), 'users')
            
            users_per_class[classs] = []
            min_class_count.setdefault(classs, 0)
            min_class_count[classs] += len(users)
    

    for src, payload in all_datasets.items():
        for classs, users in payload.items():
            users_per_class[classs] += users

    #watch
    min_class_count = sorted( min_class_count.items(), key=lambda item: item[1] )[0][1]
    #min_class_count = 100000000000000

    print('class counts:')
    print('min_class_count:', min_class_count)
    flat_ds = []
    for c in users_per_class:
        print('\t', c, len(users_per_class[c]), 'users')
        shuffle(users_per_class[c])
        flat_ds += users_per_class[c][:min_class_count]
        users_per_class[c] = 0

    
    print()
    print('final dist:')
    src_dist = {}
    for u in flat_ds:
        users_per_class[ u['class'] ] += 1
        src_dist.setdefault(u['src'], 0)
        src_dist[ u['src'] ] += 1
    
    print('users_per_class:', users_per_class)
    print('src_dist:')
    for src, cnt in src_dist.items():
        print('\tsrc:', src, cnt)

    return flat_ds

def get_bloc_for_info_ops_users(args, bloc_model, gen_bloc_params, start_datetime='', end_datetime='', **kwargs):
    
    print('\nget_bloc_for_info_ops_users()')
    print('\ttweets_files:', args.tweets_files)
    print('\toutpath:', args.outpath)

    date_ranges = kwargs.get('date_ranges', [])
    driver_file = args.tweets_files[0]
        
    campaign = driver_file.split('/')[-2]
    args.min_user_count = 999999999#10, 
    print(f'args.min_user_count coerced: {args.min_user_count}\n'*20)

    args.tweets_path = args.tweets_path.strip()
    args.tweets_path = args.tweets_path if ( args.tweets_path.endswith('/') or args.tweets_path == '' ) else args.tweets_path.strip() + '/'

    payload = get_driver_post_stats( f'{args.tweets_path}{driver_file}', start_datetime=start_datetime, end_datetime=end_datetime, date_ranges=date_ranges )
    driver_post_dates_dets = get_driver_post_date_dist( payload['driver_posts_stats_yyyy_mm_dd'], payload['total_posts'] )
    return get_info_ops_drivers_control_users_tweets(driver_post_dates_dets, driver_file, campaign, args, bloc_model, gen_bloc_params)


def get_bloc_for_human_bots(args, gen_bloc_params):

    print('\nget_bloc_for_human_bots()')
    merge_lst = [
        {
            'src': 'cresci-17',
            'new_class': 'bot',
            'old_classes': ['bot-fakefollower', 'bot-socialspam', 'bot-traditionspam']
        },
        {
            'src': 'zoher-organization',
            'new_class': 'human',
            'old_classes': ['celebrity', 'human']#don't include 'organizations' since orgs tend to use bots to run their accounts    
        },
        {
            'src': 'celebrity-19',
            'new_class': 'human',
            'old_classes': ['celebrity', 'human']#don't include '#organizations' since orgs tend to use bots to run their accounts   
        },
        {
            'src': 'astroturf',
            'new_class': 'bot',
            'old_classes': ['political_Bot']
        }
    ]

    src_maps = [
        ('kevin_feedback', 'botometer-feedback-19'),
        ('botwiki', 'botwiki-19'),
        ('caverlee', 'caverlee-11'),
        ('zoher-organization', 'celebrity-19'),
        ('rtbust', 'cresci-rtbust-19'),
        ('stock', 'cresci-stock-18'),
        ('midterm-2018', 'midterm-18'),
        ('josh_political', 'political-bots-19'),
        ('pronbots', 'pronbots-19'),
        ('varol-icwsm', 'varol-17'),
        ('gregory_purchased', 'vendor-purchased-19'),
        ('verified', 'verified-19')
    ]

    all_datasets = get_bloc_for_tweets( 
        tweets_files=args.tweets_files, 
        tweets_path=args.tweets_path, 
        gen_bloc_params=gen_bloc_params, 
        max_users=args.max_users, 
        min_tweets=args.min_tweets,#50 
        max_tweets=args.max_tweets, 
        src_maps=src_maps
    )

    #don't include '#organizations' since orgs tend to use bots to run their accounts   
    for ky, val in all_datasets.items():
        val.pop('organization', None)

    merge_srcs( all_datasets, merge_lst, args.max_users )
    rename_cols( all_datasets, src_maps )
    add_more_details( all_datasets )#must call this after merge_srcs and rename_cols
    all_datasets = flatten_dataset_shuffle( all_datasets )

    '''
    #this goes without flattening
    payload = {'bot': [], 'human': []}
    for src, dataset in all_datasets.items():
        
        payload['bot'] += dataset.get('bot', [])
        payload['human'] += dataset.get('human', [])
        
        print('\tbot:', len(payload['bot']))
        print('\thuman:', len(payload['human']))

    return payload
    '''

    return all_datasets

def get_chng_dynamics_score(x, method, **kwargs):

    if( method == get_bloc_entropy or method == slide_time_window_over_bloc_tweets_v2 ):
        
        score = 0
        act_metric_weight = 1
        con_metric_weight = 1

        all_bloc_chg_metrics = kwargs['all_bloc_chg_metrics']
        metric_type = 'word_entropy' if method == get_bloc_entropy else 'avg_cosine_dist'

        action_metric_config = all_bloc_chg_metrics['action'][metric_type]
        content_metric_config = all_bloc_chg_metrics['content_syntactic'][metric_type]

        action_metric = (x.get('action', {}).get(metric_type, -1) - action_metric_config['mean'])/action_metric_config['stdev']
        content_metric = (x.get('content_syntactic', {}).get(metric_type, -1) - content_metric_config['mean'])/content_metric_config['stdev']

        score += act_metric_weight * abs(action_metric)
        score += con_metric_weight * abs(content_metric)

        return score/(act_metric_weight + con_metric_weight)

    if( method == slide_time_window_over_bloc_tweets_v3 ):

        score = 0
        act_metric_weight = 1
        con_metric_weight = 1

        score += act_metric_weight * x.get('action', {}).get('avg_cosine_dist', -1)
        score += con_metric_weight * x.get('content_syntactic', {}).get('avg_cosine_dist', -1)
        
        return score/(act_metric_weight + con_metric_weight)

    if( method == 3 ):
        score = 0

        act_vocab_len_weight = 1
        con_vocab_len_weight = 1

        score += act_vocab_len_weight * x.get('action', {}).get('prediction_error', -1)
        score += con_vocab_len_weight * x.get('content_syntactic', {}).get('prediction_error', -1)

        return score/(act_vocab_len_weight + con_vocab_len_weight)
    
    if( method == 2 ):
    
        score = 0

        act_vocab_len_weight = 1
        con_vocab_len_weight = 1

        score += act_vocab_len_weight * x.get('action', {}).get('vocab_len', -1)
        score += con_vocab_len_weight * x.get('content_syntactic', {}).get('vocab_len', -1)

        return score/(act_vocab_len_weight + con_vocab_len_weight)

    if( method == 1 ):
        score = 0

        act_total_unique_words_weight = 1
        con_total_unique_words_weight = 1

        score += act_total_unique_words_weight * x.get('action', {}).get('total_unique_words', -1)
        score += con_total_unique_words_weight * x.get('content_syntactic', {}).get('total_unique_words', -1)

        return score/(act_total_unique_words_weight + con_total_unique_words_weight)

    if( method == 0 ):
        score = 0

        act_unique_words_weight = 0.50
        act_avg_cosine_dist_weight = 0.25
        
        con_unique_words_weight = 0.15
        con_avg_cosine_dist_weight = 0.10

        score += act_unique_words_weight * x.get('action', {}).get('unique_words', {}).get('mean', 0)
        score += act_avg_cosine_dist_weight * x.get('action', {}).get('avg_cosine_dist', -1)
        
        score += con_unique_words_weight * x.get('content_syntactic', {}).get('unique_words', {}).get('mean', 0)
        score += con_avg_cosine_dist_weight * x.get('content_syntactic', {}).get('avg_cosine_dist', -1)
        
        return score/(act_unique_words_weight + act_avg_cosine_dist_weight + con_unique_words_weight + con_avg_cosine_dist_weight)

def write_change_details(accnts, args):

    manual_highlight_user_ids = ['2361870980', '782316137296044032']
    storage_path = f'{args.outpath}/stable-results/human-accounts/'
    os.makedirs(f'{storage_path}/saved-tweets/', exist_ok=True)
    outfilename = f'{storage_path}/human_change_report.txt'
    
    outfile = open(outfilename, 'w')
    outfile.write('\nwrite_change_details():')

    accnts = sorted( accnts, key=lambda x: x['change_dynamics']['change_dynamics_score'], reverse=True )
    total_users = len(accnts)

    for i in range(total_users):

        dbloc = accnts[i] 

        outfile.write('{} of {}: {} {}\n'.format(i+1, total_users, dbloc['user_id'], dbloc['screen_name'] ))
        outfile.write('total_tweets: {}\n'.format(dbloc['more_details']['total_tweets']))
        outfile.write('first_tweet_created_at_local_time: {}\n'.format(dbloc['more_details']['first_tweet_created_at_local_time']))
        outfile.write('last_tweet_created_at_local_time: {}\n'.format(dbloc['more_details']['last_tweet_created_at_local_time']))
        outfile.write('datediff: {}\n'.format(datetime.strptime(dbloc['more_details']['last_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S') -  datetime.strptime(dbloc['more_details']['first_tweet_created_at_local_time'], '%Y-%m-%d %H:%M:%S')))
        
        for alph in dbloc['bloc']:
            
            outfile.write(f'top BLOC {alph} words\n')
            top_bloc_words = dbloc['change_dynamics'][alph].pop('top_bloc_words', [])
            
            for j in range(len(top_bloc_words)):
                w = top_bloc_words[j]
                outfile.write( '\t{}. {}: {:.2f}\n'.format(j+1, w['term'], w['term_rate']) )
                if( w['term_rate'] < 0.01 ):
                    break

        outfile.write(json.dumps(dbloc['change_dynamics'], ensure_ascii=False) + '\n')

        for alph, bloc_doc in dbloc['bloc'].items():
            
            outfile.write(f'\t{bloc_doc}\n')
            cust_highlight = 'h.' if dbloc['user_id'] in manual_highlight_user_ids else ''
            if( ('highlight' in dbloc or cust_highlight != '') and alph == 'action' ):
                study_info_ops_drivers_change_context_v1(dbloc, alph, outfile.write)
                stash_usr_tweets(dbloc['tweets'], '{}/saved-tweets/{}_{}_{}_tweets.jsonl.gz'.format(storage_path, cust_highlight+start_time.split('T')[0], i+1, dbloc['user_id']))
            outfile.write('\n')
                       
        outfile.write('\n')
        outfile.write('\n')  

    print(f'wrote: {outfilename}')
    outfile.close()

def main():
    
    parser = get_generic_args()
    args = parser.parse_args()    

    args.tweets_path = args.tweets_path.strip() if args.tweets_path.strip().endswith('/') else args.tweets_path.strip() + '/'
    params = vars(args)

    gen_bloc_params = {}
    for ky, val in params.items():
        if( ky.startswith('bc_') ):
            gen_bloc_params[ky[3:]] = val

    
    merge_lst = [
        {
            'src': 'cresci-17',
            'new_class': 'bot',
            'old_classes': ['bot-fakefollower', 'bot-socialspam', 'bot-traditionspam']
        },
        {
            'src': 'zoher-organization',
            'new_class': 'human',
            'old_classes': ['celebrity', 'human', 'organization']    
        },
        {
            'src': 'celebrity-19',
            'new_class': 'human',
            'old_classes': ['celebrity', 'human', 'organization']    
        },
        {
            'src': 'astroturf',
            'new_class': 'bot',
            'old_classes': ['political_Bot']
        }
    ]

    src_maps = [
        ('kevin_feedback', 'botometer-feedback-19'),
        ('botwiki', 'botwiki-19'),
        ('caverlee', 'caverlee-11'),
        ('zoher-organization', 'celebrity-19'),
        ('rtbust', 'cresci-rtbust-19'),
        ('stock', 'cresci-stock-18'),
        ('midterm-2018', 'midterm-18'),
        ('josh_political', 'political-bots-19'),
        ('pronbots', 'pronbots-19'),
        ('varol-icwsm', 'varol-17'),
        ('gregory_purchased', 'vendor-purchased-19'),
        ('verified', 'verified-19')
    ]

    parenth_flag = '' if ('action_content_syntactic' in args.bc_bloc_alphabets or 'content_syntactic_with_pauses' in args.bc_bloc_alphabets) else '()'
    pause_flag = '|[□⚀⚁⚂⚃⚄⚅.]' if args.add_pauses is True else ''
    token_pattern = '([^ |()*])' if args.ngram > 1 else f'[^□⚀⚁⚂⚃⚄⚅. |*{parenth_flag}]+{pause_flag}'

    bloc_model = {
        'ngram': args.ngram,
        'bloc_variant': None if args.ngram > 1 else {'type': 'folded_words', 'fold_start_count': args.bc_fold_start_count, 'count_applies_to_all_char': False},
        'bloc_alphabets': args.bc_bloc_alphabets,
        'token_pattern': token_pattern,
        'tf_matrix_norm': args.tf_matrix_norm,
        'keep_tf_matrix': args.keep_tf_matrix,
        'set_top_ngrams': False,
        'top_ngrams_add_all_docs': False
    }

    if( args.task == 'gen_vocab_doc_len_model' ):
        
        #for random size max_tweets = (50, 300), for no restriction max_tweets = -1
        max_users = 1000
        max_tweets = (50, 300)
        bot_human_accnts = get_bloc_for_human_bots(args, gen_bloc_params, max_users=max_users, max_tweets=max_tweets)
        bot_human_accnts['human'] = add_change_dynamics_to_bloc(bot_human_accnts['human'])
        draw_driver_control_change_dynamics_v0([], bot_human_accnts['human'])
        write_change_details(bot_human_accnts['human'], args)

        return
    elif( args.task == 'midterm_2022_study' ):
        midterm_2022_study(args, bloc_model, gen_bloc_params)
        return

    elif( args.task == 'change_ref_dates_study' ):

        if( 'midterm_2022' in args.tweets_files[-1] ):

            #time python ../scripts/bloc_change.py --start-time='2022-11-01T00:00:00' --end-time='2022-11-30T11:59:59' --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper --task change_ref_dates_study --study-dates="2022-11-08" --study-dates-seconds-offset=604800 --outpath=./ midterm_2022_v2.jsonl.gz
            in_filename = f'{args.tweets_path}{args.tweets_files[-1]}'
            
            users_tweets = get_midterm_2022_tweets(in_filename, start_time=args.start_time, end_time=args.end_time, max_tweets=args.max_tweets, min_tweets=args.min_tweets)
            print('final len(users_tweets):', len(users_tweets), f'max_tweets: {args.max_tweets}')

            print('generating BLOC for midterm - start')
            midterm_all_user_blocs = gen_bloc_for_user( users_tweets, gen_bloc_params, accnt_details={'src': 'midterm_2022', 'class': 'midterm'} )
            print('generating BLOC for midterm - end')
            print('len(midterm_all_user_blocs):', len(midterm_all_user_blocs))
            diversity_study( args, midterm_all_user_blocs, change_dynamics_method=get_bloc_entropy, **params )
        
        elif( 'driver_tweets.csv.gz' in args.tweets_files[-1] ):
            
            #python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM --task change_ref_dates_study --study-dates="2020-01-02" --study-dates-seconds-offset=604800 --outpath=./ 2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz

            #python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM --task change_ref_dates_study --study-dates="2019-08-15" --study-dates-seconds-offset=604800 --outpath=./ 2020_03/ghana_nigeria_032020/driver_tweets.csv.gz
            args.study_dates = args.study_dates.strip().split(',')
            args.study_dates = [ datetime.strptime(d, '%Y-%m-%d') for d in args.study_dates ]
            args.study_dates_ranges = [ (d - timedelta(seconds=args.study_dates_seconds_offset), d + timedelta(seconds=args.study_dates_seconds_offset)) for d in args.study_dates ]

            if( len(args.study_dates) == 0 ):
                print('Exiting: args.study_dates is empty, set --study-dates="comma-separated date (YYYY-MM-DD)"')
                sys.exit(0)

            change_ref_dates_study(args, bloc_model, gen_bloc_params)
        
        return

    elif( args.task == 'change_segments_study' ):

        if( 'driver_tweets.csv.gz' in args.tweets_files[-1] ):
            #python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM --task change_segments_study --outpath=./ 2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz > output.txt
            info_ops_accnts = get_bloc_for_info_ops_users(args, bloc_model, gen_bloc_params) 
            shuffle(info_ops_accnts['contro_all_user_blocs'])
            info_ops_accnts['contro_all_user_blocs'] = info_ops_accnts['contro_all_user_blocs'][:len(info_ops_accnts['driver_all_user_blocs'])]
            diversity_study( args, info_ops_accnts['driver_all_user_blocs'] + info_ops_accnts['contro_all_user_blocs'], change_dynamics_method=slide_time_window_over_bloc_tweets_v2, **params )
        else:
            #time python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/bot-detection/retraining_data --task change_segments_study --outpath=./ astroturf kevin_feedback botwiki zoher-organization cresci-17 rtbust stock gilani-17 midterm-2018 josh_political pronbots varol-icwsm gregory_purchased verified > output.txt
            
            #time python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/bot-detection/retraining_data --task change_segments_study --outpath=./ kevin_feedback

            bot_human_accnts = get_bloc_for_human_bots(args, gen_bloc_params)
            diversity_study( args, bot_human_accnts, change_dynamics_method=slide_time_window_over_bloc_tweets_v2, **params )

        return
    elif( args.task == 'diversity_study' ):
        
        if( 'midterm_2022' in args.tweets_files[-1] ):
            #test: time python ../scripts/bloc_change.py --start-time='2022-09-10T00:00:00' --end-time='2022-09-15T11:59:59' --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper --task diversity_study --outpath=./ midterm_2022_v2.jsonl.gz

            #time python ../scripts/bloc_change.py --start-time='2022-11-01T00:00:00' --end-time='2022-11-30T11:59:59' --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper --task diversity_study --outpath=./ midterm_2022_v2.jsonl.gz
            in_filename = f'{args.tweets_path}{args.tweets_files[-1]}'
            
            users_tweets = get_midterm_2022_tweets(in_filename, start_time=args.start_time, end_time=args.end_time, max_tweets=args.max_tweets, min_tweets=args.min_tweets)
            print('final len(users_tweets):', len(users_tweets), f'max_tweets: {args.max_tweets}')

            print('generating BLOC for midterm - start')
            midterm_all_user_blocs = gen_bloc_for_user( users_tweets, gen_bloc_params, accnt_details={'src': 'midterm_2022', 'class': 'midterm'} )
            print('generating BLOC for midterm - end')
            print('len(midterm_all_user_blocs):', len(midterm_all_user_blocs))
            diversity_study( args, midterm_all_user_blocs, change_dynamics_method=get_bloc_entropy, **params )

        elif( 'driver_tweets.csv.gz' in args.tweets_files[-1] ):
            #python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM --task diversity_study --outpath=./ 2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz > output.txt

            info_ops_accnts = get_bloc_for_info_ops_users(args, bloc_model, gen_bloc_params) 
            shuffle(info_ops_accnts['contro_all_user_blocs'])
            info_ops_accnts['contro_all_user_blocs'] = info_ops_accnts['contro_all_user_blocs'][:len(info_ops_accnts['driver_all_user_blocs'])]
            diversity_study( args, info_ops_accnts['driver_all_user_blocs'] + info_ops_accnts['contro_all_user_blocs'], change_dynamics_method=get_bloc_entropy, **params )
        else:
            
            #time python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/bot-detection/retraining_data --task diversity_study --outpath=./ astroturf kevin_feedback botwiki zoher-organization cresci-17 rtbust stock gilani-17 midterm-2018 josh_political pronbots varol-icwsm gregory_purchased verified > output.txt
            
            #time python ../scripts/bloc_change.py --max-users=500 --max-tweets=-1 --min-tweets=20 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=300 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/bot-detection/retraining_data --task diversity_study --outpath=./ kevin_feedback

            bot_human_accnts = get_bloc_for_human_bots(args, gen_bloc_params)
            diversity_study( args, bot_human_accnts, change_dynamics_method=get_bloc_entropy, **params )

        return
    elif( args.task == 'info_ops_study' ):

        #e.g., time python ../scripts/bloc_change.py --max-users=500 --action-change-mean=0.61 --action-change-stddev=0.30 --action-change-zscore-threshold=-1.3 --bc-keep-bloc-segments --bc-bloc-alphabets action --bc-keep-tweets --bc-segmentation-type=week_number --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM --task info_ops_study 2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz
        #python ../scripts/bloc_change.py --max-users=500 --action-change-mean=0.61 --action-change-stddev=0.30 --action-change-zscore-threshold=-1.3 --content-syntactic-change-mean=0.41 --content-syntactic-change-stddev=0.37 --content-syntactic-change-zscore-threshold=-1.0 --bc-gen-rt-content --bc-keep-tweets --bc-keep-bloc-segments --bc-bloc-alphabets action content_syntactic --bc-segmentation-type=segment_on_pauses --bc-segment-on-pauses=86400 --bc-tweet-order=sorted --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM --task info_ops_study 2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz
        info_ops_study(args, bloc_model, gen_bloc_params)
        return

    all_datasets = get_bloc_for_tweets( args.tweets_file, args.tweets_path, gen_bloc_params, max_users=args.max_users, min_tweets=args.min_tweets, max_tweets=args.max_tweets, src_maps=src_maps )
    for src in all_datasets:
        print('src:', src)
        for clss, tweets in all_datasets[src].items():
            print('\tclss:', clss, len(tweets))
    
    if( args.no_merge is False ):
        merge_srcs( all_datasets, merge_lst, args.max_users )
        
    rename_cols( all_datasets, src_maps )
    add_more_details( all_datasets)#must call this after merge_srcs and rename_cols
    all_datasets = flatten_dataset_shuffle( all_datasets )

    if( args.task.endswith('cosine_sim_dist') ):
        for comp_bloc_alph in args.bc_bloc_alphabets:
            
            args.change_mean = params.get(f'{comp_bloc_alph}_change_mean', None)
            args.change_stddev = params.get(f'{comp_bloc_alph}_change_stddev', None)
            args.change_zscore_threshold = params.get(f'{comp_bloc_alph}_change_zscore_threshold', None)
            
            calc_cosine_sim_dist(all_datasets, bloc_model, args, compute_change_alphabet=comp_bloc_alph)

if __name__ == "__main__":
    main()
