#import logging
import argparse
import csv
import gzip
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import seaborn as sns
import sys
import time

from datetime import datetime

from bloc.generator import add_bloc_sequences
from bloc.subcommands import bloc_change_usr_self_cmp
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

from itertools import combinations
from random import shuffle
from scipy.stats import ks_2samp

#logging.basicConfig(format='', level=logging.INFO)
#logger = logging.getLogger(__name__)
plt.rcParams.update({"font.family": 'monospace'})

def get_generic_args():

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), description='Evaluate Twitter DNA and DNA-influenced methods')
    parser.add_argument('tweets_files', nargs='+', help='Filename/Path containing tweets to process. If filename (see --tweets-path)')

    parser.add_argument('--add-pauses', action='store_true', help='BLOC generator (for words not bigram) use pause as separate words in vocabulary True/False. False is default')
    parser.add_argument('--bloc-model', default='word', choices=['bigram', 'word'], help='BLOC tokenization method.')

    parser.add_argument('--no-merge', action='store_true', help='Do not merge dataset variants (e.g., "bot-A" and "bot-B") of sources into single class (e.g, "bot").')
    parser.add_argument('-o', '--outpath', default=os.getcwd() + '/Output/', help='Output path')
    parser.add_argument('-t', '--task', default='bot_human_cosine_sim_dist', choices=['bot_human_cosine_sim_dist', 'info_ops_study', 'cosine_sim_dist'], help='Task to run')
    parser.add_argument('--tweets-path', default='/scratch/anwala/IU/BLOC/botometer_retraining_data', help='The path to extract tweets for --tweets-files.')
    #parser.add_argument('--tweets-path-info-ops', default='/scratch/anwala/IU/BLOC/InfoOps/YYYY_MM', help='Drivers: the path to extract tweets for --tweets-files.')
    
    #max parameters
    parser.add_argument('-m', '--max-tweets', type=int, default=-1, help='Maximum number of tweets per user to consider')
    parser.add_argument('-n', '--min-tweets', type=int, default=20, help='Mininum number of tweets per user to consider')
    parser.add_argument('-u', '--max-users', type=int, default=-1, help='Maximum number of users per class to process')
    
    #BLOC parameters
    parser.add_argument('--bc-blank-mark', type=int, default=60, help="add_bloc_sequences()'s blank_mark")
    parser.add_argument('--bc-bloc-alphabets', nargs='+', default=['action', 'content_syntactic'], choices=['action', 'content_syntactic', 'content_syntactic_with_pauses', 'change', 'action_content_syntactic'], help='add_bloc_sequences()\'s BLOC alphabets to extract')
    parser.add_argument('--bc-days-segment-count', type=int, default=-1, help="add_bloc_sequences()'s days_segment_count, if > 0, segmentation_type is set to day_of_year_bin")
    parser.add_argument('--bc-fold-start-count', type=int, default=10, help="add_bloc_sequences()'s change alphabet fold_start_count")
    parser.add_argument('--bc-gen-rt-content', action='store_true', help="add_bloc_sequences()'s gen_rt_content")
    parser.add_argument('--bc-keep-bloc-segments', action='store_true', help="add_bloc_sequences()'s keep_bloc_segments")
    parser.add_argument('--bc-keep-tweets', action='store_true', help="add_bloc_sequences()'s keep_tweets")
    parser.add_argument('--bc-minute-mark', type=int, default=5, help="add_bloc_sequences()'s minute_mark")
    parser.add_argument('--bc-segmentation-type', default='week_number', choices=['week_number', 'day_of_year'], help="add_bloc_sequences()'s segmentation_type")
    parser.add_argument('--bc-sort-action-words', action='store_true', help="add_bloc_sequences()'s sort_action_words")
    
    #change/vector parameters
    parser.add_argument('--account-class', default='', help='Class labels (e.g., bot or cyborgs or humans) of accounts')
    parser.add_argument('--account-src', default='', help='Origin of accounts')
    parser.add_argument('--change-mean', type=float, help='Empirical mean cosine similarity across BLOC segments.')
    parser.add_argument('--change-stddev', type=float, help='Empirical standard deviation across BLOC segments.')
    parser.add_argument('--change-zscore-threshold', type=float, default=1.5, help='Number of standard deviations (z-score) a similarity value has to exceed to be considered significant.')
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
            with open(cf) as fd:
                
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

                    user_class = user_id_class_map.get(line[0], '') if user_class == '' else user_class
                    if( user_class == '' ):
                        continue
                    
                    tweets = getDictFromJson( line[1] )
                    if( len(tweets) < min_tweets ):
                        continue
                    
                    if( is_all_class_full(users_tweets, max_users) ):
                        break

                    count = len( users_tweets[user_class] )
                    if( count == max_users ):
                        continue

                    tweets = tweets if max_tweets == -1 else tweets[:max_tweets]
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

def study_info_ops_drivers(drivers_change_dates, bloc_collection, store_path, legend_title, **kwargs):
    
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
        
        if( len(chrt_dist) == 0 ):
            return

        counter = 0
        lg_lines = []

        all_classes = list(chrt_dist.keys())
        all_classes.sort()
        dist_slug = '_'.join(all_classes)
        slug_prefix = '' if legend_title == '' else f'{legend_title}_'
        
        driver_count = ''

        for usr_class in all_classes:

            usr_class_raw_dist_outfile = f'{store_path}{slug_prefix}{dist_slug}_{chart_type}_{usr_class}.txt'
            '''
            outfile = open(usr_class_raw_dist_outfile, 'w')
            for v in chrt_dist[usr_class]:
                outfile.write(f'{v}\n')
            outfile.close()
            '''
            print(f'skipped writing: {usr_class_raw_dist_outfile}')
            
            draw_ccdfdist( chrt_dist[usr_class], color=styles[counter]['color'], linestyle=styles[counter]['linestyle'], xlabel=styles[counter].get('xlabel', ''), ylabel=styles[counter].get('ylabel', ''), title=styles[counter].get('title', '') + stat_sig )
            lg_lines.append( mlines.Line2D([], [], color=styles[counter]['color'], label='{} ({:,})'.format(usr_class.capitalize(), len(chrt_dist[usr_class])), linestyle=styles[counter]['linestyle']) )
            driver_count = ' ({:,}) '.format(len(chrt_dist[usr_class])) if usr_class == 'driver' else ' '
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

    print('\ncalc_cosine_sim_dist():')
    print(f'\nrun_task: {args.task}')
    print(f'\tbloc alphabets: {args.bc_bloc_alphabets}')
    print(f'\tword_token_pattern:', bloc_model['token_pattern'])

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
        if( len(user_change_report['self_sim'][compute_change_alphabet]) == 0 ):
            continue
        
        '''
        if( len(user_change_report['self_sim'][compute_change_alphabet]) > 3 and [1 for sm in user_change_report['self_sim'][compute_change_alphabet] if 'changed' in sm].count(1) != 0 ):
            dumpJsonToFile('tmp_bloc.json', user_bloc)
            dumpJsonToFile('tmp_change.json', user_change_report)
            sys.exit(0)
        '''

        #co-change graph - start
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
        #'''
        #if( change_rate >= 0.5 ):
        print('**change**')
        print('change_rate:', change_rate)
        print('screen_name:', user_bloc['screen_name'], usr_class)
        #print('change_dates:', change_dates)
        print(user_bloc['bloc'][compute_change_alphabet])
        print()
        #'''
    
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
        
        draw_change_dist(change_dist, styles, 'change_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig)
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
        draw_change_profile_dist(pause_profile_dist, [styles[0], styles[1]], 'change_profile_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, lg_lines=lg_lines)
        draw_change_profile_dist(word_profile_dist, [styles[2], styles[3]], 'change_profile_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, lg_lines=lg_lines)
        draw_change_profile_dist(activity_profile_dist, [styles[4], styles[5]], 'change_profile_dist', store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, stat_sig=stat_sig, lg_lines=lg_lines, draw_plot=True)

        for agent in ['driver', 'control', 'human', 'bot']:
            if( info_ops_dets[agent]['total'] == 0 ):
                continue

            study_info_ops_drivers(info_ops_dets[agent]['dates_change_occurred'], bloc_collection, store_path=f'./change-dists/{compute_change_alphabet}/', legend_title=legend_title, info_ops_dets=info_ops_dets[agent], agent=agent)

            info_ops_dets[agent]['dates_change_dist'] = sorted( info_ops_dets[agent]['dates_change_dist'].items(), key=lambda x: x[1], reverse=True )
            info_ops_dets[agent]['dates_change_dist'] = [ 
                {'date': date_freq_tup[0], 'freq': date_freq_tup[1], 'rate': date_freq_tup[1]/info_ops_dets[agent]['dates_change_dist_total']} for date_freq_tup in info_ops_dets[agent]['dates_change_dist'] ]
            dist_dist_fname = '{}{}_{}_dist_change_dates.json'.format(f'./change-dists/{compute_change_alphabet}/', legend_title, agent)
            
            dumpJsonToFile(dist_dist_fname, info_ops_dets[agent]['dates_change_dist'])
            print('Saved:', dist_dist_fname)

def info_ops_study(args, bloc_model, gen_bloc_params):
    print('\ninfo_ops_study()')

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
    all_tweets_files = ['2019_08/china_082019_2/driver_tweets.csv.gz']
    for driver_file in all_tweets_files:
    #for driver_file in args.tweets_files:
        
        #------------------------------------#
        #------------------------------------#
        campaign = driver_file.split('/')[-2]
        args.min_user_count = 999999999#10, 

        args.tweets_path = args.tweets_path.strip()
        args.tweets_path = args.tweets_path if ( args.tweets_path.endswith('/') or args.tweets_path == '' ) else args.tweets_path.strip() + '/'

        payload = get_driver_post_stats( f'{args.tweets_path}{driver_file}', start_datetime='', end_datetime='' )
        driver_post_dates_dets = get_driver_post_date_dist( payload['driver_posts_stats_yyyy_mm_dd'], payload['total_posts'] )
        info_ops_study_drivers_vs_control_users(driver_post_dates_dets, driver_file, campaign, args, bloc_model, gen_bloc_params)

        #print('debug break')
        #break

def gen_bloc_for_user(users, gen_bloc_params):

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
        ou['output']['tweet_count'] = len(ou['input']['args']['tweets'])
        all_user_blocs.append( ou['output'] )

    print('\ngen_bloc_for_user(): DONE add_bloc_sequences() for', len(job_lst), 'users')
    return all_user_blocs

def info_ops_study_drivers_vs_control_users(driver_post_dates_dets, driver_file, campaign, args, bloc_model, gen_bloc_params):

    def get_all_per_user_tweets(all_tweets, max_tweets):

        users_tweets = {}
        
        for twt in all_tweets:
            uid = twt['user']['id']
            users_tweets.setdefault(uid, [])
            users_tweets[uid].append( twt )

        for uid in users_tweets:
            if( len(users_tweets[uid]) > max_tweets ):
                users_tweets[uid] = sorted( users_tweets[uid], key=lambda k: k['tweet_time'] )
                users_tweets[uid] = users_tweets[uid][:max_tweets]

        return users_tweets

    
    print('\ninfo_ops_study_drivers_vs_control_users():')
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
            print(f'\treached new year ({yyyy}), breaking.')
            break

        driver_post_dates.append(d)
        driver_post_dates_dict[d] = True
        
        if( yyyy_cm_driver_count_map[yyyy] >= args.min_user_count ):
            print(f'\t{yyyy}, drivers:', yyyy_cm_driver_count_map[yyyy])
            stop_yyyy = yyyy
        


    driver_post_dates.sort()
    #find the year with at least args.min_user_count drivers - end


    maximum_tweets_per_driver_and_control = 100
    driver_control_users_file = get_driver_control_filename_mk_tweet_path(tweets_path=f'{args.tweets_path}{driver_file}')
    
    print('\tgetting full timeline driver & control tweets from {} to {}'.format(driver_post_dates[0], driver_post_dates[-1]))
    #get all tweets posted by driver and control for dates driver posted with hashtags (driver_post_dates_dict)
    driver_all_user_posts = get_driver_per_day_tweets( f'{args.tweets_path}{driver_file}', driver_post_dates_dict, start_datetime='', end_datetime='' )
    contro_all_user_posts = get_driver_control_per_day_tweets(driver_control_users_file, driver_post_dates_dict, start_datetime='', end_datetime='')
    

    print('\tdone getting full timeline driver & control tweets from {} to {}'.format(driver_post_dates[0], driver_post_dates[-1]))
    print('\n\tdriver/control statistics - start')
    print('\t\ntotal number of active days:', len(driver_post_dates))
    print('\t\ttotal drivers:', driver_all_user_posts['total_drivers'])
    print('\t\ttotal control:', contro_all_user_posts['total_control'])
    print()
    print('\t\tgross driver tweet count:', len(driver_all_user_posts['driver_posts']))
    print('\t\tgross control tweet count:', len(contro_all_user_posts['control_driver_posts']))
    print('\tdriver/control statistics - end\n')
    print()

    gross_stats = {
        'date_rng': [unfiltered_driver_post_dates[0], unfiltered_driver_post_dates[-1]],
        'driver_count': driver_all_user_posts['total_drivers'],
        'driver_control_count': contro_all_user_posts['total_control'],
        'driver_total_posts': len(driver_all_user_posts['driver_posts']),
        'driver_control_total_posts': len(contro_all_user_posts['control_driver_posts'])
    }
        
    driver_all_user_blocs = gen_bloc_for_user( get_all_per_user_tweets(driver_all_user_posts['driver_posts'], maximum_tweets_per_driver_and_control), gen_bloc_params )
    contro_all_user_blocs = gen_bloc_for_user( get_all_per_user_tweets(contro_all_user_posts['control_driver_posts'], maximum_tweets_per_driver_and_control), gen_bloc_params )

    for i in range(len(driver_all_user_blocs)):
        driver_all_user_blocs[i]['src'] = campaign
        driver_all_user_blocs[i]['class'] = 'driver'

    for i in range(len(contro_all_user_blocs)):
        contro_all_user_blocs[i]['src'] = campaign
        contro_all_user_blocs[i]['class'] = 'control'

    
    #driver_all_user_blocs[0]: dict_keys(['bloc', 'tweets', 'bloc_segments', 'created_at_utc', 'screen_name', 'user_id', 'bloc_symbols_version', 'more_details', 'tweet_count'])
    calc_cosine_sim_dist(driver_all_user_blocs + contro_all_user_blocs, bloc_model, args, legend_title=f'{campaign}', campaign_timerange=f'{unfiltered_driver_post_dates[0]} to {unfiltered_driver_post_dates[-1]}')
    

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
            
            if( yyyy_mm_dd not in driver_post_dates_dict ):
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

    min_class_count = sorted( min_class_count.items(), key=lambda item: item[1] )[0][1]
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
            'old_classes': ['celebrity', 'organization']    
        },
        {
            'src': 'celebrity-19',
            'new_class': 'human',
            'old_classes': ['celebrity', 'organization']    
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
    pause_flag = '|[□⚀⚁⚂⚃⚄⚅]' if args.add_pauses is True else ''
    word_token_pattern = f'[^□⚀⚁⚂⚃⚄⚅ |*{parenth_flag}]+{pause_flag}'

    bloc_model = {
        'ngram': args.ngram,
        'bloc_variant': {'type': 'folded_words', 'fold_start_count': args.bc_fold_start_count, 'count_applies_to_all_char': False},
        'bloc_alphabets': args.bc_bloc_alphabets,
        'token_pattern': word_token_pattern,
        'tf_matrix_norm': args.tf_matrix_norm,
        'keep_tf_matrix': args.keep_tf_matrix,
        'set_top_ngrams': False,
        'top_ngrams_add_all_docs': False
    }

    
    if( args.task == 'info_ops_study' ):
        #e.g., time python ../scripts/bloc_change.py --max-users=500 --change-mean=0.61 --change-stddev=0.30 --change-zscore-threshold=1.3 --bc-keep-bloc-segments --bc-bloc-alphabets action --add-pauses --bloc-model=word --tweets-path=/scratch/anwala/IU/BLOC/bloc-intro-paper/info-ops-driver-v-control/YYYY_MM/ --task info_ops_study 2021_12/MX_0621_YYYY/MX_0621_2020/driver_tweets.csv.gz
        info_ops_study(args, bloc_model, gen_bloc_params)
        return

    all_datasets = get_bloc_for_tweets( args.tweets_files, args.tweets_path, gen_bloc_params, max_users=args.max_users, min_tweets=args.min_tweets, max_tweets=args.max_tweets, src_maps=src_maps )
    
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
            calc_cosine_sim_dist(all_datasets, bloc_model, args, compute_change_alphabet=comp_bloc_alph)

if __name__ == "__main__":
    main()
