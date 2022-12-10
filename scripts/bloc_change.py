#import logging
import argparse
import csv
import gzip
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys
import time

from bloc.generator import add_bloc_sequences
from bloc.subcommands import bloc_change_usr_self_cmp
from bloc.util import dumpJsonToFile
from bloc.util import five_number_summary
from bloc.util import genericErrorInfo
from bloc.util import get_default_symbols
from bloc.util import getDictFromJson

from random import shuffle

#logging.basicConfig(format='', level=logging.INFO)
#logger = logging.getLogger(__name__)

def get_generic_args():

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), description='Evaluate Twitter DNA and DNA-influenced methods')
    parser.add_argument('tweets_files', nargs='+', help='Filename/Path containing tweets to process. If filename (see --tweets-path)')

    parser.add_argument('--add-pauses', action='store_true', help='BLOC generator (for words not bigram) use pause as separate words in vocabulary True/False. False is default')
    parser.add_argument('--bloc-model', default='word', choices=['bigram', 'word'], help='BLOC tokenization method.')

    parser.add_argument('--no-merge', action='store_true', help='Do not merge dataset variants (e.g., "bot-A" and "bot-B") of sources into single class (e.g, "bot").')
    parser.add_argument('-o', '--outpath', default=os.getcwd() + '/Output/', help='Output path')
    parser.add_argument('-t', '--task', default='pca_general', choices=['evaluate'], help='Task to run')
    parser.add_argument('--tweets-path', default='/scratch/anwala/IU/BLOC/botometer_retraining_data', help='The path to extract tweets for --tweets-files.')
    
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

    payload = {}
    all_bloc_symbols = get_default_symbols()
    
    for f in tweets_files:

        f = tweets_path + f + '/tweets.jsons.gz' if f.find('tweets') == -1 else f
        cf = '/'.join( f.split('/')[:-1] ) + '/userIds.txt'
        src = f.split('/')[-2]
        
        print('tweets file:', f)
        print('src:', src)
        
        if( os.path.exists(f) is False ):
            print('\ttweets file doesn\'t exist, returning')
            continue

        user_id_class_map, all_classes = get_user_id_class_map( cf )
        print('all_classes:', all_classes)
        if( len(user_id_class_map) == 0 ):
            print('\tuser_id_class_map is empty, returning')
            continue
        
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

                    user_class = user_id_class_map.get(line[0], '')
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

def run_tasks(bloc_collection, args):

    def draw_dist(summary_stats, dist, color):
        #bin_width = 2 * (summary_stats['q3'] - summary_stats['q1']) * (len(dist) ** (-1/3))
        #print('bin_width:', bin_width)

        ax = sns.histplot( dist, binwidth=0.05, color=color, alpha=0.5 ) 
        ax.set_xlabel ('Cosine similarity')
        ax.set_ylabel ('Number of accounts')
        
        #ax.set_title('(source: midwest)')

    parenth_flag = '' if ('action_content_syntactic' in args.bc_bloc_alphabets or 'content_syntactic_with_pauses' in args.bc_bloc_alphabets) else '()'
    pause_flag = '|[□⚀⚁⚂⚃⚄⚅]' if args.add_pauses is True else ''
    word_token_pattern = f'[^□⚀⚁⚂⚃⚄⚅ |*{parenth_flag}]+{pause_flag}'
    
    print(f'\nrun_task: {args.task}')
    print(f'\tbloc alphabets: {args.bc_bloc_alphabets}')
    print(f'\tword_token_pattern:', word_token_pattern)


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


    human_bot_dist = {'human': [], 'bot': []}
    for user_bloc in bloc_collection:
        
        user_change_report = bloc_change_usr_self_cmp(user_bloc, bloc_model, bloc_model['bloc_alphabets'], change_mean=args.change_mean, change_stddev=args.change_stddev, change_zscore_threshold=args.change_zscore_threshold)
        if( len(user_change_report['self_sim']['action']) == 0 ):
            continue

        sim_vals = [s['sim'] for s in user_change_report['self_sim']['action']]
        usr_class = user_bloc['class']#user_bloc['src']
        
        human_bot_dist[usr_class] += sim_vals

    
    human_sum = five_number_summary(human_bot_dist['human'])
    bot_sum = five_number_summary(human_bot_dist['bot'])

    print('Human summary')
    print(human_sum)
    print('Bot')
    print(bot_sum)

    draw_dist( human_sum, human_bot_dist['human'], color='green' )
    draw_dist( bot_sum, human_bot_dist['bot'], color='red' )
    plt.savefig('tmp.png', dpi=300)

    


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
    run_tasks(all_datasets, args)

if __name__ == "__main__":
    main()

