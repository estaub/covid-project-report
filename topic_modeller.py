import hashlib
from dataclasses import dataclass
from typing import Union

import gensim
import nltk
import numpy as np
import pandas as pd
from gensim import corpora, models
from langdetect import DetectorFactory, detect_langs
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from timer import Timer
from utils import mostRecentTsv

DetectorFactory.seed = 0

nltk.download('wordnet')
stemmer = SnowballStemmer('english')

# SOURCE COLUMNS in PROJECT dataframe
COL_github_repo_url = 'github_repo_url'
COL_repo_description = 'repo_description'
COL_topics = 'topics'
COL_owner_repo_name = 'owner_repo_name'
COL_owner_name = 'owner_name'
COL_owner_type = 'owner_type'
COL_organization_bio = 'organization_bio'
COL_repo_created_day = 'repo_created_day'
COL_primary_language_name = 'primary_language_name'
COL_license_name = 'license_name'
COL_is_github_pages = 'is_github_pages'
COL_has_readme = 'has_readme'
COL_has_wiki = 'has_wiki'
COL_has_merged_prs = 'has_merged_prs'
COL_has_issues = 'has_issues'
COL_has_contributor_guide = 'has_contributor_guide'
COL_has_code_of_conduct = 'has_code_of_conduct'
COL_count_of_public_forks = 'count_of_public_forks'
COL_count_of_stars = 'count_of_stars'
COL_count_of_watchers = 'count_of_watchers'
COL_count_distinct_contributors = 'count_distinct_contributors'
COL_count_contributions = 'count_contributions'
COL_count_commits = 'count_commits'
COL_count_commit_comments = 'count_commit_comments'
COL_count_created_issues = 'count_created_issues'
COL_count_pull_requests_created = 'count_pull_requests_created'
COL_count_pull_requests_reviews = 'count_pull_requests_reviews'
COL_count_comments_on_issues_and_pull_requests = 'count_comments_on_issues_and_pull_requests'
# COMPUTED COLUMNS on PROJECT DATAFRAME
COL_hash = 'hash'
COL_lang = 'lang'
COL_tokens = 'tokens'


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


stopwords = set(list(gensim.parsing.preprocessing.STOPWORDS) + [
    'covid', 'corona', 'virus', 'coronavirus', 'ncov', 'project', 'data', 'http', 'https',
    'github', 'pandemic', 'sar', 'challeng', 'inform','info'
])

replacements = {
    'tracker': 'track'
}


def tokenize(row: pd.Series) -> [str]:
    if row[COL_tokens]:
        return row[COL_tokens]
    result = []

    def split(text: str):
        for token in gensim.utils.simple_preprocess(text):
            token = replacements.get(token, token)
            if token not in stopwords and len(token) > 3:
                result.append(lemmatize_stemming(token))

    split(row[COL_repo_description])
    bigrams = [a + '_' + b for a, b in nltk.bigrams(result)]
    split(row[COL_owner_repo_name])
    split(row[COL_topics])
    return result + bigrams


def lang_detect_row(row: pd.Series):
    if row[COL_lang]:
        return row[COL_lang]
    desc = row[COL_repo_description]
    try:
        lang_pairs = None
        if desc:
            lang_pairs = detect_langs(desc)
    except Exception as e:
        lang_pairs = None
    if lang_pairs and len(lang_pairs) and lang_pairs[0].prob > .66:
        lang = lang_pairs[0].lang
    else:
        lang = '__'
    return lang


# @dataclass
# class SeedDoc:
#     name: str
#     weight: float
#     words: [str]
#
#     def make_seed(self) -> [[str]]:
#         return [self.words]
#
#
# seeds = [
#     SeedDoc('junk', 1, ['junk', 'garbage', 'flotsam', 'jetsam']),
#     SeedDoc('statistics', 1, ['statist', 'covidstat', 'data', ]),
# ]


## cache design
# pickle of a format version-number and a dataframe containing:
# index: repo url
# hash of source columns: df.apply(lambda x: hash(tuple(x)), axis = 1)
# computed columns: token-list, natural-language
@dataclass
class CacheEntry:
    source_hash: int
    tokens: [str]
    lang: Union[str, None]


def hash_project_source(project):
    src_hash = project[COL_hash]
    if src_hash:
        return src_hash
    hasher = hashlib.md5()

    hashee = project[COL_repo_description] + '/' + project[COL_primary_language_name] + '/' + \
             project[COL_github_repo_url]
    hasher.update(hashee.encode('utf-8'))
    ret = hasher.hexdigest()
    return ret


cache_filename = 'cache.zip'


def write_cache(projects):
    projects[COL_hash] = projects.apply(hash_project_source, axis=1)
    cache_df = pd.DataFrame(data={COL_lang: projects[COL_lang],
                                  COL_tokens: projects[COL_tokens],
                                  COL_hash: projects[COL_hash]},
                            index=projects[COL_github_repo_url])
    cache_df.to_pickle(cache_filename)


def read_cache(projects):
    def do_hashes_match(cache_row):

        new_hash = projects[COL_hash].get(cache_row.name)  # get returns None if missing
        return new_hash == cache_row[COL_hash]

    try:
        with Timer('computing hashes'):
            projects[COL_hash] = projects.apply(hash_project_source, axis=1)
        with Timer('reading cache'):
            cache_df = pd.read_pickle(cache_filename)
        with Timer('filtering cache for hash matches'):
            matching_cache_df = cache_df[cache_df.apply(do_hashes_match, axis=1)]
        with Timer('merging cache into projects'):
            projects.update(matching_cache_df)
    except Exception as e:
        # printf('read_cache exception')
        print('read_cache exception: ' + str(e))


@dataclass
class TopicDef():
    name: str
    words: [str]
    desc: str = ''
    group: str = ''


ntopics = 20

topic_defs = [
    TopicDef('mask', ['mask', 'face', 'face_mask', 'wear', 'wear_mask','shield','face_shield']),
    TopicDef('social-distancing', ['social_distanc']),
    TopicDef('mobile', ['mobil', 'android','flutter','nativ']),
    TopicDef('hospital', ['ventil', 'hospit', 'patient','doctor']),
    TopicDef('case', ['case']),
    TopicDef('india', ['india']),
    TopicDef('brazil', ['brasil']),
    # TopicDef('usa', ['usa']),
    TopicDef('contact-tracing', ['contact_trace', 'trace', 'contact']),
    TopicDef('visualization', ['chart', 'dashboard', 'visual']),
    TopicDef('death',['death']),
    TopicDef('prediction',['forecast', 'predict']),
]


def create_eta(topic_defs: [TopicDef], etadict: corpora.Dictionary, ntopics: int) -> np.ndarray:
    # create a (ntopics, nterms) matrix and fill with 1
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)
    for topic_idx, topic_def in enumerate(topic_defs):  # for each word in the list of priors
        for word in topic_def.words:
            keyindex = [index for index, term in etadict.items() if
                        term == word]  # find word in dict
            if (len(keyindex) > 0):  # if it's in the dictionary
                eta[topic_idx, keyindex[0]] = 1e7  # put a large number in there
            else:
                print(
                    f'create_eta: word "{word}" of topic {topic_def.name} not found in dictionary')
    eta = np.divide(eta, eta.sum(axis=0))  # normalize so probabilities sum to 1 over all topics
    return eta

# return doc freq, word freq, and sorted BoW of words in same docs as this one.
def get_neighbors(word: str, dict: corpora.Dictionary, bow_corpus: [[int, int]]) -> [str, int]:

    word_idx = dict.token2id.get(word, -1)
    if word_idx < 0:
        print('get_neighbors: word not in dictionary: ' + word)
        return 0,0,[]
    neighbor_dict={}
    for bow in bow_corpus:
        if next(((widx,_) for widx,_ in bow if widx==word_idx), False):
            # doc has our word
            for widx_other,_ in bow:
                if widx_other != word_idx:
                    neighbor_dict[widx_other] = neighbor_dict.get(widx_other, 0) + 1
    neighbor_bow = sorted(neighbor_dict.items(), key=lambda x: x[1], reverse=True)
    ret = [(dict[widx_neighbor], c) for widx_neighbor, c in neighbor_bow]
    return dict.dfs[word_idx], dict.cfs[word_idx], ret

def report_topic_words(topic_defs, dictionary, bow_corpus):
    print('TOPIC WORD REPORT')
    for topic_def in topic_defs:
        print('  TOPIC  ' + topic_def.name)
        for word in topic_def.words:
            dfs, cfs, neighbor_bow = get_neighbors(word, dictionary, bow_corpus)
            neighbor_bow = neighbor_bow[:10]
            print(f'    {word}: dfs={dfs} cfs={cfs} {str(neighbor_bow)}')


def lsa():
    with Timer('read tsv'):
        tsv_path = mostRecentTsv()
        projects = pd.read_csv(tsv_path, sep='\t', index_col=COL_github_repo_url)

        projects[COL_lang] = ''
        projects[COL_tokens] = ''
        projects[COL_hash] = ''
        projects[COL_github_repo_url] = projects.index
        projects.fillna(value='', inplace=True)

    print('' + str(len(projects)) + ' repos')

    read_cache(projects)

    has_empty_lang = '' in projects[COL_lang].values
    has_empty_tokens = '' in projects[COL_tokens].values
    cache_needs_writing = has_empty_lang or has_empty_tokens

    if has_empty_lang:
        with Timer('detecting language'):
            langs = projects.apply(lang_detect_row, axis=1)
            projects[COL_lang] = langs

    if has_empty_tokens:
        with Timer('preprocessing tokens'):
            tokens = projects.apply(tokenize, axis=1)
            projects[COL_tokens] = tokens

    if cache_needs_writing:
        with Timer('writing cache'):
            write_cache(projects)

    with Timer('analyzing topics'):
        # todo extract LDA function
        inputs = projects[COL_tokens]

        dictionary = gensim.corpora.Dictionary(inputs)

        # def probe_dict(word: str):
        #     idx = dictionary.token2id.get(word, None)
        #     coll_freq = dictionary.cfs.get(idx, None)
        #     doc_freq = dictionary.dfs.get(idx,None)
        #     return f'probe: word={word} idx={idx} cfs={coll_freq} dfs={doc_freq}'
        #
        # print('before filter')
        # print(probe_dict('case'))
        dictionary.filter_extremes(no_below=50, no_above=0.5, keep_n=10000)
        if len(dictionary) == 0:
            print('No words in dictionary')
            return
        dictionary[len(dictionary) - 1]  # DO NOT REMOVE! HACK TO INIT id2token
        dictFrame = pd.DataFrame({'token': dictionary.id2token})
        freqs = dictFrame.index.map(lambda i: dictionary.cfs.get(i, 0))
        dictFrame['freq'] = freqs
        print(dictFrame.head())
        freqFrame = dictFrame.sort_values(by='freq', ascending=False)
        print(freqFrame.head())
        tokenFrame = dictFrame.sort_values(by='token')
        print(tokenFrame.head())
        bow_corpus = [dictionary.doc2bow(doc) for doc in inputs]

        report_topic_words(topic_defs, dictionary, bow_corpus)
        # tfidf = models.TfidfModel(bow_corpus)
        # corpus_tfidf = tfidf[bow_corpus]

        # create seed matrix - from https://gist.github.com/scign/2dda76c292ef76943e0cd9ff8d5a174a
        eta = create_eta(topic_defs, dictionary, ntopics)
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=ntopics, eta=eta,
                                               id2word=dictionary,
                                               passes=2,
                                               chunksize=100000, workers=3)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    return lda_model


print(lsa())

# ['websit' 'hopkin_univers' 'descript' 'italia' 'case_death' 'notif', 'interact_visual' 'deploy' 'exposur' 'map' 'close' 'dato' 'page' nan, 'updat' 'confirm_case' 'finder' 'statist_countri' 'hopkin' 'protocol', 'research_dataset' 'tree' 'doc' 'visualis' 'so
