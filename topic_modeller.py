from dataclasses import dataclass
from typing import Union

import gensim
import nltk
import pandas as pd
from gensim import corpora, models
from langdetect import DetectorFactory, detect_langs
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import hashlib
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
    'github', 'pandemic', 'sar'
])

replacements = {
    'tracker': 'track'
}


def preprocess(row):
    if row[COL_hash]:
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


def lang_detect_row(row):
    if row[COL_hash]:
        return row[COL_lang]
    desc = row[COL_repo_description]
    try:
        lang_pairs = detect_langs(desc) if desc else None
    except Exception as e:
        lang_pairs = None
    if lang_pairs and len(lang_pairs) and lang_pairs[0].prob > .66:
        lang = lang_pairs[0].lang
    else:
        lang = None
    return lang


def dummy(row):
    desc = row[COL_repo_description]
    return 'dummy'


@dataclass
class SeedDoc:
    name: str
    weight: float
    words: [str]

    def make_seed(self) -> [[str]]:
        return [self.words]


seeds = [
    SeedDoc('junk', 1, ['junk', 'garbage', 'flotsam', 'jetsam']),
    SeedDoc('statistics', 1, ['statist', 'covidstat', 'data', ]),
]


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

    hashee = project[COL_repo_description]+'/'+project[COL_primary_language_name]+'/'+project[COL_github_repo_url]
    hasher.update(hashee.encode('utf-8'))
    ret = hasher.hexdigest()
    return ret


'''
('novel coronavirus (covid-19) cases, provided by jhu csse', '', 'https://github.com/CSSEGISandData/COVID-19')
('novel coronavirus (covid-19) cases, provided by jhu csse', '', 'https://github.com/CSSEGISandData/COVID-19')
7225982222729356966
'''
cache_filename = 'cache.zip'


def write_cache(projects):
    projects[COL_hash] = projects.apply(hash_project_source, axis=1)
    cache_df = pd.DataFrame(data={COL_lang: projects[COL_lang],
                                  COL_tokens: projects[COL_tokens],
                                  COL_hash: projects[COL_hash]},
                            index=projects[COL_github_repo_url])
    cache_df.to_pickle(cache_filename)


def read_cache(projects):
    def merge_cache(cache_row):
        try:
            url = cache_row[COL_github_repo_url]

            row = projects.loc[url, :]
            if hash_project_source(row) == cache_row[COL_hash]:
                row[COL_lang] = cache_row[COL_lang]
                row[COL_tokens] = cache_row[COL_tokens]
        except KeyError:
            pass  # repo has been deleted
        except Exception as e:
            print('merge_cache error: ' + str(e))

    try:
        cache_df = pd.read_pickle(cache_filename)
        cache_df[COL_github_repo_url] = cache_df.index
        cache_df.apply(merge_cache, axis=1)
    except Exception as e:
        # printf('read_cache exception')
        print('read_cache exception: ' + str(e))


def lsa():
    tsvPath = mostRecentTsv()
    projects = pd.read_csv(tsvPath, sep='\t', index_col=COL_github_repo_url)

    projects = projects.head(100)  # todo remove

    projects[COL_lang] = ''
    projects[COL_tokens] = ''
    projects[COL_hash] = ''
    projects[COL_github_repo_url] = projects.index
    projects.fillna(value='', inplace=True)
    read_cache(projects)

    print(len(projects))

    # todo add fake topic-seeding docs
    # todo add lang, preprocessing cache.  Lang detection is very slow.
    # todo seed topics using eta https://gist.github.com/scign/2dda76c292ef76943e0cd9ff8d5a174a
    langs = projects.apply(lang_detect_row, axis=1)
    preprocessed = projects.apply(preprocess, axis=1)
    projects[COL_lang] = langs
    projects[COL_tokens] = preprocessed
    write_cache(projects)

    inputs = preprocessed

    dictionary = gensim.corpora.Dictionary(inputs)

    # def probe_dict(word: str):
    #     idx = dictionary.token2id.get(word, None)
    #     coll_freq = dictionary.cfs.get(idx, None)
    #     doc_freq = dictionary.dfs.get(idx,None)
    #     return f'probe: word={word} idx={idx} cfs={coll_freq} dfs={doc_freq}'
    #
    # print('before filter')
    # print(probe_dict('case'))
    dictionary.filter_extremes(no_below=50, no_above=0.5, keep_n=100000)
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
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=2,
                                           chunksize=100000, workers=3)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    return lda_model


print(lsa())


# ['websit' 'hopkin_univers' 'descript' 'italia' 'case_death' 'notif', 'interact_visual' 'deploy' 'exposur' 'map' 'close' 'dato' 'page' nan, 'updat' 'confirm_case' 'finder' 'statist_countri' 'hopkin' 'protocol', 'research_dataset' 'tree' 'doc' 'visualis' 'so

@dataclass
class TopicDef():
    name: str
    words: [str]
    desc: str = ''
    group: str = ''


mytopics = [
    TopicDef('mask', ['mask']),
    TopicDef('social-distancing', ['social_distanc']),
    TopicDef('mobile', ['mobil', 'android', 'ios']),
    TopicDef('hospital', ['ventil', 'icu', 'hospit']),
    TopicDef('case', ['case']),
    TopicDef('italy', ['italy', 'caso']),
    TopicDef('india', ['india']),
    TopicDef('usa', ['']),
    TopicDef('antibody', ['seri']),
    TopicDef('contact-tracing', ['contact_tracing']),
    TopicDef('visualization', ['chart', 'track']),

]
