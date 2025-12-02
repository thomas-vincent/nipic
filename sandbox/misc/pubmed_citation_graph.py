# -*- coding: utf-8 -*-
"""
pubmed2svg PM_ID_1 PM_ID_2 ...

PM_ID can either be a PMID or an url
"""
import sys
import os
import os.path as op
import tempfile
import urllib.request, urllib.error, urllib.parse

import re
import time
from math import ceil, floor
import traceback

from collections import namedtuple, defaultdict

import logging
logger = logging.getLogger('BibGraph')
console_handler = logging.StreamHandler(stream=sys.stdout)
fmt = '%(name)s | %(asctime)s %(levelname)-8s  %(message)s'
dfmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=fmt, datefmt=dfmt)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


from Bio import Entrez

Entrez.email = 'thomas.vincent@icm-mhi.org'
Entrez.tool = 'pybibgraph'

svg_header = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
    xmlns:svg="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">
"""

svg_footer = """</svg>"""

import pygraphviz as pgv

MIN_CITED_IN_COUNTS = 2 # to display cited refs
MIN_CITING_COUNTS = 2 # to display citing refs
MAX_NB_REFS = 1000 # 500 # -1 for everything, note there is retmax=1000

# Primary node: ref from the initial query
primary_node_attrs = {'style': 'filled', 'fillcolor' : '#FF8382', 'shape':'box'}

# cited ref (not in initial query)
cited_node_attrs = {'style': 'filled', 'fillcolor' : '#729FCF'}

# citing ref (not in initial query)
citing_node_attrs = {'style': 'filled', 'fillcolor' : '#8AE234'}

if 0:
    gg = pgv.AGraph(directed=True, overlap="scale")
    gg.add_node('A', label='initial', **primary_node_attrs)
    gg.add_node('B', label='cited', **cited_node_attrs)
    gg.add_node('C', label='citing', **citing_node_attrs)
    gg.draw('test.svg', prog='dot')
    sys.exit(0)

PubmedRef = namedtuple('PubmedRef', 'label pm_id url title pub_types')
Keyword = namedtuple('Keyword', 'type value')
# class PubmedRef:

#     def __init__(self):
#         self.label = label
#         self.pm_id = pm_id
#         self.url = url

#     def add_cited_in(self, cited_in):
#         pass
    
#     def set_cited_in(self, list_cited_in):  
#         self.cited_in

#     def get_citation_count(self, cited_in):
#         if self.cited_in == None:
#             self.set_cited_in(pubmed_)


import validators
import string

def slugify(s):
    safechars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits + '.-')
    return ''.join([c if c in safechars else '_' for c in s]).replace('__', '_').strip('_')

def refs_from_ids(ids):
    stream = Entrez.esummary(db="pubmed", id=ids)
    summaries = Entrez.read(stream)

    pm_refs = []
    for entry in summaries:
        # from IPython import embed; embed()
        if 'AuthorList' in entry and len(entry['AuthorList']) > 0:
            label = entry['AuthorList'][0] + ' ' + entry.get('EPubDate', 'XXXX')[:4]
        else:
            label = entry['Id']
        entry_url  = ('https://doi.org/' + entry['DOI']) if 'DOI' in entry else None
        if not validators.url(entry_url):
            entry_url = None
        pub_types = entry.get('PubTypeList', [])

        title = entry.get('Title', 'NA')
        pm_refs.append(PubmedRef(label, entry['Id'], entry_url, title, pub_types))
    return pm_refs

def main():
    # query = 'TCD+AND+NIRS'
    # query = 'cerebral+AND+pulsatility'
    # query = '%28cardiac+disease+AND+brain%29+AND+%28%28%222018%2F01%2F01%22%5BDate+-+Publication%5D+%3A+%222022%2F09%2F28%22%5BDate+-+Publication%5D%29%29'
    # query = '%28%28cardiac+disease%29+AND+%28brain+imaging%29%29+AND+%28%28%222018%2F01%2F01%22%5BDate+-+Publication%5D+%3A+%222022%2F09%2F28%22%5BDate+-+Publication%5D%29%29'
    query = '%22Cerebral+Small+Vessel+Diseases%2Fdiagnostic+imaging%22%5BMeSH%5D+OR+%22Cerebral+Small+Vessel+Diseases%2Fphysiopathology%22%5BMAJR%5D'
    query = '%28cerebral+blood+flow%29+AND+%28health%29'
    query = "(cerebral blood flow) AND health"
    query = '(cerebral autoregulation) NOT (Review[Publication Type])'
    query = '(("cerebral autoregulation") NOT (Review[Publication Type])) AND (cognition)'
    query = '(("White Matter Hyperintensities") NOT (Review[Publication Type])) AND (physical exercise)'
    query = '((atrophy) AND (cognition)) NOT (Review[Publication Type])'
    query = '((cerebral blood flow) AND (cognition) AND ((fitness) OR (physical exercise) OR (physical activity))) NOT (Review[Publication Type])'
    retmax = 1000
    stream = Entrez.esearch(db="pubmed", term=query, sort='relevance', retmax=retmax)
    record = Entrez.read(stream)
    logger.info('Esearch gave %d results.', len(record['IdList']))
    if len(record['IdList']) == 0:
        sys.exit(0)

    refs = refs_from_ids(record['IdList'])

    refs = refs[:MAX_NB_REFS]
    logger.info('Processing %d refs...', len(refs))

    graph = pgv.AGraph(directed=True, overlap='scale')

    for ref in refs:
        graph.add_node(ref.pm_id, label=ref.label, URL=ref.url,
                       tooltip=ref.title,
                       **primary_node_attrs)
        # nn = graph.get_node(ref.pm_id)
        # nn.attr = .copy()
        logger.debug('Added primary ref: %s (%s)', ref.label, ref.pm_id)

    # TODO: show citation counts
    logger.info('Getting all citations ...')
    nb_trials = 5
    records = None
    for trial in range(nb_trials):
        try:
            records = Entrez.read(Entrez.elink(dbfrom="pubmed", id=[r.pm_id for r in refs],
                                               linkname='pubmed_pubmed_refs',
                                               retmax=retmax))
        except:
            logger.warning('Attempt %d of %d failed.', trial+1, nb_trials)
            logger.warning(traceback.format_exc())
            time.sleep(3)
            continue
        break
    if records is None:
        logger.error('Pubmed query failed')
        sys.exit(1)
    cited_in = parse_cite_results(records)
    cited_in_counts = defaultdict(int)
    for ref in refs:
        cited_refs = cited_in[ref.pm_id]
        for cited_ref in cited_refs:
            cited_in_counts[cited_ref.pm_id] += 1

    logger.info('Getting all citers ...')
    records = None
    for trial in range(nb_trials):
        try:
            records = Entrez.read(Entrez.elink(dbfrom="pubmed", id=[r.pm_id for r in refs],
                                               linkname='pubmed_pubmed_citedin',
                                               retmax=retmax))
        except:
            logger.warning('Attempt %d of %d failed.', trial+1, nb_trials)
            logger.warning(traceback.format_exc())
            time.sleep(3)
            continue
        break
    if records is None:
        logger.error('Pubmed query failed')
        sys.exit(1)

    citing = parse_cite_results(records)
    # citing = pubmed_get_citing(refs)
    citing_counts = defaultdict(int)
    for ref in refs:
        citing_refs = citing[ref.pm_id]
        for citing_ref in citing_refs:
            citing_counts[citing_ref.pm_id] += 1


        logger.debug('Parsing citation of: %s', ref.label)
        for cited_ref in cited_in[ref.pm_id]:
            if cited_in_counts[cited_ref.pm_id] >= MIN_CITED_IN_COUNTS:
                if not graph.has_node(cited_ref.pm_id):

                    graph.add_node(cited_ref.pm_id, label=cited_ref.label,
                                   tooltip=cited_ref.title,
                                   URL=cited_ref.url, **cited_node_attrs)
                    # nn = graph.get_node(ref.pm_id)
                    # nn.attr = primary_node_attrs.copy()
                    logger.debug('Added cited ref: %s (%s)' %
                          (cited_ref.label, cited_ref.pm_id))
                if cited_ref.pm_id != ref.pm_id:
                    graph.add_edge(ref.pm_id, cited_ref.pm_id, penwidth=4)

        logger.debug('Parsing citing refs of: %s', ref.label)
        for citing_ref in citing[ref.pm_id]:
            if citing_counts[citing_ref.pm_id] >= MIN_CITING_COUNTS:
                if not graph.has_node(citing_ref.pm_id):

                    graph.add_node(citing_ref.pm_id, label=citing_ref.label,
                                   tooltip=citing_ref.title,
                                   URL=citing_ref.url, **citing_node_attrs)

                    logger.debug('Added citing ref: %s (%s)' %
                                (citing_ref.label, citing_ref.pm_id))
                if citing_ref.pm_id != ref.pm_id:
                    graph.add_edge(citing_ref.pm_id, ref.pm_id, penwidth=4)

    output_label = 'ref_network_%s' % slugify(query)
    graph.layout(prog='sfdp')
    graph.write('%s.dot' % output_label)
    map_fn = '%s.map' % output_label
    graph.draw(path=map_fn, format="cmapx")
    svg_fn = '%s.svg' % output_label
    graph.draw(path=svg_fn, format="svg")

def pubmed_query(query):
    # retmax=1000
    url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax=50&sort=relevance&term={query}'
           .format(query=query))
    logger.info('Pubmed query url:')
    logger.info(url)
    connection = urllib.request.urlopen(url)
    time.sleep(1.5)
    soup = BeautifulSoup(connection.read(),
                         from_encoding=connection.headers.get_content_charset(),
                         features="xml")
    ref_ids = [ref_id.contents[0] for ref_id in soup.find_all('Id')]
    logger.debug('Fetched %d Ids', len(ref_ids))
    return pubmed_refs_from_ids(ref_ids)

def pubmed_refs_from_ids(ids):
    url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' \
               'db=pubmed&retmode=xml&id='

    entries = chunked_soup_query(ids, url_base,
                                 lambda soup: soup.find_all(string=re.compile('pubmedarticle', re.IGNORECASE)))

    # MAX_URL_LENGTH = 2000
    # url_base = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='
    # nb_chunks = int(math.ceil(len(','.join(ids))*1.0 / (MAX_URL_LENGTH - len(url_base))))
    # nb_ids_per_chunk = int(math.floor(len(ids)*1.0 / nb_chunks))
    # entries = []
    # for chunk_ids in chunks(ids, nb_ids_per_chunk):
    #     url = url_base + ','.join(chunk_ids)
    #     assert(len(url) < MAX_URL_LENGTH)
  
    #     connection = urllib2.urlopen(url)
    #     soup = BeautifulSoup(connection.read(),
    #                          from_encoding=connection.info().getparam('charset'))

    #     entries.extend(soup.find_all('pubmedarticle'))

    pm_refs = []
    for entry_id, entry in zip(ids, entries):
        # from IPython import embed; embed()
        if entry.find('authorlist') is not None \
           and entry.find('authorlist').find_all('author') is not None and \
           entry.find('authorlist').find_all('author')[0].find('lastname') is not None:
            authors =  entry.find('authorlist').find_all('author')

            year = get_year(entry.find('pubdate'))
            first_author = authors[0].find('lastname').contents[0]
            label = first_author + year
        else:
            label = entry_id
        entry_url  = pubmed_get_url(entry_id)
        pub_types = [e.contents[0] for e in
                     entry.find(string=re.compile('publicationtypelist', re.IGNORECASE))
                     .find_all(string=re.compile('publicationtype', re.IGNORECASE))]

        title = entry.find(string=re.compile('articletitle', re.IGNORECASE)).contents[0]
        mesh_list = entry.find(string=re.compile('meshheadinglist', re.IGNORECASE))
        if mesh_list is not None:
            mesh_items = mesh_list.find_all(string=re.compile('meshheading', re.IGNORECASE))
            keywords = [(mi.find(string=re.compile('descriptorname', re.IGNORECASE)).attrs['ui'],
                         mi.find(string=re.compile('descriptorname', re.IGNORECASE)).contents[0])
                         for mi in mesh_items]
        else:
            kw_list = entry.find(string=re.compile('keywordlist', re.IGNORECASE))
            keywords = []
            if kw_list is not None:
                keyword_items = kw_list.find_all(string=re.compile('keyword', re.IGNORECASE))
                keywords = [Keyword('NA', kwi.contents[0])
                            for kwi in keyword_items if len(kwi.contents)>0]
            #else:
            #    keywords = mesh_terms_from_title(title)
        pm_refs.append(PubmedRef(label, entry_id, entry_url, title,
                                 pub_types, keywords))
    return pm_refs

def re_terms(term_list):
    return ''.join(['(?=.*\b(%s)\b)' % t for t in term_list]) + '.*'

mesh_regexps = {('D015150','Echocardiography, Doppler') : \
                re.compile('.*doppler.*', re.IGNORECASE),
                ('D019265','Spectroscopy, Near-Infrared'): \
                re.compile('.*nirs', re.IGNORECASE),
                ('D002560','Cerebrovascular Circulation'):
                re.compile(re_terms(['cerebr|brain', 'vascular|blood']),
                           re.IGNORECASE)}

def mesh_terms_from_title(title):
    terms = []
    for mesh_id, mesh_re in mesh_regexps.items():
        if mesh_re.match(title):
            terms.append(Keyword(*mesh_id))
    return terms

# def pubmed_refs_from_ids_bak(ids):
#     """ Using esummary to get minimal information -> no mesh term """
#     url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=' + ','.join(ids)
#     connection = urllib2.urlopen(url)
#     soup = BeautifulSoup(connection.read(),
#                          from_encoding=connection.info().getparam('charset'))

#     entries = soup.find_all('docsum')
#     pm_refs = []
#     for entry_id, entry in zip(ids, entries):
#         author_list =  entry.find(lambda tag: 'AuthorList' in tag.attrs.values())
#         first_author_item = author_list.find(lambda tag: 'Author' in tag.attrs.values())
#         if first_author_item is not None:
#             year = get_year(.contents[0])
#             first_author = first_author_item.contents[0].split(' ')[0]
#             label = first_author + year
#         else:
#             label = entry_id
#         entry_url  = pubmed_get_url(entry_id)

#         pm_refs.append(PubmedRef(label, entry_id, entry_url, title,
#                                  pubmed_cats,  ))

#     return pm_refs

def get_year(item):
    year_item = item.find('year')
    if year_item is not None:
        return year_item.contents[0]
    else:
        mdate_item = item.find('medlinedate')
        if mdate_item is not None:
            return mdate_item.contents[0][:4]
        else:
            pdate_item = item.find(lambda tag: 'PubDate' in tag.attrs.values())
            if pdate_item is not None:
                return pdate_item.contents[0][:4]
            else:
                raise Exception('Cannot retrieve date')

def pubmed_get_url(pm_id):
    return 'http://www.ncbi.nlm.nih.gov/pubmed/' + pm_id

def pubmed_get_citing(refs):
    url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' \
               'dbfrom=pubmed&linkname=pubmed_pubmed_citedin&'
    return chunked_soup_query(refs, url_base, parse_cite_results,
                              lambda r: 'id=' + r.pm_id, join_char='&')

    # url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&' + '&'.join(['id='+ref.pm_id for ref in refs]) 

    # connection = urllib2.urlopen(url)
    # soup = BeautifulSoup(connection.read(),
    #                      from_encoding=connection.info().getparam('charset'))

    # return parse_cite_results(soup)


def pubmed_get_cited_by(refs):
    url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' \
               'dbfrom=pubmed&linkname=pubmed_pubmed_refs&'
    return chunked_soup_query(refs, url_base, parse_cite_results,
                              lambda r: 'id=' + r.pm_id, join_char='&')
    # url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_refs&' + '&'.join(['id='+ref.pm_id for ref in refs]) 

    # connection = urllib2.urlopen(url)
    # soup = BeautifulSoup(connection.read(),
    #                      from_encoding=connection.info().getparam('charset'))

    # return parse_cite_results(soup)

def chunked_soup_query(items, url_base, extract_from_soup, format_item=None,
                       join_char=','):

    if format_item is None:
        format_item = lambda i:i

    MAX_URL_LENGTH = 2000
    logger.debug('Chunked soup query for URL %s, with %d items', url_base, len(items))
    nb_chunks = int(ceil(len(join_char.join([format_item(i) for i in items]))*1.0 / \
                         (MAX_URL_LENGTH - len(url_base))))
    nb_items_per_chunk = int(floor(len(items)*1.0 / nb_chunks))
    result = None
    for ichunk, chunk_ids in enumerate(chunks(items, nb_items_per_chunk)):
        url = url_base + join_char.join([format_item(item) for item in chunk_ids])
        assert(len(url) < MAX_URL_LENGTH)
        logger.debug('URL for chunk %d: %s', ichunk, url)
        try:
            connection = urllib.request.urlopen(url)
        except urllib.error.HTTPError as err:
            print('HTTPError %s' % err)
            if err.code == 429:
                time_wait = int(err.hdrs['retry-after'])
                print('Waiting %d sec before retrying...' % time_wait)
                time.sleep(time_wait)
                connection = urllib.request.urlopen(url)
            else:
                raise

        soup = BeautifulSoup(connection.read(),
                             from_encoding=connection.headers.get_content_charset(),
                             features="xml")
        from IPython import embed; embed()
        if result is None:
            result = extract_from_soup(soup)
        else:
            if isinstance(result, dict):
                result.update(extract_from_soup(soup))
            else: # ASSUME list type
                result.extend(extract_from_soup(soup))
        time.sleep(1.5)
    return result


def parse_cite_results(records):
    citations = {}
    for record in records:
        ref_id = record['IdList'][0]
        if len(record.get('LinkSetDb', [])) > 0:
            ids = [l['Id'] for l in record['LinkSetDb'][0]['Link']]
            citations[ref_id] = refs_from_ids(ids)
        else:
            citations[ref_id] = []
    return citations

def __parse_cite_results(soup):
    citations = {}
    for result in soup.find_all(string=re.compile('linkset', re.IGNORECASE)):
        ref_id = (result.find(string=re.compile('idlist', re.IGNORECASE))
                  .find(string=re.compile('id', re.IGNORECASE)).contents[0])
        linked_refs = result.find(string=re.compile('linksetdb', re.IGNORECASE))
        if linked_refs is not None:
            ids = [l.contents[0] for l in linked_refs.find_all('id')]
            citations[ref_id] = pubmed_refs_from_ids(ids)
        else:
            citations[ref_id] = []

    return citations

def chunks(items, chunks_size):
    """Yield successive chunks of given size from list of items."""
    for i in range(0, len(items), chunks_size):
        yield items[i:i + chunks_size]

def bak():
    items = []
    coord_y = 16
    delta_y = 12
    icon_height = 10
    for pubmed_ressource in sys.argv[1:]:
        if not pubmed_ressource.startswith('http'):
            pubmed_ressource = 'https://www.ncbi.nlm.nih.gov/pubmed/%s' \
                               % pubmed_ressource

        print('reading content from %s ...' % pubmed_ressource)
        connection = urllib.request.urlopen(pubmed_ressource)
        soup = BeautifulSoup(connection.read(),
                             from_encoding=connection.info().getparam('charset'), features="xml")
        title = soup.find('title').text
        if len(title) == 0:
            raise Exception('Cannot find title in %s' % pubmed_ressource)  
        title = title.replace('- PubMed', '').replace('- NCBI', '').strip()
    
        coord_y += delta_y
    
        items.append(svg_pat.format(article_title=title,
                                    pubmed_link=pubmed_ressource,
                                    txt_y=coord_y, icon_y=coord_y-icon_height))
        
    fout = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
    fout.write('\n'.join([svg_header] + items + [svg_footer]))
    fout.close()
    
    if op.exists(fout.name):
        os.system('inkscape %s' % fout.name)
        
if __name__ == '__main__':
    main()
