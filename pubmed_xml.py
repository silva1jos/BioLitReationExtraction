""" Code for downloading pubmed abstracts and saving as XML files"""
from xml.etree import ElementTree
import requests
import os.path
import time


def get_files(pubmed_ids):
    """ Given a list of pubmed_ids make a request to the server for 200 ids. """
    if len(pubmed_ids) > 200:
        raise ValueError("Can have up to 200 ids per request")
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    options = {'db': 'pubmed', 'retmode': 'xml', 'id': ','.join(pubmed_ids)}
    r = requests.get(url=base_url, params=options)
    return r.text


def write_files(directory, xml_string):
    """ From the result of a Entrez Utils request, as a string, break the
    PubmedArticleSet into PubmedArticles and save as xml named by pubmed ID """
    tree = ElementTree.fromstring(xml_string)
    for child in tree:
        pubmed_id = child[0][0].text
        ElementTree.ElementTree(child).write(os.path.join(os.path.abspath(directory),
                                                          pubmed_id + '.xml'),
                                             encoding='utf-8')


def many_files(directory, pubmed_ids):
    """ get and write abstracts for many pubmed files breaking up the requests to
    work with the pubmed servers. """
    i = 0
    n_pubmed_ids = len(pubmed_ids)
    while True:
        if i * 200 > n_pubmed_ids:
            break
        if i % 3 == 0:
            print(i)
            time.sleep(1)  # if over 3 requests sent per second, server gives back error
        write_files(directory, get_files(pubmed_ids[i*200:(i+1)*200]))
        i += 1


if __name__ == '__main__':
    many_files('abstracts', ['16120569'])
