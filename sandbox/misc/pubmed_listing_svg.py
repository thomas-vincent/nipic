"""
pubmed2svg PM_ID_1 PM_ID_2 ...

PM_ID can either be a PMID or an url
"""
import sys
import os
import os.path as op
import tempfile
import urllib2
import re

from bs4 import BeautifulSoup

svg_header = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
    xmlns:svg="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">
"""

svg_pat = """
  <g>
    <text y="{txt_y}" x="34" style="font-style:normal;font-weight:normal;font-size:11px">
      <tspan y="{txt_y}" x="34"> {article_title} </tspan>
    </text>
    <a xlink:href="{pubmed_link}">
        <g transform="matrix(0.22619255,0,0,0.22619255,4,{icon_y})">
           <g>
		<path fill="#231F1F" d="M50.487,2.04v29.034c0,0,11.162-1.299,11.162,9.604V13.201C61.649,2.818,50.487,2.04,50.487,2.04"/>
		<path fill="#AAAAAA" d="M55.715,0v25.547c0,0,6.162,1.189,6.162,12.091c0,10.903,0.031-20.283,0.031-26.124    C61.909,1.132,55.715,0,55.715,0"/>
	   </g>
	<path fill="#3978AD" d="M62.077,12.941v26.422c0,0,5.359-15.01,27.887,4.328c0-7.008,0.043-24.683,0.043-30.525   C67.739-3.057,62.077,12.941,62.077,12.941"/>
	<path fill="#3978AD" d="M0,32.718h5v-6.399h2.848c4.388,0,6.852-1.996,6.852-6.55c0-4.427-2.869-6.349-7.018-6.349H0V32.718z    M5,17.419h0.853c1.965,0,3.646,0.027,3.646,2.503c0,2.396-1.81,2.396-3.646,2.396H5V17.419z"/>
	<path fill="#3978AD" d="M16.699,19.619v7.871c0,4.304,3.073,5.629,7.05,5.629c3.977,0,7.051-1.324,7.051-5.629v-7.871h-4.7v6.974   c0,1.654-0.414,2.825-2.351,2.825s-2.35-1.171-2.35-2.825v-6.974H16.699z"/>
	<path fill="#3978AD" d="M33.6,32.718h4.7v-1.459h0.051c0.841,1.282,2.42,1.859,3.949,1.859c3.898,0,6.6-3.225,6.6-7.013   c0-3.764-2.677-6.988-6.549-6.988c-1.503,0-3.057,0.574-4.05,1.747v-9.347h-4.7L33.6,32.718z M41.1,23.118c1.897,0,3,1.378,3,3.014   c0,1.688-1.102,2.986-3,2.986c-1.897,0-3-1.298-3-2.986C38.1,24.496,39.202,23.118,41.1,23.118L41.1,23.118z"/>
	<polygon fill="#FFFFFF" points="64.2,32.718 69.192,32.718 70.805,21.607 70.856,21.607 75.286,32.718 77.284,32.718    81.918,21.607 81.969,21.607 83.377,32.718 88.399,32.718 85.477,13.419 80.485,13.419 76.336,23.733 72.418,13.419 67.477,13.419     "/>
	<path fill="#3978AD" d="M96.7,24.218c0.281-1.374,1.408-2.1,2.764-2.1c1.254,0,2.406,0.83,2.639,2.1H96.7z M106.426,26.431   c0-4.49-2.65-7.313-7.227-7.313c-4.293,0-7.299,2.636-7.299,7.013c0,4.53,3.264,6.987,7.637,6.987c3.008,0,5.965-1.406,6.711-4.5   h-4.492c-0.512,0.869-1.23,1.201-2.23,1.201c-1.924,0-2.926-1.019-2.926-2.901h9.826V26.431z"/>
	<path fill="#3978AD" d="M118.399,32.718h4.701v-21.2h-4.701v9.347c-0.969-1.173-2.549-1.747-4.051-1.747   c-3.873,0-6.549,3.225-6.549,6.987c0,3.762,2.727,7.012,6.6,7.012c1.527,0,3.133-0.578,3.947-1.859h0.053V32.718z M115.6,23.118   c1.896,0,3,1.378,3,3.014c0,1.688-1.102,2.986-3,2.986c-1.896,0-3-1.298-3-2.986C112.6,24.496,113.702,23.118,115.6,23.118   L115.6,23.118z"/>
</g></a></g>"""

svg_footer = """</svg>"""

items = []
coord_y = 16
delta_y = 12
icon_height = 10
for pubmed_ressource in sys.argv[1:]:
    if not pubmed_ressource.startswith('http'):
        pubmed_ressource = 'https://www.ncbi.nlm.nih.gov/pubmed/%s' \
                           % pubmed_ressource

    print 'reading content from %s ...' % pubmed_ressource
    connection = urllib2.urlopen(pubmed_ressource)
    soup = BeautifulSoup(connection.read(),
                         from_encoding=connection.info().getparam('charset'))
    title = soup.find('title').text
    if len(title) == 0:
        raise(Exception('Cannot find title in %s' % pubmed_ressource))

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
    
