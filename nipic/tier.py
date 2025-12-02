import os.path as op
import sys

import logging
logger = logging.getLogger('nipic')

script_dir = op.dirname(op.realpath(__file__))

def tier_root(running_dir=None):
    if running_dir is None:
        running_dir = script_dir
    logger.debug('running dir is: %s', running_dir)
    test_rdir = 'command_files'
    MAX_DEPTH = 5
    for depth in range(MAX_DEPTH):
        test_rdir = op.join('..', test_rdir)
        check_command_dir = op.join(running_dir, test_rdir)
        logger.debug('Check for tier command_files dir: %s', check_command_dir)
        if op.exists(check_command_dir):
            tier_dir = op.abspath(op.join(check_command_dir, '..'))
            logger.debug('Found TIER root: %s', tier_dir) 
            return tier_dir
    return tier_root(running_dir=op.dirname(sys.argv[0]))
    raise FileNotFoundError('TIER root not found (parent dir containing command_files)')

def origin_dir(running_dir=None):
    return op.join(tier_root(running_dir=running_dir), 'data_origin')

def analysis_dir(running_dir=None):
    return op.join(tier_root(running_dir=running_dir), 'data_analysis')

import tempfile
import unittest
import shutil
import os

class TestTier(unittest.TestCase):
    def setUp(self):
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_proper_dir(self):

        command_sub_dir = op.join(self.tmp_dir, 'projects', 'tier_root', 'command_files', 'sub_command')
        os.makedirs(command_sub_dir)

        origin_directory = op.join(self.tmp_dir, 'projects', 'tier_root', 'data_origin')
        os.makedirs(origin_directory)

        analysis_directory = op.join(self.tmp_dir, 'projects', 'tier_root', 'data_analysis')
        os.makedirs(analysis_directory)

        self.assertEqual(origin_dir(running_dir=command_sub_dir), origin_directory)
        self.assertEqual(analysis_dir(running_dir=command_sub_dir), analysis_directory)

    def test_no_command_dir(self):
        my_script_dir = op.join(self.tmp_dir, 'projects', 'scripts')
        os.makedirs(my_script_dir)

        self.assertRaises(FileNotFoundError, origin_dir, running_dir=my_script_dir)
        self.assertRaises(FileNotFoundError, analysis_dir, running_dir=my_script_dir)
