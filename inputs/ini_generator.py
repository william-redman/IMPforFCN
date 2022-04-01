# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 19:39:15 2022

@author: redman
"""

import configparser

config = configparser.ConfigParser() 
config['IO'] = {'save_dir' : '9', 'prefix' : 'IMP'}

with open('test.ini', 'w') as configfile:
    config.write(configfile)