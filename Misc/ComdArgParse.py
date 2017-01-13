#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for command line arguments

Created on Wed Jan  4 11:51:21 2017

@author: xiaomuliu
"""

import sys, getopt

def ParseArg():
    """
    Parse command line argument
    (only supports up to one parameter besides -h,-i,-o)
    """
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:p:",["input=","output=","param="])
    except getopt.GetoptError:
        print(sys.argv[0]+' -i <input file/path> -o <output file/path> -p <parameter> ')
        sys.exit(2)
    inputfile = None    
    outputfile = None
    param = None    
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0]+' -i <input file/path> -o <output file/path> -p <parameter>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
        elif opt in ("-p", "--param"):
            param = arg
        else:
            assert False, "unknown option"   

    return {'input':inputfile,'output':outputfile,'param':param}
        
if __name__ == "__main__":
   Args = ParseArg()
   print(Args['input'], Args['output'], Args['param'])
   