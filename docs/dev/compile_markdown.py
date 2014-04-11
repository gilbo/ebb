#!/usr/bin/env python

TEMPLATE_0 = u"""
<html>
<head>

<link rel="stylesheet" type="text/css" href="css/style.css" />

<title></title>

</head>


<body><div id="body-wrapper">
"""

TEMPLATE_1 = u"""
</div></body>
</html>
"""


import argparse
import os



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=
                  'Generate a formatted markdown file.')
  parser.add_argument('markdown_file', help=
                  'path to the markdown file')
  args = parser.parse_args()
  md_file = args.markdown_file

  # figure out the html file
  html_file = md_file
  if html_file[-3:] == '.md':
    html_file = html_file[:-3]
  html_file = html_file + '.html'

  os.system("echo '"+TEMPLATE_0+"' >"+html_file)
  os.system("markdown_py "+md_file+" >>"+html_file)
  os.system("echo '"+TEMPLATE_1+"' >>"+html_file)