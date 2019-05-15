"""
Rename all vlcsnap png files in a directory.

E.g. suppose that a directory contains the following files:
vlcsnap-2019-05-09-14h46m06s223.png
vlcsnap-2019-05-09-14h48m40s799.png
vlcsnap-2019-05-09-15h13m52s967.png

They will be renamed in the following way:
suitcase-1.png
suitcase-2.png
suitcase-3.png

"""

import os

path = '.'
rename_prefix = 'suitcase-'
i = 1

for root, dirs, files in os.walk(path):
	for filename in files:
		if 'vlcsnap' in filename and '.png' in filename: 
			new_filename = rename_prefix + str(i) + '.png'
			print(filename + ' -> ' + new_filename)
			os.rename(filename, new_filename)
			i = i + 1
