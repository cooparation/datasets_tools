#!/usr/bin/env python

import os
import sys
import shutil

# cp the images in sourceDir which have labels to targetDir
def cp_files(labelsDir, sourceDir, targetDir):
    for walk in os.walk(labelsDir):
        objIndex = 0
        for each in walk[2]:
            objIndex += 1
            each = each[:-4]
            pictureName = each + '.jpg'
            #pictureName = each.replace('.xml', '.jpg')
            imageFile = sourceDir +'/'+ pictureName
            print "cp ", imageFile, ' to ',targetDir
            if os.path.exists(imageFile):
               shutil.copy(imageFile, targetDir)
        print 'data num:', objIndex

if __name__ == '__main__':
    if (len(sys.argv) == 4):
        labelsDir = sys.argv[1]
        sourceDir = sys.argv[2]
        targetDir = sys.argv[3]
    else:
        print "Usage:", sys.argv[0], 'labelsDir sourceImgDir targetImgDir'
        sys.exit(1)
    cp_files(labelsDir, sourceDir, targetDir)
