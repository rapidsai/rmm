#!/bin/bash
set -e

DIFF_FILES=$(mktemp)
LARGE_FILES=$(mktemp)
FILESIZE_LIMIT=5242880
RETVAL=0

# Activate rapids environment for Git LFS access
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Checkout target branch
git checkout --force $TARGET_BRANCH
# Switch back to PR branch
git checkout --force current-pr-branch
# Checkout newest commit of PR branch
git checkout -fq $COMMIT_HASH;

# Get list of files changed in current PR
git diff --name-only $TARGET_BRANCH..current-pr-branch > ${DIFF_FILES}

echo '### Files modified in current PR'
while read FILE; do 
    echo 'Size check ' $FILE
    if [ -f "$WORKSPACE/$FILE" ]; then
        if [ `du -b "$WORKSPACE/$FILE" | awk '{print $1}'` -gt $FILESIZE_LIMIT ]; then 
            RETVAL=1
            echo $FILE >> ${LARGE_FILES}
        fi
    fi
done < ${DIFF_FILES}

if [ $RETVAL == 1 ]; then
    echo "### Files exceeding the $FILESIZE_LIMIT size limit.  Please see documentation for" 
    echo "### large file handling:  https://docs.rapids.ai/resources/git/#large-files-and-git"
    cat  ${LARGE_FILES}  
    echo "###"
    else
    echo "### All files under the $FILESIZE_LIMIT size limit"
fi

rm -f ${DIFF_FILES} ${LARGE_FILES}
exit $RETVAL
