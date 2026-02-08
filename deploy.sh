#!/usr/bin/env bash
set -e

mkdocs build
if git status | grep -q "modified:"; then
    echo "Files were modified."
    git commit -am "new docs"
    git push
else
    echo "Nothing modified."
fi

git checkout gh-pages
git rm -r --ignore-unmatch ./*
git checkout main -- ./site
git mv ./site/* ./
git add . 

if git status | grep -q "modified:"; then
    echo "Files were modified."
    git commit -am "new docs"
    git push
else
    echo "Nothing modified."
fi

git checkout main
