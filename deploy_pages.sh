#!/usr/bin/env bash
set -e

mkdocs build
if git status | grep -q "Your branch is up to date"; then
    echo "Branch is up to date."
else
    echo "Branch is not up to date."
    git commit -am "new docs"
    git push
fi

git checkout gh-pages
git rm -r --ignore-unmatch ./*
git checkout main -- ./site
git mv ./site/* ./
git add . 
git commit -m "updating the project page"
git push
git checkout main
