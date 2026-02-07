#!/usr/bin/env bash
set -e

mkdocs build
git commit -am "new docs"
git push
git checkout gh-pages
git rm -rf ./*
git checkout main -- ./site
git mv ./site/* ./
git add . 
git commit -m "updating the project page"
git push
git checkout main
