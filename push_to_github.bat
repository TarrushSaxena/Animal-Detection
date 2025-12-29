@echo off
echo Initializing Git...
if not exist .git (
    git init
    echo Git initialized.
) else (
    echo Git already initialized.
)

echo Adding files...
git add .

echo Committing...
git commit -m "Initial commit"

echo Renaming branch to main...
git branch -M main

echo Configuring remote...
git remote remove origin 2>nul
git remote add origin https://github.com/TarrushSaxena/Animal-Detection.git

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause
