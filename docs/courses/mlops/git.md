---
template: lesson.html
title: "Git"
description: Tracking changes in our work for reproducibility and collaboration.
keywords: git, versioning, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

Whether we're working individually or with a team, it's important that we have a system to track changes to our projects so that we can revert to previous versions and so that others can reproduce our work and contribute to it. [Git](https://git-scm.com/){:target="_blank"} is a distributed versional control system that allows us do exactly this. Git runs *locally* on our computer and it keeps track of our files and their histories. To enable collaboration with others, we can use a *remote* host ([GitHub](https://github.com/){:target="_blank"}, [GitLab](https://gitlab.com/){:target="_blank"}, [BitBucket](https://bitbucket.org/){:target="_blank"}, etc.) to host our files and their histories. We'll use git to push our local changes and pull other's changes to and from the remote host.

!!! note
    Git is traditionally used to store and version small files <100MB (scripts, READMEs, etc.), however, we can still version large artifacts (datasets, model weights, etc.) using text pointers pointing to blob stores. These pointers will contain information such as where the asset is located, it's specific contents/version (ex. via hashing), etc. We'll see this in action in our [versioning](versioning.md){:target="_blank"} lesson where we'll create a pointer to a specific version of our dataset and models.

## Application

Instead of creating an overwhelming list of Git commands (you know how I feel about [list dumps](../../pivot.md){:target="_blank"}), let's learn about the important concepts and commands with quick chronological case studies that we'll absolutely need to know to real collaborative project with a team.

### Getting started

#### Set up

To follow along, we need to create a [GitHub](https://github.com/){:target="_blank"} (or any other remote host) account first and set our credentials globally on our local machine.
```bash
# Set credentials via terminal
git config --global user.name <username>
git config --global user.email <email>
```
We can quickly validate that we set the proper credentials like so:
```bash
# Check credentials
git config --global user.name
git config --global user.email
```

#### Create

Create a project in a *working directory*.
```bash
# Create project
mkdir git-tutorial
cd git-tutorial
```
For the purpose of this case study, we'll add some simple files. First will be a README.md and then another file called `do_not_push.txt` which we won't check into our repository.
```bash
# Create some files
touch README.md do_not_push.txt .gitignore
```
Now we'll go ahead and add some text to our README.md file. We can simply open the file in an IDE (ex. VS code) and add some text into it.
```bash
# Git Tutorial

This is a decent tutorial on Git.

```

#### Initialize git

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/environments.png">
</div>

*Initialize* a *local repository* (.git directory) to track our files.
```bash
# Initialize git
git init
```
We can see what files are untracked or yet to be committed.
```bash
# Check status
git status
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/status1.png" style="border-radius: 7px;">
</div>

We can see that we have our do_not_push.txt file as an untracked file in our working directory, as well as some other clutter (mac .DS_Store file). We can create a .gitignore file to make sure we aren't checking in these files.

!!! note
    Check out our tagifai's [.gitignore](https://github.com/GokuMohandas/MLOps/blob/main/.gitignore){:target="_blank"} for an more complete example or [generate](https://www.toptal.com/developers/gitignore){:target="_blank"} our own based on the tools we're using.

```bash
# Inside .gitignore
.DS_Store
do_not_push.txt
```

If we run `git status` now, we'll see the updated list of untracked files that we could commit.

```bash
# Untracked files
git status  # note that do_not_push.txt is not here
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/status2.png" style="border-radius: 7px;">
</div>


#### Add to stage

*Add* our work from the *working directory* to the *staging area*.

- We can add one file at a time:
```bash
# Add a file to stage
git add README.md
```
- We can add all files:
```bash
# Add all files to stage
git add .
```

Now running `git status` will show us all the staged files:

<div class="ai-center-all">
    <img width="450" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/status3.png" style="border-radius: 7px;">
</div>

#### Commit to repo

*Commit* the files in the *staging area* to the *local repository*. The default branch will be called `main` as it will be the main branch all future work will eventually merge with.

```bash
# Commit to local repo
git commit -m "initial commit"
git branch -M main  # rename branch to main (if needed)
```

The commit requires a message indicating what changes took place. We can use `git commit --amend` to edit the commit message if needed. If we do a `git status` check we'll see that there is nothing else to commit from our staging area.

<div class="ai-center-all">
    <img width="400" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/status4.png" style="border-radius: 7px;">
</div>

#### Push to remote

*Push* our commit to a *remote repository* (GitHub). We only need to add the remote origin address once and then we can push our local repository to the remote with a simple *push* command.
```bash
# Push to remote
git remote add origin https://github.com/GokuMohandas/git-tutorial.git
git push -u origin main  # pushing the contents of our main branch to the remote repository
                         # origin is short for the name of the remote repository
```
We first need to create a new remote repository to push our commit to by filling out this GitHub [form](https://github.com/new){:target="_blank"} (make sure we're logged into GitHub first). Let's call the repository `git-tutorial` and don't add any of the default files since we already have them. Once we're done, we'll see a HTTPS link like above which we can use to establish the connection between our local and the remote repositories. Now if we go our GitHub repository link, we'll see the files that we pushed.

### Developing

Now we're ready to start adding to our project and committing the changes.

!!! note
    If we (or someone else) doesn't already have the local repository set up and connected with the remote host, we can use the *clone* command.
    ```bash
    # Clone
    git clone https://github.com/GokuMohandas/git-tutorial <PATH_TO_PROJECT_DIR>
    ```

#### Create a branch

When we want to add or change something, such as adding a feature, fixing a bug, etc., it's always best practice to create a separate branch before developing. This is especially crucial when working with a team so we can cleanly merge our work with the main branch after discussion and review.

We'll create a branch called `good` (for a real project, our branch names should be much more meaningful) and check into it using:
```bash
# Create a new branch
git checkout -b good
```

We can see all the branches we've created with the following command where the * indicates our current branch.
```bash
# View branches
git branch
```

<pre class="output">
* <span style="color: #5A9C4B;">good</span>
<span style="color: #000;">main</span>
</pre>

We can easily switch between existing branches using:
```bash
# Switch between branches
git checkout <BRANCH_NAME>
```

Once we're in a branch, we can work on our project and commit those changes. So we'll go ahead and change the word "decent" to "good" in our README and save that file. Now if we do `git status` we'll see that our README.md file has some unstaged changes. So we'll go ahead and add our changes to the staging area, commit the change to our local repository and then push our commits to the remote repository.

```bash
# Add, commit and push
git add .
git commit -m "added the word good"
git push origin good
```

Note that we are pushing this branch to our remote repository, which doesn't yet exist there, so GitHub will create it accordingly.


#### Pull request (PR)

When we push our new branch to the remote repository, we'll need to create a pull request (PR) to merge with another branch (ex. our main branch in this case).

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/branch.png" style="border-radius: 7px;">
</div>
<div class="ai-center-all mt-2">
    <small>Visualize the git history using the <a href="https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph" target="_blank">Git Graph</a> extension on VS Code.</small>
</div>

When merging our work with another branch (ex. main), it's called a pull request because we're requesting the branch to *pull* our committed work. We can create the pull request using steps outlined here: [Creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/merge_gh.png">
</div>

!!! note
    We can  merge branches and resolve conflicts using git CLI commands but I prefer to use the online interface because we can easily visualize the changes, have discussion with teammates, etc.
    ```bash
    # Merge via CLI
    git push origin good
    git checkout main
    git merge good
    git push origin main
    ```

#### Pull

Once we accepted the pull request, our main branch is now updated with our changes. However, the update only happened on the remote repository so we should pull those changes to our local main branch as well.

```bash
# Pull updates
git checkout main
git pull origin main
```

#### Delete branches

Once we're done working with a branch, we can delete it to prevent our repository from cluttering up. We can easily delete both the local and remote versions of the branch with the following commands:
```bash
# Delete branches
git branch -d <BRANCH_NAME>  # local
git push origin --delete <BRANCH_NAME>  # remote
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/delete_branch.png" style="border-radius: 7px;">
</div>

### Collaborating

So far, the workflows for integrating our iterative development has been very smooth but in a collaborative setting, we may need to resolve conflicts.

#### Merge conflicts

Let's say there are two branches (`great` and `fantastic`) that were created from the `main` branch. Here's what we're going to try and simulate:

1. Developer A and B fork the `main` branch to make some changes
2. Developer A makes a change and submits a PR to the `main` branch.
3. Developer B makes a change to the same line as Developer A and submits a PR to `main`.
4. We have a merge conflict now since both developers altered the same line.
5. Choose which version of the code to keep and resolve the conflict.

```bash
# Branches from main
git checkout main
git checkout -b great
git checkout main
git checkout -b fantastic
```

In each branch, change the same word (ex. "good)" to another word but make sure they're different in each branch. We'll first submit a PR with the `great` branch:

```bash
# PR 1
git checkout good
git add .
git commit -m "changed good to great"
git push origin good
```

Create and merge the PR (there should be no conflicts) to the `main` branch.

Now we'll create a PR with the `great` branch and this time when we try to merge with the `main` branch, we'll see a conflict.

```bash
# PR 2
git checkout great
git add .
git commit -m "changed good to fantastic"
git push origin great
```

When we try to merge this PR, we have to resolve the conflicts between this new PR and what already exists in the `main` branch.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/resolve.png">
</div>

We can resolve the conflict by choosing which content (current `main` which merged with the `great` branch or this `fantastic` branch) to keep and delete the other one. Then we can merge the PR successfully and update our local `main` branch.

<div class="ai-center-all">
    <img width="350" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/conflict_gh.png">
</div>

```bash
# Update main after PR
git checkout main
git pull origin main
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/merge.png" style="border-radius: 7px;">
</div>

!!! note
    We only have a conflict because both branches were forked from a previous version of the `main` branch and they both happened to alter the same content. Had we created one branch first and then updated main before creating the second branch, we wouldn't have any conflicts. But in a collaborative setting, different developers may fork off the same version of the branch anytime.

#### Stash

Alternatively, instead of resolving the conflict after submitting the PR for the `fantastic` branch, we could've done so before the PR follow this workflow:

1. Developer A and B fork the `main` branch to make some changes
2. Developer A makes a change and submits a PR to the remote `main` branch.
3. Pull the latest version of the remote `main` branch to update the local `main` branch.
4. Developer B makes a change to the same line as Developer A.
5. Developer B *stash*es their work and *rebase*s with the now updated local `main` branch.
6. Developer B *applies* their stashed work on top of the rebased branch.
7. Developer B resolves conflicts locally.
8. Developer B creates a PR, which is now conflict free.

First we'll need to update our local `main` branch since the `great` PR was merged into the remote `main` branch.
```bash
# Update
git checkout main
git pull origin main
```

Now we want to update our `fantastic` branch with the updated local `main` branch but we've already made changes in the `fantastic` branch so we need to stash them first.

```bash
# Stash changes
git checkout fantastic
git stash
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/stash.png" style="border-radius: 7px;">
</div>

#### Rebase

This makes the `fantastic` branch the same as the previous `main` branch it was forked from so we can now rebase to make `fantastic` catch up to the latest version of the `main` branch.

```bash
# Rebase
git rebase main
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/rebase.png" style="border-radius: 7px;">
</div>

While still in the `fantastic` branch, we can reapply the changes we stashed on top of the updated branch.

```bash
# Apply stash
git stash list  # view all available stashes
git stash apply 0  # apply a saved stash
```

This time we'll have to resolve conflicts directly in our IDE (note that VS Code provides a simple button to accept the incoming change).

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/conflict_vs.png" style="border-radius: 7px;">
</div>

Once we accept the incoming change, we now have some uncommitted changes on our `fantastic` branch which we'll add, commit and push. Now when we perform a PR with the `main` branch, there will be no further conflicts to resolve.

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/apply.png" style="border-radius: 7px;">
</div>

Once we're done using the stash, we can drop it to keep things clean.

```bash
# Drop stash
git stash drop 0  # remove the applied stash (optional)
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/drop.png" style="border-radius: 7px;">
</div>


The stashing and rebasing resolution process is similar as the GitHub interface in that we need to choose which content to keep between the two branch conflicts but it is a nice way to keep our branch updated with the latest releases as we develop.

#### Squash

Rebase is also useful for squashing commits if we have many of them lined up before pushing to our remote host.
```bash
# Squash commits
git rebase -i origin/main
```
This will open up an interactive text editor where we can choosing which commits to squash (replace `pick` with `squash`) and after saving another text editor will appear to allow us to create a summarizing commit message. We can also do this on the online Git interface before merging the pull request.

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/squash.png">
</div>

### Inspection

Git allows us to inspect the current and previous states of our work at many different levels. Let's explore the most commonly used commands.

#### Status

We've used the status command quite a bit already as it's very useful to quickly see the status of our working tree.

```bash
# Status
git status
git status -s  # short format
```

#### Log

If we want to see the log of all our commits, we can do so using the log command. We can also do the same by inspecting specific branch histories on the Git online interface.

```bash
# Log
git log
git log --oneline  # short version
```
<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/log.png" style="border-radius: 7px;">
</div>

!!! note
    Commit IDs are 40 characters long but we can represent them with the first few (seven digits is the default for a Git SHA). If there is ambiguity, Git will notify us and we can simply add more of the commit ID.

#### Diff

If we want to know the difference between two commits, we can use the diff command.

```bash
# Diff
git diff  # all changes between current working tree and previous commit
git diff <COMMIT_A> <COMMIT_B>  # diff b/w two commits
git diff <COMMIT_A>:<PATH_TO_FILE> <COMMIT_B>:<PATH_TO_FILE>  # file diff b/w two commits
```

<div class="ai-center-all">
    <img width="350" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/diff.png" style="border-radius: 7px;">
</div>

#### Blame

One of the most useful inspection commands is blame, which allows us to see what commit was responsible for every single line in a file.
```bash
# Blame
git blame <PATH_TO_FILE>
git blame -L 1,3 <PATH_TO_FILE>  # blame for lines 1 and 3
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/blame.png" style="border-radius: 7px;">
</div>


### Time travel

Sometimes we may have done something we wish we could change. It's not always possible to do this in life, but in the world of Git, it is!

#### Restore

Sometimes we may just want to undo adding or staging a file, which we can easily do with the *restore* command.
```bash
# Restore
git restore -- <PATH_TO_FILE> <PATH_TO_FILE> # will undo any changes
git restore --stage <PATH_TO_FILE>  # will remove from stage (won't undo changes)
```

#### Reset

Now if we already made the commit but haven't pushed to remote yet, we can reset to the previous commit by moving the branch pointer to that commit. Note that this will undo all changes made since the previous commit.
```bash
# Reset
git reset <PREVIOUS_COMMIT_ID>  # or HEAD^
```

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/reset.png">
</div>

!!! note
    `HEAD` is a quick way to refer to the previous commit. Both `HEAD` and any previous commit ID can be accompanied with a `^` or `~` symbol which acts as a relative reference. <COMMIT_ID>`^`n refers to the nth parent of the commit while <COMMIT_ID>`~`n refers to the nth grandparent. Of course we can always just explicitly use commit IDs but these short hands can come in handy for quick checks without doing `git log` to retrieve commit IDs.

#### Revert

But instead of moving the branch pointer to a previous commit, we can continue to move forward by adding a new commit to revert certain previous commits.

```bash
# Revert
git revert <COMMIT_ID> ...  # rollback specific commits
git revert <COMMIT_TO_ROLLBACK_TO>..<COMMIT_TO_ROLLBACK_FROM>  # range
```

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/revert.png">
</div>

#### Checkout

Sometimes we may want to temporarily switch back to a previous commit just to explore or commit some changes. It's best practice to do this in a separate branch and if we want to save our changes, we need to create a separate PR. Note that if you do checkout a previous commit and submit a PR, you may override the commits in between.
```bash
# Checkout
git checkout -b <BRANCH_NAME> <COMMIT_ID>
```

<div class="ai-center-all">
    <img width="550" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/git/checkout.png" style="border-radius: 7px;">
</div>


### Best practices
There so many different works to work with git and sometimes it can became quickly unruly when fellow developers follow different practices. Here are a few, widely accepted, best practices when it comes to working with commits and branches.

#### Commits
- Commit often such that each commit has a clear associated change which you can approve / rollback.
- Try and [squash](#squash) commits if you have multiple before pushing to the remote host.
- Avoid monolithic commits (even if you regularly stash and rebase) because it can cause many thigns to break and creates a code review nightmare.
- Attach meaningful messages to commits so developers know exactly what the PR entails.
- Use tags to represent meaningful and stable releases of your application.
```bash
# Tags
git tag -a v0.1 -m "initial release"
```
- Don't delete commit history (reset), instead use [revert](#revert) to rollback and provide reasoning.

#### Branches
- Create branches when working on a feature, bug, etc. because it makes adding and reverting to the `main` branch very easy.
- Avoid using cryptic branch names.
- Maintain your `main` branch as the "demo ready" branch that always works.
- Protect branches with [rules](https://docs.github.com/en/github/administering-a-repository/managing-a-branch-protection-rule){:target="_blank"} (especially the `main` branch).

<!-- Citation -->
{% include "cite.md" %}