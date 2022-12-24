# GitHub: How to make a fork of public repository private?
Based on a stackoverflow answer by [Martin Konicek](https://stackoverflow.com/users/90998/martin-konicek)

https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private/30352360#30352360

# Create an empty private repository from Github UI.

# Duplicate the public repository to the private repository
```bash
git clone --bare https://github.com/lsw9021/MASS.git
cd MASS.git
git push --mirror https://github.com/NJITBioDynamics/MASS.git
cd ..
rm -rf MASS.git
```

# Clone the private repository in a folder
```bash
git clone https://github.com/NJITBioDynamics/MASS.git
```

# Pull the latest commits from upstream and VSFork
```bash
git remote add upstream https://github.com/lsw9021/MASS.git
```
To avoid accidentally push to the upstream:
```bash
git remote set-url --push upstream no_push
```
You can check if the remote repository and no_push are set up successfully
```bash
git remote -v
```
If you need to fetch new update from the upstream:
```bash
git fetch upstream
git rebase upstream/master
```

You can fetch changes or commits from others' forks as wekl, e.g.
```bash
git remote add VSFork https://github.com/ValentinSiderskiyPhD/MASS.git
git remote -v
git fetch VSFork
git rebase VSFork/master
```

# Push new changes to the private repository
Make sure you are checking out the origin/master or the branch you want to push
```bash
git push
```

# Create a pull request to the public repo:
Use the GitHub UI to create a fork of the public repo (the small "Fork" button at the top right of the public repo page). Then:

```bash
git clone https://github.com/lsw9021/MASS.git
cd MASS
git remote add NJIT_MASS https://github.com/NJITBioDynamics/MASS.git
git checkout -b pull_request_NJIT_MASS
git pull NJIT_MASS master
git push origin pull_request_NJIT_MASS
```
Now you can create a pull request via the Github UI for public-repo, as described [here](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).
Once project owners review your pull request, they can merge it.
Of course the whole process can be repeated (just leave out the steps where you add remotes).


