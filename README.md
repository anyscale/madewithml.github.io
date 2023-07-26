# Made With ML

https://madewithml-staging.github.io

### Set up
```bash
pyenv local 3.10.11
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Serve docs
```bash
python3 -m mkdocs serve
```

### Production
```bash
git add .
git commit -m "message"
git push origin master
```

### Staging
```bash
mkdocs gh-deploy --force
```

### Newsletter
1. `SendInBlue code`, `SPF` and `DKIM` records from SendInBlue to Cloudflare
2. Cloudflare Page Rules redirects `newsletter.madewithml.com` to `madewithml.com/misc/newsletter`
3. `docs/index.md`, `docs/misc/newsletter.md` and `docs/overrides/newsletter.html` all have newsletter links that need to be changed
4. mkdocs.yml has the `Subscribe` button going to `misc/newsletter.md`
5. We also have the related pages `docs/misc/confirmation.md` and `docs/misc/subscribed.md`
