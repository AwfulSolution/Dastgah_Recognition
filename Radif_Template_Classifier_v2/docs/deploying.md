# Deploying

Two hosts, because the analysis does not fit in one.

The front end is a Vite build of 174 KB gzipped and belongs on a CDN. The API
imports librosa, which imports numba, which imports llvmlite: **268 MB of
dependencies before any code**, against Vercel's 250 MB serverless limit. The
largest single item is llvmlite at 130 MB and it cannot be dropped, because
pYIN -- the pitch tracker the whole classifier rests on -- runs through numba.
So the API is a container on Fly.io and the front end on Vercel proxies to it.

Serialising the pair models as coefficients rather than pickled estimators
removed scikit-learn, 37 MB, from the API image. It is still 18 MB too large
for a serverless function.

## API, on Fly.io

```bash
cd Radif_Template_Classifier_v2
fly launch --no-deploy          # once; accept the existing fly.toml
fly deploy
fly secrets set ALLOWED_ORIGINS="https://<your-app>.vercel.app"
```

`fly.toml` keeps one machine warm (`min_machines_running = 1`). That is
deliberate: a cold start pays numba's JIT warm-up on top of the analysis, so the
first upload after an idle period would otherwise take far longer than the rest.

Check it:

```bash
curl https://dastgah-classifier.fly.dev/api/health
curl -F "file=@some-recording.mp3" https://dastgah-classifier.fly.dev/api/analyze
```

## Front end, on Vercel

`vercel.json` builds `web/` and rewrites `/api/*` to the Fly app, so the browser
makes same-origin requests and CORS only has to allow the Vercel origin.

```bash
npx vercel            # preview
npx vercel --prod     # production
```

There is no `vercel` CLI installed on this machine; `npx` fetches it. A
`netlify-cli` is installed, and Netlify works the same way if preferred --
publish `web/dist`, with a redirect from `/api/*` to the Fly URL.

**Change the Fly hostname in `vercel.json`** if `fly launch` assigns a different
app name than `dastgah-classifier`, or the front end will proxy to nothing.

## What the deployed thing answers

The untrained template classifier, over six dastgahs with avazes folded into
their mothers. 74.9% on Nava's 1,568 recordings, 89.0% on those over three
minutes, against a 16.7% chance line.

Alongside it, when the top two are a confusable pair, a model trained on only
that pair reports its own reading and whether it agrees. **That is not the
answer and the interface says so.** Those models gain 1.8 points on the corpus
they were fitted to and lose on every other one, and an upload from an
unfamiliar source is exactly that case. They are carried because their
disagreements mark the recordings where a listener's ear decides something this
project's data cannot.
