# Insanely Mystical Whisper

## Description
It's an adaptation of [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) for https://mystic.ai

At the moment, it's using v2, as v3 is [reported to be worse (?)](https://deepgram.com/learn/whisper-v3-results).

Usage:
0. Get access to [Catalyst API](https://docs.mystic.ai/docs/getting-started) 
1. From [Docs](https://docs.mystic.ai/docs/getting-started): `pip install pipeline-ai`, `pipeline cluster login catalyst-api API_TOKEN -u https://www.mystic.ai -a`
2. Rename `env.example` to `env` and fill in the information
3. `pdm install` ~~(preferred to pip, sorry)~~
4. `pdm run start` will run the commands ~~(there's one, but it's automatic)~~
