# Claim Verification
This folder contains code logic that enables quote verification.

## Approach

1. Get the claim from the claim_extractor (Interface needs to be clarified with Yufei. Currently, just read from the `claims.json`). Is done by `ClaimVerifier`

2. For each of the claims perform internet search. Is done by the `SearchManager` using `perplexity`. The resutls include found information alongside with the sources

3. The Information found in the internet is returned to the `ClaimVerifier`. The verification is performed by the `OpenAI` model. Results will be stored in the `verification_response.json` on the repositorys root level

## Future work
* Include source reliability

* Improve the claim verification

* Create one pipileine: starting with the the content extraction, ending with claim verification.

* Unify the schemes with Yufei
