# Curriculum Gap Audit

Curriculum items with no current detector. Grouped by level — low-level gaps are highest priority since the curriculum places more weight on early-mastered concepts.

## Low level — 0 gap(s)

_No gaps at this level._

## Medium level — 6 gap(s)

- **Capitals for words in titles** _(capitalization)_ — example: The book was called The Journey across Canada.
  - Title detection is hard without explicit markup. Out of scope for now.
- **Abstract nouns** _(grammar)_ — example: peace, joy, imagination, reconciliation
  - No error detector exists for misuse of abstract nouns specifically. Would require lexicon-based usage feature; low priority.
- **Collective nouns** _(grammar)_ — example: fleet, bunch, cluster, batch
- **Pronouns as adjectives** _(grammar)_ — example: Neither came at the time of the meeting.
- **Superlative adjectives** _(grammar)_ — example: They use the smallest bag to carry their groceries.
- **Interrogative adjectives** _(grammar)_ — example: what, which, whose

## High level — 15 gap(s)

- **Capitals for proper adjectives** _(capitalization)_ — example: They like to eat Indian, Korean, and Japanese food.
  - Possible new feature; spaCy NER on adjectival use of nationality/place terms.
- **Capital letters in dialogue** _(capitalization)_ — example: The woman exclaimed, 'We are so happy that you enjoyed your day.'
  - No reliable detector. Would require quotation-aware sentence segmentation.
- **Quotes for title of short work** _(punctuation)_ — example: "The Peaceful Day" is a short story.
- **Commas for nouns of direct address** _(punctuation)_ — example: Dad, where is my baseball bat?
  - Likely under-covered. Vocative comma detection is tricky.
- **Commas for appositives/participles** _(punctuation)_ — example: The store owner, who sells fruit and vegetables, has one child.
  - Could be addressed via spaCy: detect appositive (appos) or participial (acl) modifiers and check comma boundaries.
- **Colons for introducing lists** _(punctuation)_ — example: There were many colours in the painting: red, purple, and green.
- **Colons in formal letters/memos** _(punctuation)_ — example: To whom it may concern:
- **Semicolons for independent clauses** _(punctuation)_ — example: I went to the market; I forgot to buy corn soup.
- **Commas to set off conjunctive adverbs** _(punctuation)_ — example: Your dog got into the neighbour's yard; in addition, he dug up the flowers.
- **Ellipses or dashes for pauses/breaks** _(punctuation)_ — example: Yesterday he enrolled...in art instead of music.
  - Stylistic — not necessarily an error to grade against.
- **Intensive pronouns** _(grammar)_ — example: He himself was responsible for his success in sports.
- **Reflexive pronouns** _(grammar)_ — example: She bought herself a mango.
  - Reflexive misuse (e.g., "myself" for "me") sometimes flagged but coverage thin.
- **Interrogative verbs** _(grammar)_ — example: Did you break my favourite toy?
- **Proper adjectives** _(grammar)_ — example: Canadian provinces
  - Same gap as cap_proper_adjective; share new-feature work.
- **Correlative conjunctions** _(grammar)_ — example: Either we walk to school or we ride our bicycles.
  - Either/or, neither/nor, both/and parallelism — no current detector. Possible spaCy-based feature.

## Summary

- Total gaps: **21**
- Low-level gaps: **0** (highest priority — early-mastered concepts)
- Medium-level gaps: **6**
- High-level gaps: **15**
