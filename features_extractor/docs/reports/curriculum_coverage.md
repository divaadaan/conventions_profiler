# Curriculum Coverage Report

Source: `docs/curriculum_mapping.yaml` (version 0.1-draft, locale en-CA, status partial-draft)

## Capitalization

| Curriculum Item | Level | Coverage | Detectors |
|---|---|---|---|
| Mid-sentence unexpected capitals (negative inverse signal) | low | covered (medium confidence) | new_feature:`cap_intrusion_rate` (medium) |
| Capital letters at start of sentences | low | covered (high confidence) | lt_rule:`UPPERCASE_SENTENCE_START` (high) · new_feature:`cap_sentence_initial_rate` (high) |
| Capital letters for proper nouns | medium | covered (high confidence) | lt_category:`CASING` (medium) · new_feature:`cap_proper_noun_rate` (high) |
| Capitals for words in titles | medium | GAP | _(none)_ |
| Capital letters in dialogue | high | GAP | _(none)_ |
| Capitals for proper adjectives | high | GAP | lt_category:`CASING` (low) |

## Punctuation

| Curriculum Item | Level | Coverage | Detectors |
|---|---|---|---|
| Punctuation marks at end of sentences | low | covered (high confidence) | spacy_signal:`sentence.last_alpha_token followed by terminal punct` (high) · new_feature:`bound_missing_term_punct_count` (high) |
| Apostrophes for contractions | medium | covered (high confidence) | lt_rule:`EN_CONTRACTION_SPELLING` (high) · lt_rule:`ITS_IT_S` (high) |
| Apostrophes for possessives | medium | covered (medium confidence) | lt_rule:`POSSESSIVE_APOSTROPHE` (medium) · lt_rule:`PLURAL_APOSTROPHE` (medium) |
| Commas to separate items in lists | medium | covered (medium confidence) | lt_rule:`SERIAL_COMMA` (medium) · lt_category:`PUNCTUATION` (low) |
| Quotation marks for direct speech | medium | covered (medium confidence) | lt_rule:`EN_UNPAIRED_QUOTES` (medium) |
| Colons in formal letters/memos | high | GAP | _(none)_ |
| Colons for introducing lists | high | GAP | _(none)_ |
| Commas for appositives/participles | high | GAP | lt_category:`PUNCTUATION` (low) |
| Commas to set off conjunctive adverbs | high | GAP | _(none)_ |
| Commas for nouns of direct address | high | GAP | lt_category:`PUNCTUATION` (low) |
| Commas after transitional words | high | covered (medium confidence) | lt_rule:`COMMA_AFTER_INTRODUCTORY_PHRASE` (medium) |
| Ellipses or dashes for pauses/breaks | high | GAP | _(none)_ |
| Quotes for title of short work | high | GAP | _(none)_ |
| Semicolons for independent clauses | high | GAP | _(none)_ |

## Syntax

| Curriculum Item | Level | Coverage | Detectors |
|---|---|---|---|
| Simple sentences: declarative | low | covered (medium confidence) | new_feature:`syn_simple_sent_rate` (medium) · spacy_signal:`single ROOT verb + no clausal subordination` (medium) |
| Simple sentences: exclamatory | low | covered (high confidence) | spacy_signal:`terminal '!'` (high) |
| Simple sentences: imperative | low | covered (low confidence) | new_feature:`syn_simple_sent_rate` (low) |
| Simple sentences: interrogative | low | covered (medium confidence) | spacy_signal:`terminal '?' + WH-word or aux-fronting` (medium) |
| Complex sentences | high | covered (medium confidence) | new_feature:`syn_complex_sent_rate` (medium) |
| Complex sentences with adverbial clauses | high | covered (high confidence) | spacy_signal:`advcl dependency present` (high) · new_feature:`syn_advcl_rate` (high) |
| Complex sentences with prepositional clauses | high | covered (medium confidence) | spacy_signal:`prep dependency present` (medium) |
| Complex sentences with adjective or relative clauses | high | covered (high confidence) | spacy_signal:`relcl dependency present` (high) · new_feature:`syn_relcl_rate` (high) |
| Compound sentences | high | covered (medium confidence) | new_feature:`syn_compound_sent_rate` (medium) |
| Compound-complex sentences | high | covered (medium confidence) | new_feature:`syn_compound_complex_sent_rate` (medium) |
| Dependent clauses | high | covered (high confidence) | spacy_signal:`mark/advcl/ccomp/xcomp/acl/relcl dependents` (high) |
| Sentence fragments and run-on sentences | high | covered (high confidence) | spacy_signal:`dep_fragment_count` (high) · lt_rule:`COMMA_COMPOUND_SENTENCE` (high) · lt_rule:`COMMA_COMPOUND_SENTENCE_2` (high) · new_feature:`bound_boundary_error_rate` (high) · new_feature:`err_simple_sent_error_rate` (medium) · new_feature:`err_complex_sent_error_rate` (medium) |
| Independent clauses | high | covered (high confidence) | spacy_signal:`ROOT verb + nsubj count` (high) |

## Grammar

| Curriculum Item | Level | Coverage | Detectors |
|---|---|---|---|
| Common adjectives | low | covered (high confidence) | spacy_signal:`POS=ADJ tagging` (high) |
| Common nouns | low | covered (high confidence) | spacy_signal:`POS=NOUN tagging` (high) |
| Plural nouns | low | covered (low confidence) | lt_category:`GRAMMAR` (low) |
| The pronoun "I" | low | covered (high confidence) | lt_rule:`I_LOWERCASE` (high) · lt_category:`CASING` (medium) |
| Singular nouns | low | covered (medium confidence) | lt_rule:`HE_VERB_AGR` (medium) · lt_rule:`PEOPLE_VBZ` (medium) |
| Abstract nouns | medium | GAP | _(none)_ |
| Collective nouns | medium | GAP | lt_category:`GRAMMAR` (low) |
| Comparative adjectives | medium | covered (high confidence) | lt_rule:`COMP_THAN` (medium) · lt_rule:`LESS_MORE_THEN` (high) |
| Interjections | medium | covered (high confidence) | spacy_signal:`POS=INTJ` (high) |
| Interrogative adjectives | medium | GAP | _(none)_ |
| Personal pronouns: object | medium | covered (medium confidence) | lt_category:`GRAMMAR` (medium) |
| Personal pronouns: possessive | medium | covered (high confidence) | lt_rule:`YOUR_YOU_RE` (high) · lt_rule:`THEIR_IS` (high) · lt_rule:`ITS_TO_IT_S` (high) |
| Personal pronouns: subject | medium | covered (medium confidence) | lt_rule:`PRP_VBZ` (medium) · lt_rule:`PRP_PAST_PART` (medium) |
| Possessive nouns | medium | covered (medium confidence) | lt_rule:`POSSESSIVE_APOSTROPHE` (medium) · lt_rule:`PLURAL_APOSTROPHE` (medium) |
| Pronouns as adjectives | medium | GAP | _(none)_ |
| Superlative adjectives | medium | GAP | lt_category:`GRAMMAR` (low) |
| Progressive tense | medium | covered (high confidence) | spacy_signal:`aux + VBG` (high) |
| Active vs. Passive voice | high | covered (high confidence) | spacy_signal:`nsubjpass + auxpass` (high) |
| Adverbial phrases | high | covered (medium confidence) | spacy_signal:`advmod with multiple tokens / advcl` (medium) |
| Recognition of words that adverbs modify | high | covered (high confidence) | spacy_signal:`advmod dependency` (high) |
| Common conjunctions | high | covered (high confidence) | spacy_signal:`POS=CCONJ or SCONJ` (high) |
| Coordinating conjunctions | high | covered (high confidence) | spacy_signal:`cc dependency` (high) · lt_rule:`COMMA_COMPOUND_SENTENCE` (high) |
| Correlative conjunctions | high | GAP | _(none)_ |
| Direct objects for nouns | high | covered (high confidence) | spacy_signal:`dobj dependency present` (high) |
| Gerunds as nouns | high | covered (medium confidence) | spacy_signal:`VBG with NOUN-like dependency role (nsubj/dobj)` (medium) |
| Imperative verbs | high | covered (medium confidence) | spacy_signal:`bare-form verb sentence-initial, no nsubj` (medium) |
| Indirect objects | high | covered (medium confidence) | spacy_signal:`iobj/dative dependency` (medium) |
| Intensive pronouns | high | GAP | _(none)_ |
| Interrogative verbs | high | GAP | lt_category:`GRAMMAR` (low) |
| Linking verbs | high | covered (high confidence) | spacy_signal:`cop dependency` (high) |
| Participles | high | covered (high confidence) | spacy_signal:`acl dependency with VBG/VBN` (high) |
| Predicate adjectives | high | covered (high confidence) | spacy_signal:`acomp dependency on copula/linking verb` (high) |
| Predicate nouns | high | covered (medium confidence) | spacy_signal:`attr/acomp dependency on copula` (medium) |
| Prepositional phrases | high | covered (high confidence) | spacy_signal:`prep + pobj` (high) |
| Prepositions | high | covered (high confidence) | spacy_signal:`POS=ADP / dep=prep` (high) · lt_category:`GRAMMAR` (medium) |
| Proper adjectives | high | GAP | lt_category:`CASING` (low) |
| Reflexive pronouns | high | GAP | lt_category:`GRAMMAR` (low) |
| Relative pronouns | high | covered (high confidence) | spacy_signal:`relcl dependency with relative-pronoun head` (high) · lt_category:`GRAMMAR` (medium) |
| Subordinating conjunctions | high | covered (high confidence) | spacy_signal:`mark dependency` (high) · lt_rule:`THE_CC` (medium) |
| Verb tense: future | high | covered (high confidence) | lt_rule:`MD_BASEFORM` (high) |
| Verb tense: past | high | covered (high confidence) | spacy_signal:`VBD/VBN tagging` (high) · lt_rule:`PRP_PAST_PART` (medium) |
| Perfect tense | high | covered (high confidence) | spacy_signal:`have/has/had + VBN` (high) · lt_rule:`PRP_PAST_PART` (medium) |
| Verb tense: present | high | covered (high confidence) | lt_rule:`HE_VERB_AGR` (high) · lt_rule:`PEOPLE_VBZ` (high) |
| Forms of the verb "to be" | high | covered (high confidence) | lt_rule:`HE_VERB_AGR` (high) · lt_rule:`AGREEMENT_SENT_START` (high) · lt_rule:`SINGULAR_AGREEMENT_SENT_START` (high) · lt_rule:`THE_ARE` (medium) |

## Summary

- Total mapped curriculum items: **77**
- Items with no current detector (gap): **21**
- Estimated coverage: **73%**
