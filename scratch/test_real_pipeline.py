import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from src.article_extractor import extract_article_from_url
from src.claim_extractor import extract_claims
from src.search_provider import TavilySearchProvider, is_real_search_configured
from src.evidence_retriever import retrieve_evidence_for_claims
from src.source_evaluator import evaluate_claim_evidence
from src.claim_analyzer import synthesize_claim_stance, generate_article_report

url = "https://m.rediff.com/news/report/chaos-at-jantar-mantar-as-cops-lathi-charge-cjp-protesters/20260720.htm"

print("==================================================")
print("STEP 1: ARTICLE EXTRACTION")
print("==================================================")
ext_res = extract_article_from_url(url)
assert ext_res.success, f"Extraction failed with status: {ext_res.status}"
art = ext_res.article
print(f"Title: {art.title}")
print(f"Source / Domain: {art.domain}")
print(f"Canonical URL: {art.canonical_url}")
print(f"Final URL: {art.final_url}")
print(f"Word Count: {art.word_count}")
print(f"Character Count: {art.character_count}")

print("\n==================================================")
print("STEP 2: CLAIM EXTRACTION")
print("==================================================")
claims = extract_claims(art.text, max_claims=5)
for c in claims:
    print(f"Claim ID: {c.claim_id}")
    print(f"Sentence Index: {c.source_sentence_index}")
    print(f"Priority Score: {c.metadata.get('priority_score')}")
    print(f"Claim Text: \"{c.text}\"\n")

print("==================================================")
print("STEP 3 & 4: REAL WEB SEARCH & SOURCE COLLECTION")
print("==================================================")
assert is_real_search_configured(), "Real search provider is NOT configured!"
provider = TavilySearchProvider()
print(f"Using Provider: {provider.name} (is_real={provider.is_real})")

claims_evidence = retrieve_evidence_for_claims(claims, provider=provider, max_sources_per_claim=3)

for cr in claims_evidence:
    print(f"\n--- Claim {cr.claim.claim_id} ---")
    print(f"Query Used: \"{cr.search_query.query_text}\"")
    print(f"Status: {cr.status.value}")
    print(f"Sources Retrieved: {len(cr.evidence_items)}")
    for ev in cr.evidence_items:
        print(f"  [{ev.evidence_id}] {ev.title} | {ev.domain} | Date: {ev.publication_date}")
        print(f"       URL: {ev.url}")
        print(f"       Snippet: {ev.snippet[:120]}...")

print("\n==================================================")
print("STEP 5: SOURCE QUALITY & PROVENANCE EVALUATION (PHASE 6)")
print("==================================================")
evaluations = {}
for cr in claims_evidence:
    ev_summary = evaluate_claim_evidence(cr)
    evaluations[cr.claim.claim_id] = ev_summary
    print(f"\nClaim {ev_summary.claim_id} Quality Summary:")
    print(f"  Primary: {ev_summary.primary_source_count}, Independent: {ev_summary.independent_source_count}, Opinion: {ev_summary.opinion_source_count}, Duplicate Wire: {ev_summary.duplicate_wire_count}")
    print(f"  Unique Domains: {ev_summary.unique_domain_count} ({', '.join(ev_summary.unique_domains)})")
    print(f"  Note: {ev_summary.quality_assessment_note}")
    for item in ev_summary.evaluated_items:
        e = item.evidence
        q = item.evaluation
        print(f"    [{e.evidence_id}] Quality: {q.quality_score}/100 | {q.category.value} | Status: {q.independence_status.value} | Relevance: {q.relevance_level.value} | Strength: {q.evidence_strength.value}")
        if q.demotion_reason:
            print(f"         Notice: {q.demotion_reason}")
        if q.ranking_reasons:
            print(f"         Ranking Rationale: {'; '.join(q.ranking_reasons[:3])}")

print("\n==================================================")
print("STEP 6, 7 & 8: CLAIM/EVIDENCE STANCE & ROBUSTNESS ANALYSIS")
print("==================================================")
claim_analyses = []
for claim in claims:
    ev_summary = evaluations.get(claim.claim_id)
    items = ev_summary.evaluated_items if ev_summary else []
    analysis = synthesize_claim_stance(claim, items)
    claim_analyses.append(analysis)
    print(f"\nClaim {analysis.claim_id} Verdict:")
    print(f"  Text: \"{analysis.claim_text}\"")
    print(f"  Importance: {analysis.importance.value}")
    print(f"  Final Stance: {analysis.final_stance.value}")
    print(f"  Evidence Strength: {analysis.evidence_strength.value}")
    print(f"  Strength Note: {analysis.strength_explanation}")
    if analysis.disputed_summary:
        print(f"  Dispute Notice: {analysis.disputed_summary}")
    print(f"  Supporting IDs: {analysis.supporting_evidence_ids}")
    print(f"  Contradicting IDs: {analysis.contradicting_evidence_ids}")
    print(f"  Neutral IDs: {analysis.neutral_evidence_ids}")
    print(f"  Explanation: {analysis.explanation}")
    if analysis.evidence_limitations:
        print(f"  Limitations: {analysis.evidence_limitations}")

print("\n==================================================")
print("STEP 10: OVERALL ARTICLE ASSESSMENT & DIVERSITY SUMMARY")
print("==================================================")
report = generate_article_report(claim_analyses)
print(f"Overall Result: {report.overall_assessment.value}")
print(f"Summary Explanation: {report.summary_explanation}")
print(f"Total Claims: {report.total_claims}")
print(f"Supported Claims: {report.supported_claims}")
print(f"Contradicted Claims: {report.contradicted_claims}")
print(f"Uncertain Claims: {report.uncertain_claims}")
print(f"High Importance Claims: {report.high_importance_claims}")
print(f"Total Unique Domains: {report.total_unique_domains}")
print(f"Total Independent Sources: {report.total_independent_sources}")
print(f"Total Primary Sources: {report.total_primary_sources}")
print(f"Total Syndicated Sources: {report.total_syndicated_sources}")
print(f"Total Opinion Sources: {report.total_opinion_sources}")
print(f"Key Limitations: {report.key_limitations}")
