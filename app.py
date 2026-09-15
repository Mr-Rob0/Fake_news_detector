import streamlit as st
from src.input_handler import validate_and_normalize_input
from src.article_extractor import extract_article_from_url, ExtractionStatus
from src.claim_extractor import extract_claims, Claim
from src.search_provider import (
    SearchProvider,
    SearchStatus,
    MockSearchProvider,
    TavilySearchProvider,
    get_default_search_provider,
    is_real_search_configured,
)
from src.evidence_retriever import (
    retrieve_evidence_for_claims,
    ClaimEvidenceResult,
)
from src.source_evaluator import (
    evaluate_claim_evidence,
    ClaimEvaluationSummary,
    SourceCategory,
    IndependenceStatus,
    RelevanceLevel,
    EvidenceStrength,
)
from src.claim_analyzer import (
    ClaimImportance,
    EvidenceStance,
    ClaimStance,
    OverallAssessment,
    ClaimAnalysisResult,
    ArticleVerificationReport,
    synthesize_claim_stance,
    generate_article_report,
)

st.set_page_config(
    page_title="AI News Verification System",
    page_icon="📰",
    layout="wide",
)

st.title("AI News Verification System")
st.caption("Phase 9: Domain-Agnostic Evidence-Based News Verification Pipeline")

st.markdown(
    """
    This system verifies news stories and statements through an end-to-end evidence pipeline:
    extracting factual claims, retrieving external sources, evaluating provenance and 
    independence, and delivering explainable verdicts (**SUPPORTED**, **CONTRADICTED**, or **UNCERTAIN**).
    """
)

# Sidebar Configuration
with st.sidebar:
    st.header("Pipeline Configuration")
    st.info("🔧 **Current Phase:** Phase 9 (Full Verification Pipeline)")

    st.markdown("### Search Provider Status")
    real_available = is_real_search_configured()

    if real_available:
        st.success("🟢 **Search Provider: Real Web Search (Tavily API)**")
        st.caption("Live external web queries are executed against the Tavily Search API.")
        force_mock = st.checkbox("Use Demo / Mock Provider instead", value=False)
        active_provider = get_default_search_provider(prefer_real=not force_mock)
    else:
        st.warning("🟡 **Search Provider: Demo / Mock — no API key configured**")
        st.caption(
            "To enable real web search, set `TAVILY_API_KEY` in your environment or in a `.env` file. "
            "The system is currently using deterministic mock results for offline demonstration."
        )
        active_provider = get_default_search_provider(prefer_real=False)

    st.divider()
    max_claims = st.slider("Max Claims to Extract", min_value=1, max_value=5, value=3)
    max_sources = st.slider("Max Sources per Claim", min_value=1, max_value=5, value=3)

st.divider()

# Input controls
col1, col2 = st.columns([1, 1])
with col1:
    news_text = st.text_area(
        label="Paste News / Article Text",
        placeholder="Enter the news content or statement here (minimum 20 characters)...",
        height=180,
    )

with col2:
    article_url = st.text_input(
        label="Article URL",
        placeholder="https://example.com/news-article",
    )
    st.caption("Provide either news text OR an article URL, then click Verify News.")

verify_button = st.button("Verify News", type="primary")


def render_overall_assessment(report: ArticleVerificationReport) -> None:
    """Displays the top-level article assessment card with summary metrics, diversity stats, and limitations."""
    st.markdown("---")
    st.markdown("## Overall Article Assessment")

    # Assessment Badge & Color Coding
    if report.overall_assessment == OverallAssessment.LIKELY_SUPPORTED:
        st.success(f"### 🟢 LIKELY SUPPORTED\n\n**Verdict Summary:** {report.summary_explanation}")
    elif report.overall_assessment == OverallAssessment.LIKELY_CONTRADICTED:
        st.error(f"### 🔴 LIKELY CONTRADICTED\n\n**Verdict Summary:** {report.summary_explanation}")
    elif report.overall_assessment == OverallAssessment.MIXED:
        st.warning(f"### 🟠 MIXED VERACITY\n\n**Verdict Summary:** {report.summary_explanation}")
    else:
        st.info(f"### 🔵 INSUFFICIENT EVIDENCE (UNCERTAIN)\n\n**Verdict Summary:** {report.summary_explanation}")

    # Summary Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Claims", report.total_claims)
    m2.metric("Supported", report.supported_claims)
    m3.metric("Contradicted", report.contradicted_claims)
    m4.metric("Uncertain", report.uncertain_claims)
    m5.metric("High Importance", report.high_importance_claims)

    # Phase 8 Source Diversity & Provenance Summary
    st.markdown("#### Source Diversity & Provenance Summary")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Unique Domains", report.total_unique_domains)
    d2.metric("Independent Sources", report.total_independent_sources)
    d3.metric("Primary / Official", report.total_primary_sources)
    d4.metric("Syndicated / Wire", report.total_syndicated_sources)
    d5.metric("Opinion / Commentary", report.total_opinion_sources)

    if report.key_limitations:
        with st.expander("⚠️ Key Evidence Limitations & Epistemic Boundaries", expanded=False):
            for lim in report.key_limitations:
                st.write(f"- {lim}")
            st.caption(
                "Notice: This assessment is strictly grounded in retrieved external sources. "
                "The system does not issue absolute truth claims or artificial confidence percentages."
            )


def render_claim_analysis_section(
    claims_analyses: list[ClaimAnalysisResult],
    claims_evidence: list[ClaimEvidenceResult],
    evaluations_dict: dict[str, ClaimEvaluationSummary],
    provider: SearchProvider,
) -> None:
    """Renders claim-by-claim stance analyses, traceable evidence IDs, and underlying sources."""
    st.markdown("---")
    st.markdown("## Claim-by-Claim Verification Analysis")

    provider_label = "Real Web Search (Tavily API)" if provider.is_real else "Demo / Mock Provider"
    provider_badge = "🟢" if provider.is_real else "🟡"
    st.caption(f"Evidence Source: {provider_badge} `{provider_label}` | Grounded Stance Comparison")

    ev_result_dict = {cr.claim.claim_id: cr for cr in claims_evidence}

    for analysis in claims_analyses:
        claim_id = analysis.claim_id
        eval_summary = evaluations_dict.get(claim_id)
        cr = ev_result_dict.get(claim_id)

        with st.container():
            # Header Row
            st.markdown(f"### Claim `{claim_id}`")
            st.info(f"**Assertion:** \"{analysis.claim_text}\"")
            if cr:
                st.caption(f"🔍 **Search Query Used:** `{cr.search_query.query_text}`")
                if cr.search_query.additional_queries:
                    st.caption(f"Complementary Queries: {', '.join([f'`{q}`' for q in cr.search_query.additional_queries[:2]])}")

            # Stance, Importance & Evidence Strength Badges
            col_imp, col_stance, col_str = st.columns([1, 1, 1])
            with col_imp:
                if analysis.importance == ClaimImportance.HIGH:
                    st.markdown("**Claim Importance:** 🔴 `HIGH (Central Assertion)`")
                elif analysis.importance == ClaimImportance.MEDIUM:
                    st.markdown("**Claim Importance:** 🟡 `MEDIUM (Corroborating Fact)`")
                else:
                    st.markdown("**Claim Importance:** ⚪ `LOW (Contextual Detail)`")

            with col_stance:
                if analysis.final_stance == ClaimStance.SUPPORTED:
                    st.markdown("**Verdict:** 🟢 `SUPPORTED`")
                elif analysis.final_stance == ClaimStance.CONTRADICTED:
                    st.markdown("**Verdict:** 🔴 `CONTRADICTED`")
                else:
                    st.markdown("**Verdict:** 🟡 `UNCERTAIN`")

            with col_str:
                if analysis.evidence_strength == EvidenceStrength.STRONG:
                    st.markdown("**Evidence Strength:** 🟢 `STRONG`")
                elif analysis.evidence_strength == EvidenceStrength.MODERATE:
                    st.markdown("**Evidence Strength:** 🟡 `MODERATE`")
                elif analysis.evidence_strength == EvidenceStrength.WEAK:
                    st.markdown("**Evidence Strength:** 🟠 `WEAK`")
                else:
                    st.markdown("**Evidence Strength:** ⚪ `INSUFFICIENT`")

            # Quantitative Evidence Counts & Integrity Row (Requirement 21)
            rel_cnt = analysis.relevant_sources_count
            filt_cnt = analysis.filtered_irrelevant_sources_count
            ind_cnt = analysis.independent_sources_count
            prim_cnt = analysis.primary_sources_count
            sup_cnt = len(analysis.supporting_evidence_ids)
            con_cnt = len(analysis.contradicting_evidence_ids)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Relevant", rel_cnt)
            c2.metric("Filtered (Irrel.)", filt_cnt)
            c3.metric("Independent", ind_cnt)
            c4.metric("Primary", prim_cnt)
            c5.metric("Supporting", sup_cnt)
            c6.metric("Contradicting", con_cnt)

            st.caption(
                "Consensus: "
                f"{analysis.direct_supporting_evidence_count} direct support / "
                f"{analysis.direct_contradicting_evidence_count} direct contradiction; "
                f"{analysis.independent_supporting_streams} independent supporting stream(s), "
                f"{analysis.independent_contradicting_streams} contradicting stream(s); "
                f"{analysis.syndicated_duplicate_count} duplicate or same-publisher result(s) discounted."
            )

            if analysis.disputed_summary:
                st.warning(f"⚖️ **Dispute Notice:** {analysis.disputed_summary}")

            # Categorized Evidence Breakdown
            items_by_id = {it.evidence.evidence_id: it for it in (eval_summary.evaluated_items if eval_summary else [])}
            filtered_items = eval_summary.filtered_irrelevant_items if eval_summary else []
            
            # Supporting Evidence Block
            if analysis.supporting_evidence_ids:
                st.markdown("##### 🟢 Supporting Evidence")
                for eid in analysis.supporting_evidence_ids:
                    item_w = items_by_id.get(eid)
                    if item_w:
                        ev = item_w.evidence
                        ed = item_w.evaluation
                        ind_txt = ed.independence_status.value
                        st.markdown(f"- **[{ev.title}]({ev.url})** ({ev.domain}) — Relevance: `{ed.relevance_level.value}` | Status: `{ind_txt}` | Quality: `{ed.quality_score}/100`")
                        st.caption(f'  Snippet: "{ev.snippet}"')
                        with st.expander(f"Why was [{ev.evidence_id}] considered relevant?", expanded=False):
                            st.write(f"- **Relevance Classification:** `{ed.relevance_level.value}`")
                            st.write(f"- **Source Category:** `{ed.category.value}` (Quality: `{ed.quality_score}/100`)")
                            for reason in ed.ranking_reasons:
                                st.write(f"- {reason}")
                    else:
                        st.markdown(f"- Source `{eid}`")

            # Contradicting Evidence Block
            if analysis.contradicting_evidence_ids:
                st.markdown("##### 🔴 Contradicting Evidence")
                for eid in analysis.contradicting_evidence_ids:
                    item_w = items_by_id.get(eid)
                    if item_w:
                        ev = item_w.evidence
                        ed = item_w.evaluation
                        ind_txt = ed.independence_status.value
                        st.markdown(f"- **[{ev.title}]({ev.url})** ({ev.domain}) — Relevance: `{ed.relevance_level.value}` | Status: `{ind_txt}` | Quality: `{ed.quality_score}/100`")
                        st.caption(f'  Snippet: "{ev.snippet}"')
                        with st.expander(f"Why was [{ev.evidence_id}] considered relevant / contradicting?", expanded=False):
                            st.write(f"- **Relevance Classification:** `{ed.relevance_level.value}`")
                            for reason in ed.ranking_reasons:
                                st.write(f"- {reason}")
                    else:
                        st.markdown(f"- Source `{eid}`")

            # Neutral / Context Evidence Block
            if analysis.neutral_evidence_ids:
                with st.expander(f"⚪ Neutral / Context Evidence ({len(analysis.neutral_evidence_ids)} sources)", expanded=False):
                    for eid in analysis.neutral_evidence_ids:
                        item_w = items_by_id.get(eid)
                        if item_w:
                            ev = item_w.evidence
                            ed = item_w.evaluation
                            st.markdown(f"- **[{ev.title}]({ev.url})** ({ev.domain}) — Relevance: `{ed.relevance_level.value}`")
                            st.caption(f'  Snippet: "{ev.snippet}"')
                            with st.expander(f"Why was [{ev.evidence_id}] considered relevant?", expanded=False):
                                st.write(f"- **Relevance Classification:** `{ed.relevance_level.value}`")
                                for reason in ed.ranking_reasons:
                                    st.write(f"- {reason}")
                        else:
                            st.markdown(f"- Source `{eid}`")

            # Filtered / Irrelevant Sources Expander (Requirement 21)
            if filtered_items:
                with st.expander(f"🛡️ Filtered Irrelevant Sources ({len(filtered_items)} items rejected)", expanded=False):
                    st.caption("These search results were filtered out at the Relevance Gate to prevent keyword-overlap false contradictions.")
                    for f_item in filtered_items:
                        f_ev = f_item.evidence
                        f_ed = f_item.evaluation
                        st.markdown(f"- **[{f_ev.title}]({f_ev.url})** ({f_ev.domain})")
                        st.caption(f'  Snippet: "{f_ev.snippet}"')
                        st.error(f"❌ **Why was this source rejected?** {f_ed.demotion_reason or 'Source does not address claim entity or subject.'}")
                        if f_ed.ranking_reasons:
                            with st.expander(f"Detailed rejection audit for [{f_ev.evidence_id}]", expanded=False):
                                for r in f_ed.ranking_reasons:
                                    st.write(f"- {r}")
                        st.markdown("---")

            # Why this verdict?
            st.markdown(f"**Why this verdict?** {analysis.explanation}")
            if analysis.strength_explanation:
                st.caption(f"Evidentiary Strength Note: {analysis.strength_explanation}")

            if analysis.evidence_limitations:
                with st.expander(f"Limitations for Claim {claim_id}", expanded=False):
                    for lim in analysis.evidence_limitations:
                        st.write(f"- {lim}")

            # Underlying Full Evidence Details Expander
            if eval_summary and eval_summary.evaluated_items:
                with st.expander(f"🔎 View Detailed Evidence Audit for Claim {claim_id} ({len(eval_summary.evaluated_items)} relevant sources)", expanded=False):
                    if cr:
                        st.caption(f"Generated Search Query: `{cr.search_query.query_text}`")
                        if cr.search_query.additional_queries:
                            st.caption(f"Multi-angle Queries: {', '.join(cr.search_query.additional_queries[:2])}")
                    for item_wrapper in eval_summary.evaluated_items:
                        item = item_wrapper.evidence
                        eval_data = item_wrapper.evaluation

                        cat_icon = "🏛️" if eval_data.is_primary else ("💭" if eval_data.category == SourceCategory.OPINION_EDITORIAL else "📰")

                        # Independence badge
                        if eval_data.independence_status == IndependenceStatus.DUPLICATE_WIRE:
                            ind_badge = "⚠️ `Duplicate Wire Copy`"
                        elif eval_data.independence_status == IndependenceStatus.SYNDICATED:
                            ind_badge = "🟡 `Syndicated Feed`"
                        elif eval_data.independence_status == IndependenceStatus.INDEPENDENT:
                            ind_badge = "🟢 `Independent`"
                        else:
                            ind_badge = "⚪ `General`"

                        # Relevance badge
                        if eval_data.relevance_level == RelevanceLevel.DIRECT:
                            rel_badge = "🎯 `Direct Relevance`"
                        elif eval_data.relevance_level == RelevanceLevel.PARTIAL:
                            rel_badge = "🔍 `Partial Context`"
                        elif eval_data.relevance_level == RelevanceLevel.IRRELEVANT:
                            rel_badge = "❌ `Irrelevant`"
                        else:
                            rel_badge = "⚪ `Weak Relevance`"

                        st.markdown(f"**[{item.evidence_id}]** {cat_icon} [{eval_data.category.value}] [{item.title}]({item.url}) — *Quality Score: {eval_data.quality_score}/100*")
                        st.markdown(f"> {item.snippet}")
                        st.markdown(f"**Provenance & Status:** {ind_badge} | {rel_badge} | Strength: `{eval_data.evidence_strength.value}`")

                        if eval_data.demotion_reason:
                            st.caption(f"⚠️ {eval_data.demotion_reason}")

                        if eval_data.ranking_reasons:
                            with st.expander(f"Why was [{item.evidence_id}] ranked this way?", expanded=False):
                                for reason in eval_data.ranking_reasons:
                                    st.write(f"- {reason}")
                        st.markdown("---")

        st.markdown("")


def render_verification_summary(report: ArticleVerificationReport) -> None:
    """Renders the final Verification Summary section with cautious epistemic framing."""
    st.markdown("---")
    st.markdown("## Verification Summary")
    st.caption("Assistance notice: Grounded evidence-assistance assessment, not an absolute truth oracle.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Factual Claims", report.total_claims)
    s2.metric("Supported Claims", report.supported_claims)
    s3.metric("Contradicted Claims", report.contradicted_claims)
    s4.metric("Uncertain Claims", report.uncertain_claims)

    p1, p2, p3 = st.columns(3)
    p1.metric("Evidence Sources", report.total_unique_domains)
    p2.metric("Independent Sources", report.total_independent_sources)
    p3.metric("Primary / Official Sources", report.total_primary_sources)

    st.markdown(f"### Overall Assessment: `{report.overall_assessment.value}`")
    st.info(f"**Synthesis:** {report.summary_explanation}")


if verify_button:
    result = validate_and_normalize_input(news_text=news_text, article_url=article_url)

    if not result.is_valid:
        st.warning(result.error_message)
    else:
        norm = result.normalized_input
        content_to_process = ""

        # Branch 1: Direct text input (Phase 2)
        if norm.input_type == "text":
            st.success("News text successfully validated and normalized!")
            st.markdown("### Input Summary (Text)")
            st.write("- **Input Type:** `Direct Text`")
            st.write(f"- **Word Count:** `{norm.metadata['word_count']}` words")
            st.write(f"- **Character Count:** `{norm.metadata['character_count']}` characters")
            with st.expander("Preview Cleaned Text", expanded=False):
                st.write(norm.cleaned_content)

            content_to_process = norm.cleaned_content

        # Branch 2: Article URL (Phase 3)
        elif norm.input_type == "url":
            with st.spinner(f"Fetching article from {norm.metadata['domain']} and extracting readable content..."):
                extraction = extract_article_from_url(norm.cleaned_content)

            if extraction.success:
                article = extraction.article
                st.success("Article content successfully fetched and extracted!")
                st.markdown(f"### {article.title}")
                st.markdown(f"**Source / Domain:** `{article.domain}`")

                if article.final_url != norm.cleaned_content:
                    st.markdown(f"**Redirected To:** `{article.final_url}`")
                if article.canonical_url:
                    st.markdown(f"**Canonical URL:** `{article.canonical_url}`")

                st.write(f"- **Word Count:** `{article.word_count}` words")
                st.write(f"- **Character Count:** `{article.character_count}` characters")

                with st.expander("View Extracted Article Content", expanded=False):
                    st.write(article.text)

                content_to_process = article.text
            else:
                st.markdown("### Article Extraction Notice")
                st.write(f"- **Status Code:** `{extraction.status.value}`")
                st.write(f"- **Target URL:** `{norm.cleaned_content}`")

                if extraction.status == ExtractionStatus.ACCESS_DENIED:
                    st.warning(f"🔒 **Automated Access Denied (HTTP 401/403)**\n\n{extraction.error_message}")
                elif extraction.status == ExtractionStatus.RATE_LIMITED:
                    st.warning(f"⏳ **Rate Limited (HTTP 429)**\n\n{extraction.error_message}")
                elif extraction.status == ExtractionStatus.NOT_FOUND:
                    st.error(f"🔍 **Article Page Not Found (HTTP 404)**\n\n{extraction.error_message}")
                elif extraction.status == ExtractionStatus.SERVER_ERROR:
                    st.error(f"⚠️ **Source Server Unavailable (HTTP 5xx)**\n\n{extraction.error_message}")
                elif extraction.status == ExtractionStatus.INTERNAL_ERROR:
                    st.error(f"⚙️ **Application Processing Error**\n\n{extraction.error_message}")
                elif extraction.status == ExtractionStatus.TIMEOUT:
                    st.error(f"⏱️ **Request Timed Out**\n\n{extraction.error_message}")
                elif extraction.status == ExtractionStatus.NETWORK_ERROR:
                    st.error(f"🌐 **Network / DNS Connection Failure**\n\n{extraction.error_message}")
                elif extraction.status in (ExtractionStatus.NO_ARTICLE_CONTENT, ExtractionStatus.CONTENT_TOO_SHORT):
                    st.warning(f"📄 **Readable Content Unreachable**\n\n{extraction.error_message}")
                else:
                    st.error(f"❌ **Extraction Error**\n\n{extraction.error_message}")

                if extraction.suggested_action:
                    st.info(f"💡 **Recommended Next Step:**\n\n{extraction.suggested_action}")
                    st.caption(
                        "To verify this story, simply copy the text from your browser, "
                        "paste it into the text box above, clear the URL, and click 'Verify News'."
                    )

        # End-to-End Pipeline (Phases 4 through 7)
        if content_to_process:
            try:
                with st.spinner("Extracting claims, retrieving external evidence, evaluating sources, and analyzing stances..."):
                    # Phase 4: Claim Extraction
                    claims = extract_claims(content_to_process, max_claims=max_claims)

                    if not claims:
                        st.info(
                            "ℹ️ **No verifiable factual claims detected:** "
                            "The provided content does not contain clear factual assertions "
                            "(e.g., statements with specific entities, numbers, dates, or institutional actions)."
                        )
                    else:
                        # Phase 5 / 5.1: Evidence Retrieval
                        claims_evidence = retrieve_evidence_for_claims(
                            claims=claims,
                            provider=active_provider,
                            max_sources_per_claim=max_sources,
                        )

                        # Phase 6: Source Evaluation
                        evaluations_dict: dict[str, ClaimEvaluationSummary] = {}
                        for cr in claims_evidence:
                            evaluations_dict[cr.claim.claim_id] = evaluate_claim_evidence(cr)

                        # Phase 7–9: Claim & Evidence Stance Analysis
                        claim_analyses: list[ClaimAnalysisResult] = []
                        for claim in claims:
                            eval_summary = evaluations_dict.get(claim.claim_id)
                            items = eval_summary.evaluated_items if eval_summary else []
                            filtered_items = eval_summary.filtered_irrelevant_items if eval_summary else None
                            analysis = synthesize_claim_stance(claim, items, filtered_items=filtered_items)
                            claim_analyses.append(analysis)

                        article_report = generate_article_report(claim_analyses)

                        # Render UI
                        render_overall_assessment(article_report)
                        render_claim_analysis_section(
                            claims_analyses=claim_analyses,
                            claims_evidence=claims_evidence,
                            evaluations_dict=evaluations_dict,
                            provider=active_provider,
                        )
                        render_verification_summary(article_report)
            except Exception as e:
                st.error(
                    "⚠️ **An unexpected error occurred during verification.**\n\n"
                    "The system encountered an internal issue while processing this content. "
                    "Please try again or use a different article/text input."
                )
                st.caption(f"Error type: {type(e).__name__}")
