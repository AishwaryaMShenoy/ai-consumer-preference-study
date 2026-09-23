import json
import os
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from recommendation import (
    ATTRIBUTES,
    normalize_weights,
    preference_strength,
    rank_products,
    top3_overlap,
)

from storage import (
    new_participant_id,
    append_result,
    append_chat,
    utc_now,
    storage_mode,
)

from ai_agent import get_response


load_dotenv()


st.set_page_config(
    page_title="AI-Mediated Consumer Preference Study",
    page_icon="🎧",
    layout="centered",
)


PRODUCTS = pd.read_csv("products.csv")


ATTRIBUTE_LABELS = {
    "price": "Price",
    "battery_hours": "Battery life",
    "anc": "Noise cancellation",
    "comfort": "Comfort",
    "sound": "Sound quality",
    "weight": "Lightness",
    "design": "Design",
}


# Product IDs are fixed; display order is randomized independently for baseline and post-test.
PRODUCT_IDS = PRODUCTS.product_id.tolist()


def init_state():
    # For local testing only, TEST_GROUP can force one condition.
    # Leave it blank/unset for real data collection so assignment remains random.
    test_group = os.getenv("TEST_GROUP", "").upper()

    assigned_group = (
        test_group
        if test_group in {"AI", "CONTROL"}
        else random.choice(["AI", "CONTROL"])
    )

    defaults = {
        "page": "consent",
        "participant_id": new_participant_id(),
        "group": assigned_group,
        "demographics": {},
        "baseline_ratings": {},
        "baseline_top3": [],
        "post_ratings": {},
        "post_top3": [],
        "trust": {},
        "autonomy": {},
        "confidence": {},
        "new_consideration": None,
        "tradeoff_reconsideration": None,
        "chat": [],
        "ai_weights": None,
        "ai_recommendations": [],
        "ai_ranked": [],
        "ai_concerns": {},
        "ai_final_top3": [],
        "ai_started": False,
        "control_started": False,
        "intervention_start": None,
        "submitted": False,
        "baseline_order": random.sample(PRODUCT_IDS, len(PRODUCT_IDS)),
        "post_order": random.sample(PRODUCT_IDS, len(PRODUCT_IDS)),
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go(page):
    st.session_state.page = page
    st.rerun()


def render_header():
    st.title("AI-Mediated Consumer Preference Study")
    st.caption(
        "Experimental study of consumer decision-making using a controlled "
        "fictional wireless-headphone catalog."
    )


def product_card(row, detailed=False):
    st.markdown(f"### {row['name']}")
    st.write(f"**₹{int(row['price']):,}**")

    st.write(
        f"🔋 {int(row['battery_hours'])}h battery · "
        f"🔇 Noise reduction {int(row['anc'])}/10 · "
        f"Comfort {int(row['comfort'])}/10"
    )

    st.write(
        f"🎵 Sound {int(row['sound'])}/10 · "
        f"⚖️ Lightness {int(row['weight'])}/10 · "
        f"✨ Design {int(row['design'])}/10"
    )

    if detailed:
        st.info(
            f"**Review summary:** {row['review_rating']}/5\n\n"
            f"**Known drawback:** {row['known_drawback']}\n\n"
            f"**Common review pattern:** {row['review_pattern']}\n\n"
            f"**Usually suits:** {row['best_for']}\n\n"
            f"**In plain language:** {row['plain_language_notes']}"
        )


def catalog_text():
    # Product knowledge supplied to the AI.
    # The same factual product information is visible to participants
    # before the consultation.
    cols = [
        "product_id",
        "name",
        "price",
        "battery_hours",
        "anc",
        "comfort",
        "sound",
        "weight",
        "design",
        "review_rating",
        "known_drawback",
        "review_pattern",
        "best_for",
        "plain_language_notes",
    ]

    return PRODUCTS[cols].to_csv(index=False)


def preference_form(key_prefix):
    vals = {}

    for a in ATTRIBUTES:
        vals[a] = st.slider(
            ATTRIBUTE_LABELS[a],
            min_value=1,
            max_value=7,
            value=4,
            key=f"{key_prefix}_{a}",
            help="1 = Not important, 7 = Extremely important",
        )

    return vals


def ranking_form(key_prefix, order):
    """Fast top-3 ranking instead of 12 pairwise choice tasks."""

    names = PRODUCTS.set_index("product_id")["name"].to_dict()
    options = [""] + order

    st.write(
        "Rank the **three headphones you would currently consider most**. "
        "You do not need to rank all ten."
    )

    st.caption(
        "1st = your strongest current choice · "
        "2nd = next choice · 3rd = third choice"
    )

    first = st.selectbox(
        "1st choice",
        options,
        format_func=lambda x: "— Select —" if x == "" else names[x],
        key=f"{key_prefix}_rank1",
    )

    remaining_1 = [x for x in options if x not in {"", first}]

    second = st.selectbox(
        "2nd choice",
        [""] + remaining_1,
        format_func=lambda x: "— Select —" if x == "" else names[x],
        key=f"{key_prefix}_rank2",
    )

    remaining_2 = [x for x in remaining_1 if x != second]

    third = st.selectbox(
        "3rd choice",
        [""] + remaining_2,
        format_func=lambda x: "— Select —" if x == "" else names[x],
        key=f"{key_prefix}_rank3",
    )

    return [x for x in [first, second, third] if x]


def compute_preference_change():
    # The main post-intervention signal is now choice/ranking change rather
    # than asking participants to repeat seven attribute sliders.
    return {}


def ai_user_messages():
    return [
        m
        for m in st.session_state.chat
        if m["role"] == "user"
    ]


# Initialize application state
init_state()
render_header()


# ============================================================
# CONSENT
# ============================================================

if st.session_state.page == "consent":

    st.markdown("## Participant Information")

    st.write(
        "This study examines consumer decision-making when selecting "
        "wireless headphones. You will state your initial preferences, "
        "rank a few products, and then either have a conversation with "
        "an AI shopping consultant or independently review product "
        "information. Your responses are recorded without your name "
        "or email address."
    )

    st.warning(
        "This is an academic research study. Participation is voluntary. "
        "Do not enter your name, email, phone number, or other identifying "
        "information. Obtain your institution/instructor approval before "
        "collecting real responses."
    )

    if storage_mode() == "APPS_SCRIPT":
        st.success(
            "Response storage is configured for the research deployment."
        )
    else:
        st.info(
            "Local/test mode: responses stay on the machine running the app. "
            "Do not use this mode for public data collection."
        )

    agree = st.checkbox(
        "I have read the information above and voluntarily agree to participate."
    )

    if st.button(
        "Continue",
        disabled=not agree,
        type="primary",
    ):
        go("demographics")


# ============================================================
# DEMOGRAPHICS
# ============================================================

elif st.session_state.page == "demographics":

    st.header("About You")

    st.caption(
        "Please do not enter your name, email, phone number, "
        "or other identifying information."
    )

    age = st.selectbox(
        "Age group",
        ["18–20", "21–23", "24–26", "27+"],
    )

    gender = st.selectbox(
        "Gender",
        [
            "Prefer not to say",
            "Female",
            "Male",
            "Non-binary",
            "Other",
        ],
    )

    headphone_use = st.selectbox(
        "How often do you use headphones?",
        [
            "Daily",
            "Several times a week",
            "Occasionally",
            "Rarely",
        ],
    )

    prior_purchase = st.selectbox(
        "Have you purchased wireless headphones before?",
        ["Yes", "No"],
    )

    ai_use = st.selectbox(
        "How often do you use AI assistants such as ChatGPT, Claude, or Gemini?",
        [
            "Never",
            "Rarely",
            "Sometimes",
            "Frequently",
            "Very frequently",
        ],
    )

    if st.button("Continue", type="primary"):

        st.session_state.demographics = {
            "age_group": age,
            "gender": gender,
            "headphone_use": headphone_use,
            "prior_headphone_purchase": prior_purchase,
            "ai_use_frequency": ai_use,
        }

        go("baseline")


# ============================================================
# BASELINE
# ============================================================

elif st.session_state.page == "baseline":

    st.header("Part 1 — Your Initial Preferences")

    st.write(
        "First, tell us what matters to you **before any shopping assistance**."
    )

    st.caption(
        "1 = Not important at all · 7 = Extremely important"
    )

    vals = preference_form("baseline")

    st.divider()

    st.subheader("Your initial shortlist")

    st.write(
        "Below are ten fictional products. You can see the same "
        "practical review information that will be available during "
        "the shopping task, so the AI is not introducing hidden "
        "product facts later."
    )

    for start in range(
        0,
        len(st.session_state.baseline_order),
        2,
    ):

        cols = st.columns(2)

        for j, pid in enumerate(
            st.session_state.baseline_order[start:start + 2]
        ):

            with cols[j]:

                row = PRODUCTS.loc[
                    PRODUCTS.product_id == pid
                ].iloc[0]

                product_card(
                    row,
                    detailed=True,
                )

    st.divider()

    top3 = ranking_form(
        "baseline",
        st.session_state.baseline_order,
    )

    if st.button("Continue", type="primary"):

        if len(top3) != 3:

            st.error(
                "Please select three different headphones "
                "for your initial shortlist."
            )

        else:

            st.session_state.baseline_ratings = vals
            st.session_state.baseline_top3 = top3

            go("intervention_intro")


# ============================================================
# INTERVENTION INTRO
# ============================================================

elif st.session_state.page == "intervention_intro":

    st.header("Part 2 — Shopping Task")

    if st.session_state.group == "AI":

        st.write(
            "You will have a short conversation with an AI shopping "
            "consultant. You can describe how you use headphones, ask "
            "about technical terms, ask about drawbacks, compare products, "
            "or challenge a recommendation. The assistant has access to "
            "the same product review summaries and practical usage "
            "information you saw before the consultation."
        )

        st.info(
            "Suggested time: about 4–6 minutes. There is no need to "
            "ask technical questions if you don't have any."
        )

        if st.button(
            "Start AI consultation",
            type="primary",
        ):

            st.session_state.ai_started = True
            st.session_state.intervention_start = time.time()

            go("ai_chat")

    else:

        st.write(
            "You will independently review the same fictional products. "
            "You can open the detailed product information, including "
            "review summaries and known drawbacks, but you will not "
            "receive personalized recommendations or an AI consultation."
        )

        st.info(
            "Suggested time: about 4–6 minutes. Review whichever "
            "products interest you most."
        )

        if st.button(
            "Start product research",
            type="primary",
        ):

            st.session_state.control_started = True
            st.session_state.intervention_start = time.time()

            go("control")


# ============================================================
# AI CHAT
# ============================================================

elif st.session_state.page == "ai_chat":

    st.header("AI Shopping Consultant")

    st.caption(
        "Ask naturally. You can describe how you use headphones, "
        "ask what a feature means, challenge a recommendation, "
        "or compare products."
    )

    if not st.session_state.chat:

        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": (
                    "Hi! I can help you compare these headphones based "
                    "on how you actually use them. You do not need to "
                    "know the technical terms. Tell me what you use "
                    "headphones for, what you care about most, and "
                    "anything you definitely want to avoid."
                ),
            }
        )

    # Display conversation
    for m in st.session_state.chat:

        with st.chat_message(
            "assistant" if m["role"] == "assistant" else "user"
        ):
            st.write(m["content"])

    prompt = st.chat_input(
        "e.g. 'I mostly travel and I don't want to charge every day'..."
    )

    if prompt:

        # Add participant message temporarily.
        st.session_state.chat.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:

            with st.spinner(
                "Updating the consultation..."
            ):

                text, weights, concerns, ranked = get_response(
                    st.session_state.chat,
                    catalog_text(),
                    normalize_weights(
                        st.session_state.baseline_ratings
                    ),
                )

        except Exception as exc:

            # Log actual technical information to Streamlit server logs.
            # This is NOT shown to participants.
            print(
                f"[Gemini consultation error] "
                f"participant={st.session_state.participant_id} "
                f"type={type(exc).__name__} "
                f"error={exc}"
            )

            # Remove failed participant message so they can try again.
            if (
                st.session_state.chat
                and st.session_state.chat[-1]["role"] == "user"
                and st.session_state.chat[-1]["content"] == prompt
            ):
                st.session_state.chat.pop()

            st.error(
                "The AI consultant is temporarily unavailable. "
                "Your study session is still active. Please wait a "
                "moment and send your message again."
            )

            st.stop()

        # Gemini succeeded
        st.session_state.ai_weights = weights
        st.session_state.ai_concerns = concerns
        st.session_state.ai_ranked = ranked.to_dict(
            "records"
        )

        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": text,
            }
        )

        st.rerun()

    # ========================================================
    # CURRENT AI RANKING
    # ========================================================

    if st.session_state.get("ai_ranked"):

        st.divider()

        st.subheader(
            "Your current recommendation ranking"
        )

        st.caption(
            "This ranking updates as the consultant learns more "
            "about what matters to you. It is not a permanent "
            "recommendation."
        )

        ranked_df = pd.DataFrame(
            st.session_state.ai_ranked
        )

        for _, r in ranked_df.iterrows():

            badge = (
                "⭐"
                if int(r["rank"]) <= 3
                else ""
            )

            with st.expander(
                f"{badge} {int(r['rank'])}. "
                f"{r['name']} — ₹{int(r['price']):,}",
                expanded=int(r["rank"]) <= 3,
            ):

                c1, c2, c3 = st.columns(3)

                c1.write(
                    f"🔋 {int(r['battery_hours'])}h battery"
                )

                c2.write(
                    f"🔇 ANC {int(r['anc'])}/10"
                )

                c3.write(
                    f"⭐ Reviews {float(r['review_rating']):.1f}/5"
                )

                c1.write(
                    f"🛋️ Comfort {int(r['comfort'])}/10"
                )

                c2.write(
                    f"🎵 Sound {int(r['sound'])}/10"
                )

                c3.write(
                    f"⚖️ Lightness {int(r['weight'])}/10"
                )

                st.write(
                    f"**Known drawback:** "
                    f"{r['known_drawback']}"
                )

                st.caption(
                    r["plain_language_notes"]
                )

    # ========================================================
    # FINAL AI SHORTLIST
    # ========================================================

    user_count = len(
        ai_user_messages()
    )

    if user_count >= 2:

        st.divider()

        st.subheader(
            "When you're ready: choose your final shortlist"
        )

        st.write(
            "The list below follows the latest AI ranking, "
            "so you do not need to remember earlier suggestions."
        )

        ranked_df = pd.DataFrame(
            st.session_state.ai_ranked
        )

        options = ranked_df[
            "product_id"
        ].tolist()

        names = dict(
            zip(
                ranked_df["product_id"],
                ranked_df["name"],
            )
        )

        st.caption(
            "The choices below are ordered using the latest "
            "AI ranking, so you can simply pick from the list "
            "rather than remember earlier suggestions."
        )

        first = st.selectbox(
            "1st choice",
            options,
            format_func=lambda pid:
                f"{int(ranked_df.loc[ranked_df.product_id == pid, 'rank'].iloc[0])}. "
                f"{names[pid]}",
            key="ai_final_1",
        )

        second_options = [
            pid
            for pid in options
            if pid != first
        ]

        second = st.selectbox(
            "2nd choice",
            second_options,
            format_func=lambda pid:
                f"{int(ranked_df.loc[ranked_df.product_id == pid, 'rank'].iloc[0])}. "
                f"{names[pid]}",
            key="ai_final_2",
        )

        third_options = [
            pid
            for pid in second_options
            if pid != second
        ]

        third = st.selectbox(
            "3rd choice",
            third_options,
            format_func=lambda pid:
                f"{int(ranked_df.loc[ranked_df.product_id == pid, 'rank'].iloc[0])}. "
                f"{names[pid]}",
            key="ai_final_3",
        )

        chosen = [
            first,
            second,
            third,
        ]

        if st.button(
            "Finish consultation",
            type="primary",
        ):

            st.session_state.ai_recommendations = options
            st.session_state.ai_final_top3 = chosen
            st.session_state.post_top3 = chosen

            try:

                append_chat(
                    st.session_state.participant_id,
                    st.session_state.chat,
                )

            except Exception as exc:

                st.error(
                    "The conversation could not be saved. "
                    "Please contact the study administrator "
                    "before continuing."
                )

                st.caption(
                    f"Technical detail: {exc}"
                )

                st.stop()

            go("evaluation")


# ============================================================
# CONTROL
# ============================================================

elif st.session_state.page == "control":

    st.header("Independent Product Research")

    st.write(
        "Explore the same products and review the same practical "
        "information. There is no personalized recommendation "
        "or conversational consultant in this condition."
    )

    for start in range(
        0,
        len(st.session_state.baseline_order),
        2,
    ):

        cols = st.columns(2)

        for j, pid in enumerate(
            st.session_state.baseline_order[start:start + 2]
        ):

            with cols[j]:

                row = PRODUCTS.loc[
                    PRODUCTS.product_id == pid
                ].iloc[0]

                product_card(
                    row,
                    detailed=True,
                )

    st.divider()

    st.write(
        "When you are ready, rank the three headphones "
        "you would seriously consider now."
    )

    chosen = ranking_form(
        "control_final",
        st.session_state.baseline_order,
    )

    if st.button(
        "Finish product research",
        type="primary",
        disabled=len(chosen) != 3,
    ):

        st.session_state.post_top3 = chosen
        st.session_state.ai_recommendations = []

        go("evaluation")


# ============================================================
# EVALUATION
# ============================================================

elif st.session_state.page == "evaluation":

    st.header(
        "Part 3 — Final Decision & Experience"
    )

    st.write(
        "Your final shortlist has been recorded. "
        "The questions below ask about the decision-making experience."
    )

    names = PRODUCTS.set_index(
        "product_id"
    )["name"].to_dict()

    st.info(
        "**Your final shortlist:** "
        + " · ".join(
            names[pid]
            for pid in st.session_state.post_top3
        )
    )

    st.write(
        "Your initial shortlist was: "
        + " · ".join(
            names[pid]
            for pid in st.session_state.baseline_top3
        )
    )

    st.divider()

    st.write(
        "Please indicate how much you agree with each statement."
    )

    scale = list(range(1, 8))

    labels = (
        "1 = Strongly disagree · "
        "7 = Strongly agree"
    )

    # ========================================================
    # TRUST
    # ========================================================

    st.subheader(
        "Trust in the shopping information"
    )

    trust_items = [
        "I trusted the information provided during the shopping task.",
        "The shopping assistance/information helped me understand the products.",
        "The product information was reliable enough for making a decision.",
    ]

    trust = {}

    for i, q in enumerate(trust_items):

        trust[q] = st.radio(
            q,
            scale,
            horizontal=True,
            key=f"trust_{i}",
        )

    st.caption(labels)

    # ========================================================
    # AUTONOMY
    # ========================================================

    st.subheader(
        "Perceived decision autonomy"
    )

    autonomy_items = [
        "I felt that I was in control of my final decision.",
        "The final decision reflected my own preferences.",
        "I felt free to reject the information or recommendations provided.",
        "The shopping assistance influenced my decision more than I intended.",
    ]

    autonomy = {}

    for i, q in enumerate(autonomy_items):

        autonomy[q] = st.radio(
            q,
            scale,
            horizontal=True,
            key=f"autonomy_{i}",
        )

    st.caption(labels)

    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.subheader(
        "Decision confidence"
    )

    confidence_items = [
        "I am confident that my final shortlist suits my needs.",
        "I feel confident about my current preference.",
        "The shopping task helped me make my decision.",
    ]

    confidence = {}

    for i, q in enumerate(confidence_items):

        confidence[q] = st.radio(
            q,
            scale,
            horizontal=True,
            key=f"confidence_{i}",
        )

    st.caption(labels)

    # ========================================================
    # PREFERENCE RECONSIDERATION
    # ========================================================

    st.subheader(
        "Preference reconsideration"
    )

    new_consideration = st.radio(
        "Did the shopping task introduce a product factor you had not seriously considered at the beginning?",
        [
            "Yes",
            "No",
            "Not sure",
        ],
        horizontal=True,
    )

    tradeoff_reconsideration = st.radio(
        "How much did you reconsider the trade-offs between product features?",
        scale,
        horizontal=True,
        help="1 = Not at all · 7 = Very much",
    )

    # ========================================================
    # SUBMIT
    # ========================================================

    if st.button(
        "Submit study",
        type="primary",
    ):

        st.session_state.trust = trust
        st.session_state.autonomy = autonomy
        st.session_state.confidence = confidence
        st.session_state.new_consideration = new_consideration
        st.session_state.tradeoff_reconsideration = (
            tradeoff_reconsideration
        )

        changes = compute_preference_change()

        baseline_weights = normalize_weights(
            st.session_state.baseline_ratings
        )

        baseline_strength = preference_strength(
            baseline_weights
        )

        duration = None

        if st.session_state.intervention_start:

            duration = round(
                time.time()
                - st.session_state.intervention_start,
                1,
            )

        autonomy_scores = list(
            autonomy.values()
        )

        # Reverse-score the final autonomy item so that
        # higher = greater autonomy.
        autonomy_scores[-1] = (
            8 - autonomy_scores[-1]
        )

        record = {
            "participant_id": st.session_state.participant_id,
            "timestamp_utc": utc_now(),
            "group": st.session_state.group,
            **st.session_state.demographics,

            "baseline_strength": baseline_strength,
            "post_strength": None,
            "preference_change_total_abs": None,
            "preference_change_signed_sum": None,

            "baseline_top1": st.session_state.baseline_top3[0],
            "post_top1": st.session_state.post_top3[0],

            "top1_changed": int(
                st.session_state.baseline_top3[0]
                != st.session_state.post_top3[0]
            ),

            "top3_overlap_count": top3_overlap(
                st.session_state.baseline_top3,
                st.session_state.post_top3,
            ),

            "choice_change_count": (
                3
                - top3_overlap(
                    st.session_state.baseline_top3,
                    st.session_state.post_top3,
                )
            ),

            "ai_recommendations": json.dumps(
                st.session_state.ai_recommendations
            ),

            "ai_message_count": len(
                st.session_state.chat
            ),

            "intervention_duration_seconds": duration,

            "trust_mean": float(
                np.mean(
                    list(trust.values())
                )
            ),

            "autonomy_mean": float(
                np.mean(
                    autonomy_scores
                )
            ),

            "confidence_mean": float(
                np.mean(
                    list(confidence.values())
                )
            ),

            "new_consideration": new_consideration,
            "tradeoff_reconsideration": (
                tradeoff_reconsideration
            ),
        }

        for a in ATTRIBUTES:

            record[
                f"baseline_{a}"
            ] = st.session_state.baseline_ratings[a]

        for i, pid in enumerate(
            st.session_state.baseline_top3,
            start=1,
        ):

            record[
                f"baseline_rank_{i}"
            ] = pid

        for i, pid in enumerate(
            st.session_state.post_top3,
            start=1,
        ):

            record[
                f"post_rank_{i}"
            ] = pid

        try:

            append_result(record)

        except Exception as exc:

            st.error(
                "We could not save your response. "
                "Please do not submit again repeatedly. "
                "Tell the study administrator what happened."
            )

            st.caption(
                f"Technical detail: {exc}"
            )

        else:

            st.session_state.submitted = True

            go("done")


# ============================================================
# DONE
# ============================================================

elif st.session_state.page == "done":

    st.header(
        "Thank you for participating!"
    )

    st.success(
        "Your responses have been recorded."
    )

    st.write(
        "The study investigates whether interaction with shopping "
        "assistance is associated with changes in consumer preferences. "
        "Your responses will be analyzed together with those of other "
        "participants."
    )

    st.info(
        f"Your anonymous study ID is "
        f"**{st.session_state.participant_id}**. "
        "Please keep this only if your research team needs it "
        "for study administration."
    )
