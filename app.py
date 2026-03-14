from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
import re

import pandas as pd
from pandas.errors import ParserError
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_gsheets import GSheetsConnection

st.set_page_config(
    page_title="Placement Eligibility Prediction",
    page_icon="🎯",
    layout="centered",
)

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset.csv"

ROUND_FILES = {
    "aptitude": BASE_DIR / "softskill1.csv",
    "technical": BASE_DIR / "technical1.csv",
    "hr": BASE_DIR / "hr1.csv",
}

ROUND_LABELS = {
    "aptitude": "Aptitude Round",
    "technical": "Technical Round",
    "hr": "HR Round",
}

ROUND_SCORE_COLUMNS = {
    "aptitude": "Aptitude_Score",
    "technical": "Technical_Score",
    "hr": "HR_Score",
}

ROUND_ORDER = ["aptitude", "technical", "hr"]
FIRST_ROUND_KEY = ROUND_ORDER[0]

INTERNSHIP_OPTIONS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, ">=6": 6}
PROJECT_OPTIONS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, ">=8": 8}
HACKATHON_OPTIONS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, ">=6": 6}
CERTIFICATION_OPTIONS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, ">=7": 7}
BACKLOG_OPTIONS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, ">=7": 7}

FEATURE_COLUMNS = [
    "Academic_Index",
    "Internship_Count",
    "Project_Count",
    "Hackathon_Count",
    "Communication_Score",
    "Technical_Score",
    "Aptitude_Score",
    "HR_Score",
    "Certification_Count",
    "Backlog_Count",
]
TARGET_COLUMN = "Placement_Status"
SUMMARY_COLUMNS = [
    "timestamp_utc",
    "name",
    "email",
    "prediction_label",
    "placement_probability",
    "aptitude_score",
    "technical_score",
    "hr_score",
    "academic_index",
    "communication_score",
    "internship_count",
    "project_count",
    "hackathon_count",
    "certification_count",
    "backlog_count",
]




def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings_to_try = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    last_error: Exception | None = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except ParserError as exc:
            last_error = exc

    raise ValueError(
        f"Could not read {path.name}. Tried encodings: {', '.join(encodings_to_try)}"
    ) from last_error



def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
        }
        .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .hero-box {
            background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.92));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        }
        .card-box {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.20);
        }
        .step-chip {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.18);
            color: #bfdbfe;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .question-box {
            background: rgba(30, 41, 59, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 16px;
            padding: 1rem;
            margin: 0.7rem 0 0.9rem 0;
            color: #e5e7eb;
            font-size: 1rem;
            line-height: 1.5;
        }
        .review-box {
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin: 0.8rem 0;
        }
        .tiny-note {
            color: #cbd5e1;
            font-size: 0.92rem;
            margin-top: -0.1rem;
            margin-bottom: 0.9rem;
        }
        .profile-summary {
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin: 0.8rem 0 1rem 0;
        }
        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            min-height: 46px;
        }
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
        }
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
            color: white !important;
        }
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.9rem;
        }
        div[data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
        div[data-testid="stMetricValue"] { color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_training_data() -> pd.DataFrame:
    df = read_csv_safe(DATASET_PATH)
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"dataset.csv is missing columns: {missing}")

    df = df[required_cols].copy().dropna()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


@st.cache_resource
def train_model():
    df = load_training_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, round(float(accuracy) * 100, 2), len(df)


@st.cache_data
def load_question_bank(round_key: str) -> list[dict]:
    df = read_csv_safe(ROUND_FILES[round_key])
    required_cols = {"question", "option_1", "option_2", "option_3", "option_4", "answer"}
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        raise ValueError(f"{ROUND_FILES[round_key].name} is missing columns: {missing}")

    records: list[dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "question": str(row["question"]),
                "options": [
                    str(row["option_1"]),
                    str(row["option_2"]),
                    str(row["option_3"]),
                    str(row["option_4"]),
                ],
                "answer": str(row["answer"]),
            }
        )
    return records


@st.cache_resource
def get_gsheet_connection():
    return st.connection("gsheets", type=GSheetsConnection)



def ensure_state() -> None:
    defaults = {
        "step": "profile",
        "user_inputs": {},
        "round_key": None,
        "questions": [],
        "question_index": 0,
        "answers": {},
        "quiz_scores": {},
        "result": None,
        "round_reviews": [],
        "saved_to_sheet": False,
        "save_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def reset_app() -> None:
    st.session_state.step = "profile"
    st.session_state.user_inputs = {}
    st.session_state.round_key = None
    st.session_state.questions = []
    st.session_state.question_index = 0
    st.session_state.answers = {}
    st.session_state.quiz_scores = {}
    st.session_state.result = None
    st.session_state.round_reviews = []
    st.session_state.saved_to_sheet = False
    st.session_state.save_status = None



def parse_cgpa(text: str):
    try:
        value = float(text.strip())
    except ValueError:
        return None
    if 0 <= value <= 10:
        return round(value, 2)
    return None



def is_valid_email(email: str) -> bool:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.fullmatch(pattern, email.strip()))



def current_answer():
    return st.session_state.answers.get(st.session_state.question_index)



def select_answer(answer_text: str) -> None:
    st.session_state.answers[st.session_state.question_index] = answer_text



def clear_answer() -> None:
    st.session_state.answers.pop(st.session_state.question_index, None)



def go_to_profile_from_quiz() -> None:
    st.session_state.step = "profile"



def previous_question() -> None:
    if (
        st.session_state.round_key == FIRST_ROUND_KEY
        and st.session_state.question_index == 0
    ):
        go_to_profile_from_quiz()
    elif st.session_state.question_index > 0:
        st.session_state.question_index -= 1



def next_question() -> None:
    if st.session_state.question_index < len(st.session_state.questions) - 1:
        st.session_state.question_index += 1



def start_round(round_key: str) -> None:
    bank = load_question_bank(round_key)
    total_available = len(bank)
    question_count = random.randint(min(5, total_available), min(15, total_available))

    st.session_state.round_key = round_key
    st.session_state.questions = random.sample(bank, question_count)
    st.session_state.question_index = 0
    st.session_state.answers = {}
    st.session_state.step = "quiz"



def next_round_key(current_round: str):
    index = ROUND_ORDER.index(current_round)
    return ROUND_ORDER[index + 1] if index + 1 < len(ROUND_ORDER) else None



def build_round_review(round_key: str) -> list[dict]:
    review_rows: list[dict] = []
    for index, question in enumerate(st.session_state.questions, start=1):
        user_answer = st.session_state.answers.get(index - 1, "Not answered")
        correct_answer = question["answer"]
        review_rows.append(
            {
                "round": ROUND_LABELS[round_key],
                "question_no": index,
                "question": question["question"],
                "your_answer": user_answer,
                "correct_answer": correct_answer,
                "status": "Correct" if user_answer == correct_answer else "Wrong",
            }
        )
    return review_rows



def calculate_score() -> float:
    correct = sum(
        st.session_state.answers.get(i) == q["answer"]
        for i, q in enumerate(st.session_state.questions)
    )
    total = len(st.session_state.questions)
    return round((correct / total) * 100, 2) if total else 0.0



def get_selected_label(options: dict[str, int], value: int, fallback: str) -> str:
    for label, number in options.items():
        if number == value:
            return label
    return fallback



def get_profile_defaults() -> dict[str, object]:
    saved = st.session_state.user_inputs or {}
    return {
        "full_name": saved.get("Full_Name", ""),
        "email": saved.get("Email", ""),
        "cgpa": "" if "Academic_Index" not in saved else f"{saved['Academic_Index'] / 10:.2f}".rstrip("0").rstrip("."),
        "internship_label": get_selected_label(INTERNSHIP_OPTIONS, saved.get("Internship_Count", 0), "0"),
        "project_label": get_selected_label(PROJECT_OPTIONS, saved.get("Project_Count", 0), "0"),
        "hackathon_label": get_selected_label(HACKATHON_OPTIONS, saved.get("Hackathon_Count", 0), "0"),
        "communication_score": float(saved.get("Communication_Score", 6.5)),
        "certification_label": get_selected_label(CERTIFICATION_OPTIONS, saved.get("Certification_Count", 0), "0"),
        "backlog_label": get_selected_label(BACKLOG_OPTIONS, saved.get("Backlog_Count", 0), "0"),
    }



def render_profile_summary() -> None:
    user_inputs = st.session_state.user_inputs
    if not user_inputs:
        return

    st.markdown('<div class="profile-summary">', unsafe_allow_html=True)
    st.markdown("**Entered details**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Name: {user_inputs['Full_Name']}")
        st.write(f"Email: {user_inputs['Email']}")
        st.write(f"CGPA: {user_inputs['Academic_Index'] / 10:.2f}")
        st.write(f"Internships: {user_inputs['Internship_Count']}")
        st.write(f"Projects: {user_inputs['Project_Count']}")
    with col2:
        st.write(f"Hackathons: {user_inputs['Hackathon_Count']}")
        st.write(f"Communication Score: {user_inputs['Communication_Score']:.1f}")
        st.write(f"Certifications: {user_inputs['Certification_Count']}")
        st.write(f"Backlogs: {user_inputs['Backlog_Count']}")
    st.markdown("</div>", unsafe_allow_html=True)



def save_summary_to_google_sheet() -> None:
    if st.session_state.saved_to_sheet or st.session_state.result is None:
        return

    result = st.session_state.result
    user_inputs = st.session_state.user_inputs

    summary_row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "name": user_inputs["Full_Name"],
        "email": user_inputs["Email"],
        "prediction_label": "Eligible for placement" if result["prediction"] == 1 else "Not eligible for placement",
        "placement_probability": result["probability"],
        "aptitude_score": result["scores"]["Aptitude Round"],
        "technical_score": result["scores"]["Technical Round"],
        "hr_score": result["scores"]["HR Round"],
        "academic_index": user_inputs["Academic_Index"],
        "communication_score": user_inputs["Communication_Score"],
        "internship_count": user_inputs["Internship_Count"],
        "project_count": user_inputs["Project_Count"],
        "hackathon_count": user_inputs["Hackathon_Count"],
        "certification_count": user_inputs["Certification_Count"],
        "backlog_count": user_inputs["Backlog_Count"],
    }

    try:
        conn = get_gsheet_connection()
        existing = conn.read(worksheet="summary", ttl=0)
        if existing is None or len(existing) == 0:
            updated = pd.DataFrame([summary_row], columns=SUMMARY_COLUMNS)
        else:
            existing_df = pd.DataFrame(existing).copy()
            for column in SUMMARY_COLUMNS:
                if column not in existing_df.columns:
                    existing_df[column] = ""
            updated = pd.concat(
                [existing_df[SUMMARY_COLUMNS], pd.DataFrame([summary_row])],
                ignore_index=True,
            )

        conn.update(worksheet="summary", data=updated)
        st.session_state.saved_to_sheet = True
        st.session_state.save_status = ("success", "Response saved to Google Sheet.")
    except Exception as exc:
        st.session_state.save_status = ("error", f"Google Sheet save failed: {exc}")



def make_prediction() -> None:
    model, model_accuracy, row_count = train_model()

    payload = {
        "Academic_Index": st.session_state.user_inputs["Academic_Index"],
        "Internship_Count": st.session_state.user_inputs["Internship_Count"],
        "Project_Count": st.session_state.user_inputs["Project_Count"],
        "Hackathon_Count": st.session_state.user_inputs["Hackathon_Count"],
        "Communication_Score": st.session_state.user_inputs["Communication_Score"],
        "Technical_Score": st.session_state.quiz_scores.get("Technical_Score", 0.0),
        "Aptitude_Score": st.session_state.quiz_scores.get("Aptitude_Score", 0.0),
        "HR_Score": st.session_state.quiz_scores.get("HR_Score", 0.0),
        "Certification_Count": st.session_state.user_inputs["Certification_Count"],
        "Backlog_Count": st.session_state.user_inputs["Backlog_Count"],
    }

    input_df = pd.DataFrame([payload])[FEATURE_COLUMNS]
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1]) * 100

    st.session_state.result = {
        "prediction": prediction,
        "probability": round(probability, 2),
        "model_accuracy": model_accuracy,
        "training_rows": row_count,
        "model_name": "Logistic Regression",
        "scores": {
            "Aptitude Round": st.session_state.quiz_scores.get("Aptitude_Score", 0.0),
            "Technical Round": st.session_state.quiz_scores.get("Technical_Score", 0.0),
            "HR Round": st.session_state.quiz_scores.get("HR_Score", 0.0),
        },
    }



def finish_round() -> None:
    round_key = st.session_state.round_key
    st.session_state.quiz_scores[ROUND_SCORE_COLUMNS[round_key]] = calculate_score()
    st.session_state.round_reviews.extend(build_round_review(round_key))
    upcoming = next_round_key(round_key)

    if upcoming is None:
        make_prediction()
        save_summary_to_google_sheet()
        st.session_state.step = "result"
        st.session_state.round_key = None
        st.session_state.questions = []
        st.session_state.question_index = 0
        st.session_state.answers = {}
    else:
        start_round(upcoming)



def skip_question() -> None:
    clear_answer()
    if st.session_state.question_index == len(st.session_state.questions) - 1:
        finish_round()
    else:
        st.session_state.question_index += 1



def render_profile_step() -> None:
    defaults = get_profile_defaults()

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="step-chip">Step 1</div>', unsafe_allow_html=True)
    st.subheader("Enter your details")
    st.markdown(
        '<div class="tiny-note">Fill your details to begin the interview rounds. You can come back and edit these details from the first Aptitude question using the Previous button.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.user_inputs:
        st.info("Your previous inputs are loaded below. Editing them will restart the rounds from the Aptitude round.")

    with st.form("profile_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            full_name = st.text_input("Full Name", value=defaults["full_name"], placeholder="Enter your full name")
            email = st.text_input("Email", value=defaults["email"], placeholder="Enter your email")
            cgpa_text = st.text_input("CGPA (0 to 10)", value=defaults["cgpa"], placeholder="Example: 8.25")
            internship_options = list(INTERNSHIP_OPTIONS.keys())
            internship_label = st.selectbox(
                "Internship Count",
                internship_options,
                index=internship_options.index(defaults["internship_label"]),
            )
            project_options = list(PROJECT_OPTIONS.keys())
            project_label = st.selectbox(
                "Project Count",
                project_options,
                index=project_options.index(defaults["project_label"]),
            )

        with col_b:
            hackathon_options = list(HACKATHON_OPTIONS.keys())
            hackathon_label = st.selectbox(
                "Hackathon Count",
                hackathon_options,
                index=hackathon_options.index(defaults["hackathon_label"]),
            )
            communication_score = st.slider(
                "Communication Score (0 to 10)",
                0.0,
                10.0,
                float(defaults["communication_score"]),
                0.1,
            )
            certification_options = list(CERTIFICATION_OPTIONS.keys())
            certification_label = st.selectbox(
                "Certification Count",
                certification_options,
                index=certification_options.index(defaults["certification_label"]),
            )
            backlog_options = list(BACKLOG_OPTIONS.keys())
            backlog_label = st.selectbox(
                "Backlog Count",
                backlog_options,
                index=backlog_options.index(defaults["backlog_label"]),
            )

        submitted = st.form_submit_button(
            "Start Interview",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        cgpa_value = parse_cgpa(cgpa_text)
        if not full_name.strip():
            st.error("Please enter your full name.")
        elif not is_valid_email(email):
            st.error("Please enter a valid email address.")
        elif cgpa_value is None:
            st.error("Please enter a valid CGPA between 0 and 10.")
        else:
            st.session_state.user_inputs = {
                "Full_Name": full_name.strip(),
                "Email": email.strip(),
                "Academic_Index": round(cgpa_value * 10, 2),
                "Internship_Count": INTERNSHIP_OPTIONS[internship_label],
                "Project_Count": PROJECT_OPTIONS[project_label],
                "Hackathon_Count": HACKATHON_OPTIONS[hackathon_label],
                "Communication_Score": float(communication_score),
                "Certification_Count": CERTIFICATION_OPTIONS[certification_label],
                "Backlog_Count": BACKLOG_OPTIONS[backlog_label],
            }
            st.session_state.round_reviews = []
            st.session_state.saved_to_sheet = False
            st.session_state.save_status = None
            st.session_state.result = None
            st.session_state.quiz_scores = {}
            start_round(FIRST_ROUND_KEY)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)



def render_quiz_step() -> None:
    round_key = st.session_state.round_key
    questions = st.session_state.questions
    idx = st.session_state.question_index
    question = questions[idx]
    selected = current_answer()
    is_first_question_of_first_round = round_key == FIRST_ROUND_KEY and idx == 0

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="step-chip">Step 2 · {ROUND_LABELS[round_key]}</div>', unsafe_allow_html=True)
    st.subheader(ROUND_LABELS[round_key])
    st.progress((idx + 1) / len(questions))
    st.caption(f"Question {idx + 1} of {len(questions)}")

    with st.expander("View your entered details", expanded=is_first_question_of_first_round):
        render_profile_summary()
        if is_first_question_of_first_round:
            st.caption("Use Previous to return to the user data section and edit details.")

    st.markdown(
        f'<div class="question-box"><strong>{question["question"]}</strong></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    for option_index, option_text in enumerate(question["options"]):
        with col1 if option_index % 2 == 0 else col2:
            if st.button(
                option_text,
                key=f"opt_{round_key}_{idx}_{option_index}",
                use_container_width=True,
            ):
                select_answer(option_text)
                st.rerun()

    if selected:
        st.success(f"Selected answer: {selected}")
    else:
        st.info("No answer selected.")

    nav1, nav2, nav3, nav4 = st.columns(4)

    with nav1:
        st.button(
            "Previous" if not is_first_question_of_first_round else "Previous (Edit Details)",
            on_click=previous_question,
            disabled=(idx == 0 and round_key != FIRST_ROUND_KEY),
            use_container_width=True,
        )

    with nav2:
        st.button(
            "Clear",
            on_click=clear_answer,
            disabled=(selected is None),
            use_container_width=True,
        )

    with nav3:
        st.button(
            "Skip",
            on_click=skip_question,
            use_container_width=True,
        )

    with nav4:
        if idx < len(questions) - 1:
            st.button(
                "Next",
                on_click=next_question,
                disabled=(selected is None),
                use_container_width=True,
                type="primary",
            )
        else:
            st.button(
                "Submit Round",
                on_click=finish_round,
                disabled=(selected is None),
                use_container_width=True,
                type="primary",
            )

    st.markdown("</div>", unsafe_allow_html=True)



def render_answer_review() -> None:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="step-chip">Answer Review</div>', unsafe_allow_html=True)
    st.subheader("Your answers vs correct answers")

    grouped: dict[str, list[dict]] = {}
    for row in st.session_state.round_reviews:
        grouped.setdefault(row["round"], []).append(row)

    for round_name, rows in grouped.items():
        st.markdown(f"### {round_name}")
        for row in rows:
            icon = "✅" if row["status"] == "Correct" else "❌"
            st.markdown('<div class="review-box">', unsafe_allow_html=True)
            st.markdown(f"**Q{row['question_no']}. {row['question']}**")
            st.write(f"Your answer: {row['your_answer']}")
            st.write(f"Correct answer: {row['correct_answer']}")
            st.write(f"Result: {icon} {row['status']}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



def render_result_step() -> None:
    result = st.session_state.result
    probability = result["probability"]

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="step-chip">Final Result</div>', unsafe_allow_html=True)
    st.subheader("Placement Eligibility Result")

    if result["prediction"] == 1:
        st.success("Eligible for placement")
    else:
        st.error("Not eligible for placement")

    if st.session_state.save_status:
        status_type, message = st.session_state.save_status
        if status_type == "success":
            st.success(message)
        else:
            st.error(message)

    st.progress(min(probability, 100.0) / 100.0)
    st.metric("Placement Eligibility Probability", f"{probability:.2f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Aptitude", f'{result["scores"]["Aptitude Round"]:.2f}%')
    with col2:
        st.metric("Technical", f'{result["scores"]["Technical Round"]:.2f}%')
    with col3:
        st.metric("HR", f'{result["scores"]["HR Round"]:.2f}%')

    st.markdown("</div>", unsafe_allow_html=True)
    render_answer_review()

    st.button(
        "Start Again",
        on_click=reset_app,
        use_container_width=True,
        type="primary",
    )



def main() -> None:
    apply_custom_style()
    ensure_state()

    required_paths = [DATASET_PATH, *ROUND_FILES.values()]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        st.error(f"Missing required files: {', '.join(missing)}")
        st.stop()

    st.markdown(
        """
        <div class="hero-box">
            <h1 style="margin-bottom:0.4rem;">🎯 Placement Eligibility Prediction</h1>
            <p style="margin-bottom:0; color:#cbd5e1;">
                Enter your details, complete the 3 interview rounds in order — Aptitude, Technical, and HR — and check your placement eligibility score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.step == "profile":
        render_profile_step()
    elif st.session_state.step == "quiz":
        render_quiz_step()
    else:
        render_result_step()


if __name__ == "__main__":
    main()
