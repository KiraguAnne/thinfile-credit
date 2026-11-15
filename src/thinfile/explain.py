"""Human-readable names for features, used in reason codes and charts."""
from __future__ import annotations

import re

SOURCE = {
    "INST": "Instalments",
    "POS": "POS/cash loans",
    "CC": "Credit card",
    "PREV": "Past applications",
    "BUR": "Bureau",
    "BB": "Bureau monthly status",
}

TOKENS = {
    "LATE_RATE": "share of payments late",
    "PARTIAL_RATE": "share of payments partial",
    "DPD_MEAN": "average days past due",
    "DPD_MAX": "worst days past due",
    "DPD_STD": "variability of days past due",
    "DPD_SHARE": "share of months past due",
    "DPD_DEF_SHARE": "share of months past due (with tolerance)",
    "PAY_RATIO_MEAN": "average share of instalment paid",
    "DAYS_EARLY_MEAN": "average days paid early",
    "SHORTFALL_SUM": "total unpaid instalment amount",
    "DAYS_SINCE_LAST_DUE": "days since last instalment due",
    "UTIL_MEAN": "average balance utilisation",
    "UTIL_MAX": "peak balance utilisation",
    "UTIL_STD": "utilisation volatility",
    "PAY_TO_MIN_MEAN": "payment vs. minimum due",
    "BELOW_MIN_SHARE": "share of months paid below minimum",
    "DRAWINGS_MEAN": "average cash drawings",
    "LAST_BALANCE": "latest balance",
    "LAST_LIMIT": "latest credit limit",
    "APPROVED_RATE": "approval rate",
    "REFUSED_RATE": "refusal rate",
    "CANCELED_RATE": "cancellation rate",
    "REFUSED_N": "number refused",
    "APPROVED_N": "number approved",
    "CREDIT_TO_ASK_MEAN": "credit granted vs. amount asked",
    "CREDIT_TO_ASK_MIN": "lowest credit granted vs. asked",
    "ANNUITY_TO_CREDIT_MEAN": "average repayment rate",
    "DOWN_PAYMENT_RATE_MEAN": "average down-payment rate",
    "TERM_MEAN": "average loan term",
    "DAYS_SINCE_LAST": "days since most recent",
    "DAYS_SINCE_FIRST": "days since first",
    "NEW_CLIENT_SHARE": "share as new client",
    "EARLY_CLOSE_SHARE": "share of loans closed early",
    "COMPLETED_SHARE": "share of loans completed",
    "CLOSED_SHARE": "share of credits closed",
    "ACTIVE_DEBT_RATIO": "active debt / active credit",
    "ACTIVE_DEBT_SUM": "active debt",
    "ACTIVE_CREDIT_SUM": "active credit amount",
    "OVERDUE_SUM": "amount overdue",
    "DAY_OVERDUE_MAX": "max days overdue",
    "MAX_OVERDUE_AMT": "max amount ever overdue",
    "CARD_SHARE": "share credit cards",
    "MICROLOAN_SHARE": "share microloans",
    "ACTIVE_ENDDATE_MEAN": "average days to end of active credits",
    "WORST": "worst status",
    "FUTURE_INST_SUM": "remaining instalments",
    "LATE_TREND": "late-payment trend (recent vs. 12m)",
    "DPD_TREND": "days-past-due trend (recent vs. 12m)",
    "UTIL_TREND": "utilisation trend (recent vs. 12m)",
    "BELOW_MIN_TREND": "below-minimum trend (recent vs. 12m)",
}

APP = {
    "EXT_SOURCE_1": "External score 1", "EXT_SOURCE_2": "External score 2", "EXT_SOURCE_3": "External score 3",
    "AGE_YEARS": "Age", "DAYS_BIRTH": "Age (days)", "EMPLOYED_YEARS": "Years employed",
    "DAYS_EMPLOYED": "Employment length (days)", "EMPLOYED_TO_AGE": "Share of life employed",
    "CREDIT_TO_INCOME": "Loan amount / income", "ANNUITY_TO_INCOME": "Repayment / income",
    "ANNUITY_TO_CREDIT": "Repayment rate (inverse of term)", "CREDIT_TO_GOODS": "Loan / goods price",
    "INCOME_PER_PERSON": "Income per household member", "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Repayment amount", "AMT_GOODS_PRICE": "Goods price", "AMT_INCOME_TOTAL": "Income",
    "DAYS_ID_PUBLISH": "Days since ID document issued", "DAYS_REGISTRATION": "Days since registration change",
    "DAYS_LAST_PHONE_CHANGE": "Days since phone change", "PHONE_CHANGE_TO_AGE": "Phone change recency vs. age",
    "ID_PUBLISH_TO_AGE": "ID issue recency vs. age", "NAME_EDUCATION_TYPE": "Education",
    "NAME_INCOME_TYPE": "Income type", "OCCUPATION_TYPE": "Occupation", "ORGANIZATION_TYPE": "Employer type",
    "NAME_CONTRACT_TYPE": "Loan type", "NAME_FAMILY_STATUS": "Family status", "OWN_CAR_AGE": "Car age",
    "REGION_RATING_CLIENT_W_CITY": "Region rating", "REGION_POPULATION_RELATIVE": "Region population density",
    "N_DOCUMENTS": "Documents provided", "N_MISSING_APP": "Blank fields on application",
    "DAYS_EMPLOYED_ANOM": "Not in employment (pensioner/unemployed)", "FLAG_OWN_CAR": "Owns a car",
    "FLAG_DOCUMENT_3": "Provided document 3", "HOUR_APPR_PROCESS_START": "Hour application started",
    "CODE_GENDER": "Gender", "N_ADDRESS_MISMATCH": "Address mismatches",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults (30+ days) among social contacts",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults (60+ days) among social contacts",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social contacts with credit history (30 days)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Social contacts with credit history (60 days)",
    "REGION_RATING_CLIENT": "Region rating", "NAME_HOUSING_TYPE": "Housing type",
    "FLAG_EMP_PHONE": "Gave employer phone", "FLAG_WORK_PHONE": "Gave work phone",
    "N_CONTACT_CHANNELS": "Contact channels provided", "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Household size", "TOTALAREA_MODE": "Home floor area",
    "REG_CITY_NOT_LIVE_CITY": "Lives outside registered city", "REG_CITY_NOT_WORK_CITY": "Works outside registered city",
    "LIVE_CITY_NOT_WORK_CITY": "Works outside home city", "AMT_REQ_CREDIT_BUREAU_YEAR": "Bureau enquiries (last year)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Bureau enquiries (last quarter)", "AMT_REQ_CREDIT_BUREAU_MON": "Bureau enquiries (last month)",
    "WEEKDAY_APPR_PROCESS_START": "Weekday of application", "NAME_TYPE_SUITE": "Accompanied by",
    "FLAG_OWN_REALTY": "Owns property", "FLAG_PHONE": "Gave home phone", "FLAG_EMAIL": "Gave email",
}


def describe_feature(name: str) -> str:
    if name in APP:
        return APP[name]
    m = re.match(r"^(INST|POS|CC|PREV|BUR|BB)_(.+?)(?:_(\d+M|ALL))?$", name)
    if not m:
        return name.replace("_", " ").capitalize()
    src, body, win = m.groups()
    body_txt = TOKENS.get(body)
    if body_txt is None:
        if body.startswith("N_"):
            body_txt = "number of " + body[2:].lower().replace("_", " ")
        elif body == "N":
            body_txt = "number of records"
        else:
            body_txt = body.lower().replace("_", " ")
    suffix = f" (last {win[:-1]} months)" if win and win != "ALL" else ""
    return f"{SOURCE[src]}: {body_txt}{suffix}"
