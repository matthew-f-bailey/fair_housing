import pandas as pd

from enums import Actions


def read_data() -> pd.DataFrame:
    return pd.read_csv("data/hmda_2017_ca_all-records_codes.csv")

def preprocess(df: pd.DataFrame, keep_percent=None) -> pd.DataFrame:
    """Reads in and preprocesses data"""

    # Remove those without decision, names for readability, code is given
    no_decision = [
        Actions.WITHDRAWN_BY_APPLICANT,
        Actions.CLOSED_FOR_INCOMPLETE
    ]
    for dec in no_decision:
        df = df[df['action_taken'] != dec]

    # Combine decision into binary
    approved = [Actions.LOAN_ORIGINATED, Actions.APPROVED_NOT_ACCEPTED]
    df['approved'] = df['action_taken'].apply(
        lambda a: 1 if a in approved else 0
    )

    # Drop co-app races, take primary
    df = df.drop([
        "applicant_race_2",
        "applicant_race_3",
        "applicant_race_4",
        "applicant_race_5",
        "co_applicant_race_1",
        "co_applicant_race_2",
        "co_applicant_race_3",
        "co_applicant_race_4",
        "co_applicant_race_5",
        "co_applicant_sex",
        "co_applicant_ethnicity",
    ], axis=1)
    df = df.rename(columns={"applicant_race_1": "applicant_race"})

    # Drop unwanted columns
    keepvars = [
        "agency_code",
        "owner_occupancy",
        "preapproval",
        "loan_type",
        "property_type",
        "loan_purpose",
        "loan_amount_000s",
        "state_code",
        "county_code",
        "applicant_race",
        "applicant_sex",
        "applicant_income_000s",
        "purchaser_type",
        "population",
        "hud_median_family_income",
        "hoepa_status",
        "lien_status",
        "number_of_owner_occupied_units",
        "number_of_1_to_4_family_units",
        "hud_median_family_income",
        "tract_to_msamd_income",
        "approved",
    ]
    df = df.loc[:, keepvars]
    df = df.dropna()

    # Recode gender to 0, 1 and remove when not given
    good_sex_idx = df[(df["applicant_sex"] != 1) & (df["applicant_sex"] != 2)].index
    df.drop(good_sex_idx, inplace=True)
    df["applicant_sex"] = df["applicant_sex"].apply(lambda sex: 0 if sex==1 else 1)

    # Recode race and drop when not given
    good_race_idx = df[
        (df["applicant_race"] != 1) &
        (df["applicant_race"] != 2) &
        (df["applicant_race"] != 3) &
        (df["applicant_race"] != 4) &
        (df["applicant_race"] != 5)
    ].index
    df.drop(good_race_idx, inplace=True)
    df["applicant_race"] = df["applicant_race"].apply(lambda race: race-1)


    if keep_percent is None:
        return df

    return df[0: round(keep_percent*len(df))]
