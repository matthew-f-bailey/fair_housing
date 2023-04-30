# fair_housing

Data Source: https://www.consumerfinance.gov/data-research/hmda/historic-data/?geo=ca&records=all-records&field_descriptions=codes

The Fair Housing Act prohibits discrimination in housing because of:

- Race
- Color
- National Origin
- Religion
- Sex (including gender identity and sexual orientation)
- Familial Status
- Disability

Data

|Fields|Length|Type|
|---|---|---|
|As of Year|4|Numeric|
|Respondent ID|1| Alphanumeric|
|Agency Code|1|Alphanumeric|
|Loan Type|1|Numeric|
|Property Type|1|Alphanumeric|
|Loan Purpose|1|Numeric|
|Occupancy|1|Numeric|
|Loan Amount (000s)|5|Numeric|
|Preapproval|1|Alphanumeric|
|Action Type|1|Numeric|
|MSA/MD|5|Alphanumeric|
|State Code|2|Alphanumeric|
|County Code|3|Alphanumeric|
|Census Tract Number|7|Alphanumeric|
|Applicant Ethnicity|1|Alphanumeric|
|Co Applicant Ethnicity|1|Alphanumeric|
|Applicant Race 1|1|Alphanumeric|
|Applicant Race 2|1|Alphanumeric|
|Applicant Race 3|1|Alphanumeric|
|Applicant Race 4|1|Alphanumeric|
|Applicant Race 5|1|Alphanumeric|
|Co Applicant Race 1|1|Alphanumeric|
|Co Applicant Race 2|1|Alphanumeric|
|Co Applicant Race 3|1|Alphanumeric|
|Co Applicant Race 4|1|Alphanumeric|
|Co Applicant Race 5|1|Alphanumeric|
|Applicant Sex|1|Numeric|
|Co Applicant Sex|1|Numeric|
|Applicant Income (000s)|4|Alphanumeric|
|Purchaser Type|1|Alphanumeric|
|Denial Reason 1|1|Alphanumeric|
|Denial Reason 2|1|Alphanumeric|
|Denial Reason 3|1|Alphanumeric|
|Rate Spread|5|Alphanumeric|
|HOEPA Status|1|Alphanumeric|
|Lien Status|1|Alphanumeric|
|Edit Status|1|Alphanumeric|
|Sequence Number|7|Alphanumeric|
|Population|8|Alphanumeric|
|Minority Population %|6|Alphanumeric|
|FFIEC Median Family Income|8|Alphanumeric|
|Tract to MSA/MD Income %|6|Alphanumeric|
|Number of Owner-occupied units|8|Alphanumeric|
|Number of 1-to 4-Family units|8|Alphanumeric|
|Application Date Indicator|1|Numeric|

# HMDA LOAN/APPLICATION REGISTER CODE SHEET

## *RESPONDENT INFORMATION*

**Respondent ID**: 10 Character Identifier

**Agency**
- 1 -- Office of the Comptroller of the Currency (OCC)
- 2 -- Federal Reserve System (FRS)
- 3 -- Federal Deposit Insurance Corporation (FDIC)
- 5 -- National Credit Union Administration (NCUA)
- 7 -- Department of Housing and Urban Development (HUD)
- 9 -- Consumer Financial Protection Bureau (CFPB)

**Edit Status**
- Blank -- No edit failures
- 5 -- Validity edit failure only
- 6 -- Quality edit failure only
- 7 -- Validity and quality edit failures
## *PROPERTY LOCATION*
 **MSA/MD**: Metropolitan Statistical Area/Metropolitan Division

 **State**: Two-digit FIPS state identifier

 **County**: Three-digit FIPS county identifier

 **Tract**: Census tract number

## *LOAN INFORMATION*
 **Sequence Number**: One-up number scheme for each respondent to make each loan
unique

 **Loan Type**:
- 1 -- Conventional (any loan other than FHA, VA, FSA, or RHS loans)
- 2 -- FHA-insured (Federal Housing Administration)
- 3 -- VA-guaranteed (Veterans Administration)
- 4 -- FSA/RHS (Farm Service Agency or Rural Housing Service)

 **Property Type**:
- 1 -- One to four-family (other than manufactured housing)
- 2 -- Manufactured housing
- 3 -- Multifamily


 **Loan Purpose**:
- 1 -- Home purchase
- 2 -- Home improvement
- 3 -- Refinancing

 **Owner-Occupancy**:
- 1 -- Owner-occupied as a principal dwelling
- 2 -- Not owner-occupied
- 3 -- Not applicable

 **Loan Amount**: in thousands of dollars

 **Preapproval**:
- 1 -- Preapproval was requested
- 2 -- Preapproval was not requested
- 3 -- Not applicable

 **Action Taken**:
- 1 -- Loan originated
- 2 -- Application approved but not accepted
- 3 -- Application denied by financial institution
- 4 -- Application withdrawn by applicant
- 5 -- File closed for incompleteness
- 6 -- Loan purchased by the institution
- 7 -- Preapproval request denied by financial institution
- 8 -- Preapproval request approved but not accepted (optional reporting)

## *APPLICANT INFORMATION*

 **Ethnicity**:
- 1 -- Hispanic or Latino
- 2 -- Not Hispanic or Latino
- 3 -- Information not provided by applicant in mail, Internet, or telephone application
- 4 -- Not applicable
- 5 -- No co-applicant

 **Race**:
- 1 -- American Indian or Alaska Native
- 2 -- Asian
- 3 -- Black or African American
- 4 -- Native Hawaiian or Other Pacific Islander
- 5 -- White
- 6 -- Information not provided by applicant in mail, Internet, or telephone application
- 7 -- Not applicable
- 8 -- No co-applicant

 **Sex**:
- 1 -- Male
- 2 -- Female
- 3 -- Information not provided by applicant in mail, Internet, or telephone application
- 4 -- Not applicable
- 5 -- No co-applicant

 **Gross Annual Income**: in thousands of dollars

## *PURCHASER AND DENIAL INFORMATION*

**Type of Purchaser**
- 0 -- Loan was not originated or was not sold in calendar year covered by register
- 1 -- Fannie Mae (FNMA)
- 2 -- Ginnie Mae (GNMA)
- 3 -- Freddie Mac (FHLMC)
- 4 -- Farmer Mac (FAMC)
- 5 -- Private securitization
- 6 -- Commercial bank, savings bank or savings association
- 7 -- Life insurance company, credit union, mortgage bank, or finance company
- 8 -- Affiliate institution
- 9 -- Other type of purchaser

 **Reasons for Denial**:
- 1 -- Debt-to-income ratio
- 2 -- Employment history
- 3 -- Credit history
- 4 -- Collateral
- 5 -- Insufficient cash (downpayment, closing costs)
- 6 -- Unverifiable information
- 7 -- Credit application incomplete
- 8 -- Mortgage insurance denied
- 9 -- Other

## *OTHER DATA*
 **HOEPA Status (only for loans originated or purchased)**:
- 1 -- HOEPA loan
- 2 -- Not a HOEPA loan

 **Lien Status (only for applications and originations)**:
- 1 -- Secured by a first lien
- 2 -- Secured by a subordinate lien
- 3 -- Not secured by a lien
- 4 -- Not applicable (purchased loans)

 **Application Date Indicator**
- 0 -- Application Date >= 01-01-2004
- 1 -- Application Date < 01-01-2004
- 2 -- Application Date = NA (Not Available)

## *CENSUS INFORMATION*

 **Population**: total population in tract.

 **Minority Population %**: percentage of minority population to total population for tract.
(Carried to two decimal places)

**FFIEC Median Family Income**: FFIEC Median family income in dollars for the
MSA/MD in which the tract is located (adjusted annually by FFIEC).

**Tract to MSA/MD Median Family Income Percentage**: % of tract median family income
compared to MSA/MD median family income. (Carried to two decimal places)

**Number of Owner Occupied Units**: Number of dwellings, including individual
condominiums, that are lived in by the owner.

**Number of 1- to 4-Family units**: Dwellings that are built to house fewer than 5 families.