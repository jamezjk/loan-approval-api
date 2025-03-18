from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request model
class LoanRequest(BaseModel):
    age: int
    income: int
    purpose: str
    cibil_score: int
    past_repayment_history: str  # "clean", "delayed", "default"
    employment_type: str  # "salaried", "self-employed", "unemployed"
    monthly_debts: int  # Monthly obligations (existing loans, rent, etc.)
    loan_amount: int
    loan_tenure: int  # In years
    collateral_provided: bool = False
    co_applicant: bool = False


def evaluate_loan(
    age: int,
    income: int,
    purpose: str,
    cibil_score: int,
    past_repayment_history: str,
    employment_type: str,
    monthly_debts: int,
    loan_amount: int,
    loan_tenure: int,
    collateral_provided: bool,
    co_applicant: bool
):
    reasons = []
    base_interest_rate = 8.0  # Minimum interest rate

    # Age Check
    if age < 21 or age > 65:
        reasons.append("Applicant age must be between 21 and 65 years.")

    # Income Check
    if income < 15000:
        reasons.append("Income too low for any loan.")
    elif purpose.lower() == "home" and income < 30000:
        reasons.append("Home loans require at least ₹30,000 monthly income.")
    elif purpose.lower() == "business" and income < 50000:
        reasons.append("Business loans require at least ₹50,000 monthly income.")

    # Employment Type
    if employment_type.lower() == "unemployed":
        reasons.append("Unemployed applicants are not eligible.")
    elif employment_type.lower() == "self-employed":
        base_interest_rate += 0.5  # Self-employed = slightly higher rate

    # CIBIL Score Check
    if cibil_score < 600:
        reasons.append("CIBIL score too low for loan approval.")
    elif cibil_score < 750:
        reasons.append("Average CIBIL score. Higher interest rates may apply.")
        base_interest_rate += (750 - cibil_score) * 0.01  # Lower score = higher rate

    # Past Repayment History
    if past_repayment_history.lower() == "default":
        reasons.append("Previous loan default. Immediate rejection.")
        base_interest_rate += 3.0  # Big penalty for defaults
    elif past_repayment_history.lower() == "delayed":
        reasons.append("Delayed payments found. Needs review.")
        base_interest_rate += 1.0

    # Debt-to-Income Ratio Check (DTI)
    dti = (monthly_debts / income) * 100
    if dti > 40:
        reasons.append(f"High Debt-to-Income Ratio (DTI): {dti:.2f}% exceeds 40% limit.")
        base_interest_rate += 1.5

    # Existing Loans Check
    if monthly_debts > 0 and dti > 30:
        reasons.append("Too many existing loans. Further review needed.")
        base_interest_rate += 0.5

    # Loan Amount vs. Income Check
    max_loan = income * 50
    if loan_amount > max_loan:
        reasons.append(f"Requested loan amount exceeds 50x monthly income (Max: ₹{max_loan}).")
        base_interest_rate += 0.5

    # Collateral Check
    if loan_amount > 1000000 and not collateral_provided:
        reasons.append("Collateral is mandatory for loans exceeding ₹10 Lakhs.")
        base_interest_rate += 1.0

    # Loan Tenure Check
    if loan_tenure > 15 and purpose.lower() != "home":
        reasons.append("Only home loans can have tenures longer than 15 years.")
        base_interest_rate += 0.5

    # Co-Applicant Check
    if age > 55 and not co_applicant:
        reasons.append("Co-applicant required for applicants above 55 years of age.")
        base_interest_rate += 0.5
    elif co_applicant:
        base_interest_rate -= 0.5  # Reduces risk = lower interest

    # Interest Rate Boundaries
    interest_rate = max(8.0, min(base_interest_rate, 20.0))  # Cap at 20%

    # Decision Logic
    if (
        "Applicant age must be between 21 and 65 years." in reasons
        or "Income too low for any loan." in reasons
        or "Previous loan default. Immediate rejection." in reasons
        or "Unemployed applicants are not eligible." in reasons
    ):
        decision = "❌ Rejected"
    elif len(reasons) == 0:
        decision = "✅ Approved"
    else:
        decision = "🔄 More Info Needed"

    return {
        "decision": decision,
        "interest_rate": interest_rate,
        "reasons": reasons
    }


@app.post("/evaluate_loan")
def evaluate_loan_endpoint(request: LoanRequest):
    """API Endpoint to evaluate loan eligibility and interest rate."""
    result = evaluate_loan(
        age=request.age,
        income=request.income,
        purpose=request.purpose,
        cibil_score=request.cibil_score,
        past_repayment_history=request.past_repayment_history,
        employment_type=request.employment_type,
        monthly_debts=request.monthly_debts,
        loan_amount=request.loan_amount,
        loan_tenure=request.loan_tenure,
        collateral_provided=request.collateral_provided,
        co_applicant=request.co_applicant
    )
    return result

