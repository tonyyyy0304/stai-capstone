"""Content for ph-statutory-preemployment.pdf.

Covers the national Philippine statutory pre-employment documents that the
DLSU Faculty Manual never mentions (verified: "NBI" appears zero times in
203 pages; SSS/PhilHealth/Pag-IBIG/BIR appear only in benefits/retirement
contexts, never as pre-employment submissions -- PLAN.md T1 gap, SCOPE.md
§2). Deliberately written to apply generally to Philippine employers, not
DLSU-specific, so it stays valid if this corpus is reused for another PH
university (see CLAUDE.md "generalizable to other PH universities").
"""

DOC = {
 "doc_id": "ph-statutory-preemployment",
 "title": "Philippine Statutory Pre-employment Requirements",
 "category": "onboarding",
 "effective_date": "2025",
 "version": "1.0",
 "author": "DLSU Human Resources Management Office (HRMO)",
 "running_head": "Philippine Statutory Pre-employment Requirements -- 2025",
}

BLOCKS = [
 ("h1", "Philippine Statutory Pre-employment Requirements"),
 ("p",
 "This document covers the national government-issued documents a new "
 "hire in the Philippines must submit before or shortly after starting "
 "employment. These requirements come from Philippine national agencies "
 "(the NBI, SSS, PhilHealth, Pag-IBIG, and the BIR), not from any single "
 "employer's manual -- the DLSU Faculty Manual 2021 does not describe "
 "them: the term 'NBI' does not appear anywhere in its 203 pages, and "
 "SSS, PhilHealth, Pag-IBIG, and BIR are mentioned only in benefits and "
 "retirement contribution contexts, never as pre-employment submissions. "
 "This document exists to close that gap and applies generally to "
 "Philippine employers, not to DLSU alone."),
 ("note",
 "For faculty-specific documents (transcripts, references, teaching "
 "clearance, etc.), see 'DLSU Faculty Pre-employment Requirements "
 "Checklist'. This document covers only the national statutory set."),

 ("h2", "1. NBI Clearance"),
 ("p",
 "The National Bureau of Investigation (NBI) Clearance certifies that "
 "the holder has no pending criminal case or derogatory record on file "
 "with the NBI at the time of issuance. It is the standard background "
 "clearance requested by Philippine employers as part of pre-employment "
 "screening."),
 ("h3", "1.1 How it is obtained"),
 ("p",
 "Applicants register through the NBI Clearance online e-service, "
 "select an appointment date and NBI branch or satellite office, pay the "
 "clearance fee online or at an accredited payment center, and appear in "
 "person on the appointment date for biometric capture (photo and "
 "fingerprints). First-time applicants and applicants with a common name "
 "are more likely to receive a 'hit' (see §1.3) and should apply with "
 "extra lead time."),
 ("h3", "1.2 Fields on the clearance"),
 ("table", ([
 ["Field", "What it shows"],
 ["Full name", "As registered with the NBI; must match the name on the applicant's government-issued ID exactly, subject to standard Filipino name-form variation (e.g., inclusion or omission of a middle name)."],
 ["Date of birth", "Used together with the name for identity matching."],
 ["Date of issue", "The date the clearance was released; the basis for freshness checks (§1.4)."],
 ["NBI reference number / control number", "Unique identifier for the specific clearance document, used to verify authenticity if needed."],
 ["Purpose", "The stated reason for the clearance, e.g., 'Employment' -- a clearance issued for a different stated purpose (e.g., 'Travel Abroad') may still be accepted at an employer's discretion but should be flagged for review."],
 ["Remarks / hit status", "'No Record' for a clean clearance; a 'hit' notation if the name matched an existing record pending verification."],
 ], [3.0, 6.0])),
 ("h3", "1.3 What an NBI 'hit' means"),
 ("p",
 "A 'hit' means the applicant's name matched an existing NBI record and "
 "requires manual verification at an NBI office before a final clearance "
 "is released -- it is not, by itself, evidence of a criminal record. "
 "Many hits resolve as name-only matches to a different individual. "
 "Employers should treat a pending hit as a hold on the clearance "
 "requirement, not as an automatic disqualification, and should require "
 "the final resolved clearance before finalizing an appointment "
 "(see the companion process guide's §6 on contingent status)."),
 ("h3", "1.4 Validity and freshness"),
 ("p",
 "The NBI Clearance itself carries a printed validity of one year from "
 "its date of issue. Employer freshness policies are typically stricter "
 "than the printed validity -- many Philippine employers, including "
 "DLSU, require the clearance to have been issued within a shorter "
 "window (commonly six months) before the employment start date, "
 "specifically because a clearance issued long before the start date no "
 "longer reflects the applicant's most current record. This employer "
 "freshness window is a policy choice layered on top of the document's "
 "own one-year validity, and each institution sets its own -- do not "
 "assume the printed one-year validity is what an employer will accept."),

 ("h2", "2. Social Security System (SSS)"),
 ("p",
 "The SSS is the mandatory social insurance program for private-sector "
 "and eligible workers in the Philippines, covering retirement, "
 "disability, sickness, maternity, and death benefits."),
 ("h3", "2.1 First-time applicants"),
 ("p",
 "An individual with no existing SSS number registers via the SSS "
 "online portal or an SSS branch, submitting one primary valid ID (e.g., "
 "passport, driver's license, UMID, PhilSys ID) or two secondary IDs. "
 "Registration issues a permanent SSS number that the applicant keeps "
 "for life across all subsequent employers."),
 ("h3", "2.2 Existing SSS members"),
 ("p",
 "Applicants who already have an SSS number submit that number and a "
 "valid ID; no re-registration is needed. The employer completes an "
 "Employment Report (SSS Form R-1A or its online equivalent) to link the "
 "new hire's SSS number to the employer's SSS account, which must be "
 "filed within 30 calendar days of the employee's start date under SSS "
 "reporting rules."),
 ("h3", "2.3 What the employer needs from the new hire"),
 ("bullets", [
 "SSS number (or proof of a pending first-time application).",
 "A valid government-issued ID matching the name on the SSS record.",
 "UMID card, if already issued, which also serves as an ATM-enabled "
 "benefits disbursement card.",
 ]),

 ("h2", "3. PhilHealth"),
 ("p",
 "The Philippine Health Insurance Corporation (PhilHealth) administers "
 "the National Health Insurance Program, mandatory for all "
 "employees."),
 ("h3", "3.1 Registration"),
 ("p",
 "First-time members register using the PhilHealth Member Registration "
 "Form (PMRF), submitted online or at a PhilHealth Local Health "
 "Insurance Office, to obtain a PhilHealth Identification Number (PIN). "
 "Existing members provide their PIN directly. The employer registers "
 "as a PhilHealth-accredited employer (if not already) and reports the "
 "new hire so that employer and employee contribution shares can be "
 "remitted starting from the first applicable payroll period."),
 ("h3", "3.2 What the employer needs from the new hire"),
 ("bullets", [
 "PhilHealth Identification Number (PIN), or a completed PMRF for "
 "first-time registration.",
 "A valid government-issued ID.",
 ]),

 ("h2", "4. Pag-IBIG Fund (HDMF)"),
 ("p",
 "The Home Development Mutual Fund (HDMF), commonly called Pag-IBIG, "
 "administers a national savings and housing-loan program mandatory for "
 "employees earning above a statutory threshold, and voluntary for "
 "others below it."),
 ("h3", "4.1 Registration"),
 ("p",
 "First-time members register via the Pag-IBIG online Virtual "
 "Pag-IBIG service or a branch, using the Member's Data Form (MDF), to "
 "obtain a Pag-IBIG Membership ID (MID) number. Existing members "
 "provide their MID number directly. The employer reports the new hire "
 "so contribution remittance can begin."),
 ("h3", "4.2 What the employer needs from the new hire"),
 ("bullets", [
 "Pag-IBIG Membership ID (MID) number, or a completed MDF for "
 "first-time registration.",
 "A valid government-issued ID.",
 ]),

 ("h2", "5. Bureau of Internal Revenue (BIR)"),
 ("p",
 "The BIR administers the Tax Identification Number (TIN) system and "
 "income tax withholding, both mandatory for employees."),
 ("h3", "5.1 First-time applicants"),
 ("p",
 "An individual with no existing TIN applies using BIR Form 1902 "
 "(Application for Registration for Individuals Earning Purely "
 "Compensation Income), typically facilitated by the new employer, which "
 "is then filed with the BIR Revenue District Office (RDO) covering "
 "the employer's address. Registration issues a permanent TIN."),
 ("h3", "5.2 Existing taxpayers"),
 ("p",
 "Applicants who already have a TIN provide it directly. If their prior "
 "employer's RDO differs from the new employer's RDO, the employee must "
 "file BIR Form 1905 to transfer their registration to the new RDO -- "
 "this is a common step new hires overlook, since it is the employee's, "
 "not the employer's, responsibility to initiate."),
 ("h3", "5.3 BIR Form 2316 from the previous employer"),
 ("p",
 "Employees who were employed for any part of the current calendar year "
 "before joining must submit BIR Form 2316 (Certificate of Compensation "
 "Payment / Tax Withheld) from their previous employer. This lets the "
 "new employer correctly compute year-to-date withholding tax and "
 "determine eligibility for substituted filing at year-end. New "
 "graduates or first-time employees with no prior employer in the "
 "current calendar year do not need to submit this form."),
 ("h3", "5.4 What the employer needs from the new hire"),
 ("bullets", [
 "TIN, or a completed BIR Form 1902 for first-time registration.",
 "BIR Form 1905, if transferring RDO from a previous employer.",
 "BIR Form 2316 from the previous employer, if employed earlier in "
 "the current calendar year.",
 ]),

 ("h2", "6. Additional Commonly Requested National Documents"),
 ("table", ([
 ["Document", "When it applies", "Notes"],
 ["Barangay Clearance", "Sometimes requested alongside or in place of NBI Clearance for local-level background confirmation.", "Issued by the applicant's barangay of residence; typically valid for a shorter window than the NBI Clearance."],
 ["Police Clearance", "Requested in addition to the NBI Clearance in some regions, or as an interim document while the NBI Clearance is pending.", "Issued by the local Philippine National Police station; narrower jurisdictional coverage than the NBI Clearance."],
 ["Medical / physical fitness certificate", "Required for roles with physical or classroom-presence demands; distinct from the DLSU-specific 'certification of physical fitness to teach'.", "See 'DLSU Faculty Pre-employment Requirements Checklist' for the faculty-specific version of this document."],
 ["PhilSys National ID", "Increasingly accepted as a single valid ID for SSS, PhilHealth, Pag-IBIG, and BIR transactions.", "Not yet mandatory in place of the other IDs listed above; accepted alongside them."],
 ], [2.6, 3.6, 2.8])),

 ("h2", "7. Employer Reporting Obligations"),
 ("p",
 "Once a new hire's statutory numbers are on file, the employer, not "
 "the employee, is responsible for reporting the new employee to SSS, "
 "PhilHealth, and Pag-IBIG within the reporting windows those agencies "
 "set (commonly within 30 calendar days of the start date), and for "
 "beginning correct income tax withholding with the BIR from the first "
 "applicable payroll run. Delayed employer reporting can create gaps in "
 "an employee's contribution record even when the employee's own "
 "documents were submitted on time -- this is why the pre-employment "
 "document deadline (see the companion process guide) is set well "
 "before the start date, to leave room for employer-side reporting."),

 ("h2", "8. Validity and Freshness Summary"),
 ("table", ([
 ["Document", "Printed / official validity", "Typical employer freshness practice"],
 ["NBI Clearance", "1 year from date of issue.", "Often 6 months or less at submission -- an employer policy layered on top of the printed validity, not a replacement for it."],
 ["Barangay Clearance", "Varies by barangay, commonly 6 months to 1 year.", "Usually required within 3 months of issuance."],
 ["Police Clearance", "Commonly 6 months.", "Usually required within 3 months of issuance."],
 ["Medical / fitness certificate", "No fixed national validity; employer-set.", "Commonly required within 30 to 60 days of issuance."],
 ["SSS / PhilHealth / Pag-IBIG / TIN numbers", "Permanent, once issued.", "No freshness window -- the number itself does not expire, only the supporting ID used to verify it may."],
 ], [2.6, 3.0, 3.4])),

 ("h2", "9. Consolidated Statutory Checklist"),
 ("numbers", [
 "NBI Clearance, within the employer's freshness window, purpose "
 "stated as Employment where possible.",
 "SSS number (or first-time application proof).",
 "PhilHealth Identification Number (or completed PMRF).",
 "Pag-IBIG Membership ID (or completed MDF).",
 "Tax Identification Number (or completed BIR Form 1902).",
 "BIR Form 1905, if transferring RDO.",
 "BIR Form 2316 from the previous employer, if applicable for the "
 "current calendar year.",
 ]),
]
