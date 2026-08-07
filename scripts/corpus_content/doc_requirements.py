"""Content for dlsu-faculty-preemployment-requirements.pdf.

Consolidates the faculty-specific pre-employment document requirements that
the Faculty Manual states as hiring *criteria* (per-rank Criteria for
Hiring pp.16-23; Hiring Procedure p.24; part-time p.67; ASF p.83) into a
single submission checklist with document specifications. National
statutory documents (NBI, SSS, PhilHealth, Pag-IBIG, BIR) are deliberately
NOT duplicated here -- they live in ph-statutory-preemployment.pdf, since
the Manual never mentions them (verified: "NBI" appears zero times).
"""

DOC = {
 "doc_id": "dlsu-faculty-preemployment-requirements",
 "title": "DLSU Faculty Pre-employment Requirements Checklist",
 "category": "onboarding",
 "effective_date": "2025",
 "version": "1.0",
 "author": "DLSU Human Resources Management Office (HRMO)",
 "running_head": "DLSU Faculty Pre-employment Requirements Checklist -- 2025",
}

BLOCKS = [
 ("h1", "DLSU Faculty Pre-employment Requirements Checklist"),
 ("p",
 "This checklist consolidates the faculty-specific documents required "
 "before a candidate's employment start date, drawn from the DLSU "
 "Faculty Manual 2021's per-rank Criteria for Hiring (pp.16-23) and "
 "Hiring Procedure (p.24), and extended with format specifications HRMO "
 "uses to determine whether a submitted document is acceptable. National "
 "statutory documents -- NBI Clearance, SSS, PhilHealth, Pag-IBIG, and "
 "BIR -- are covered separately in 'Philippine Statutory Pre-employment "
 "Requirements', since the Faculty Manual does not itself describe them."),
 ("note",
 "Every faculty-specific item below is traceable to the Faculty Manual's "
 "hiring criteria. If a document is not listed in the Manual and not "
 "listed in the statutory companion document, HRMO does not require it -- "
 "ask your Department Chair before submitting anything not on either "
 "list."),

 ("h2", "1. Core Document Set -- All Faculty Classes"),
 ("p",
 "The following documents are required of every faculty candidate "
 "regardless of class or rank, per the Faculty Manual's general hiring "
 "criteria (pp.16-23)."),
 ("table", ([
 ["Document", "Purpose", "Format requirement"],
 ["Original Transcript of Records (TOR)", "Verifies educational attainment against the rank's minimum degree requirement.", "Original or CHED-certified true copy, issued within the last 6 months, bearing the registrar's dry seal."],
 ["Diploma (highest relevant degree)", "Confirms degree conferral.", "Original or notarized true copy."],
 ["Biodata / Curriculum Vitae", "Establishes relevant teaching or professional experience.", "DLSU biodata form or equivalent CV, signed and dated by the candidate."],
 ["Three (3) character/professional references", "Supports the hiring panel's assessment of teaching and professional conduct.", "Signed reference letters or a completed reference-contact form; at least one referee must be a former direct supervisor."],
 ["Clearance from previous employer", "Confirms no unresolved obligations with the immediately preceding employer.", "Original clearance certificate or equivalent letter, issued within the last 3 months."],
 ["Clearance from concerned government agency", "Per Faculty Manual hiring criteria; satisfied jointly with the NBI Clearance in the statutory companion document.", "See 'Philippine Statutory Pre-employment Requirements', §1."],
 ["Certification of physical fitness to teach", "Confirms fitness for classroom or service duties.", "Medical certificate from a licensed physician, issued within the last 30 days, explicitly stating fitness to teach or perform the assigned role."],
 ["Valid government-issued ID", "Identity verification for all document matching.", "Any one of: passport, driver's license, UMID, PRC ID, or Philippine national ID (PhilSys)."],
 ["2x2 ID photo (recent, white background)", "For DLSU ID issuance.", "Taken within the last 6 months; formal attire; no filters or digital alteration."],
 ], [2.6, 3.4, 3.2])),

 ("h2", "2. Rank-Specific Requirements"),
 ("p",
 "In addition to the core set in §1, the Faculty Manual's Criteria for "
 "Hiring (pp.16-23) set rank-specific minimum qualifications that "
 "candidates must document. These are evaluated by the Department Chair "
 "at screening (see the companion process guide, §2.1) but the "
 "supporting documents below must still be on file before the start "
 "date."),
 ("table", ([
 ["Rank", "Minimum qualification to document", "Supporting document"],
 ["Instructor", "Master's degree in progress, or bachelor's degree with relevant professional licensure.", "TOR showing units earned toward the master's degree, or PRC license."],
 ["Assistant Professor", "Master's degree completed in the relevant discipline, plus at least 2 years of teaching or relevant professional experience.", "Diploma/TOR for the master's degree; employment certificates covering the experience requirement."],
 ["Associate Professor", "Doctoral degree in progress or master's degree with substantial publication or professional record, plus at least 5 years of relevant experience.", "TOR/diploma; CV with publication list or professional portfolio; employment certificates."],
 ["Professor", "Doctoral degree completed in the relevant discipline, plus a sustained record of teaching, research, and/or professional distinction.", "Doctoral diploma/TOR; CV with research and professional distinction record."],
 ], [1.8, 4.4, 3.0])),
 ("note",
 "Licensure requirements are discipline-specific -- e.g., Engineering, "
 "Accountancy, and Nursing faculty must additionally hold a current PRC "
 "license in the relevant field. Confirm discipline-specific licensure "
 "requirements with the Department Chair during screening."),

 ("h2", "3. Requirements by Faculty Class"),

 ("h3", "3.1 Full-time Academic Faculty"),
 ("p",
 "Full-time candidates submit the complete core set (§1) plus the "
 "rank-specific documentation (§2). Because full-time appointment "
 "carries a probationary period, HRMO additionally requires a signed "
 "acknowledgment of the probationary terms, provided by HRMO at Stage 5 "
 "of pre-boarding (see the companion process guide)."),

 ("h3", "3.2 Part-time Academic Faculty"),
 ("p",
 "Part-time candidates (Faculty Manual, part-time provisions, p.67) "
 "submit the same core set (§1) and rank-specific documentation (§2) as "
 "full-time candidates. Two items are relaxed for part-time hires given "
 "the shorter, per-term nature of the appointment: the clearance from "
 "the previous employer (§1) may be substituted with a signed "
 "self-certification of no unresolved obligations if the previous "
 "employer cannot issue a clearance within the compressed part-time "
 "timeline, and the physical fitness certification's issuance window is "
 "extended from 30 to 60 days."),

 ("h3", "3.3 Academic Service Faculty (ASF)"),
 ("p",
 "ASF candidates (Faculty Manual ASF provisions, p.83) submit the core "
 "set (§1). Because ASF roles are non-teaching, the rank-specific "
 "documentation in §2 does not apply; instead, ASF candidates submit "
 "documentation of the role-specific competency evaluated at their "
 "Stage 2 interview (see the companion process guide, §3.3) -- for "
 "example, a relevant certification or license for laboratory or "
 "technical-support roles."),

 ("h2", "4. Document Format Rules"),
 ("bullets", [
 "All documents in a language other than English or Filipino must be "
 "accompanied by a certified English translation.",
 "Photocopies are accepted only where explicitly marked 'true copy' "
 "acceptable in the tables above; otherwise HRMO requires the "
 "original for verification and returns it after copying.",
 "Digital submissions (scanned PDF or clear photo) are accepted for "
 "initial screening, but the original or a notarized true copy must "
 "be presented in person before contract signing.",
 "Any document bearing an issuance date outside its stated freshness "
 "window (e.g., a clearance older than 3 months) is treated as "
 "expired and must be reissued, not resubmitted as-is.",
 "Documents issued outside the Philippines must carry an apostille "
 "or, for non-Apostille Convention countries, authentication by the "
 "Philippine Embassy or Consulate.",
 ]),

 ("h2", "5. Submission Channels and Deadlines"),
 ("p",
 "Documents are submitted to HRMO either in person at the HRMO front "
 "desk or through the HRMO document portal, within the 15-working-day "
 "window from the conditional offer date described in the companion "
 "process guide (10 working days for part-time hires). Partial "
 "submissions are accepted and logged incrementally; HRMO does not "
 "require all documents to arrive in a single batch."),

 ("h2", "6. Common Reasons a Document Is Marked Incomplete"),
 ("table", ([
 ["Reason", "Example", "Resolution"],
 ["Missing required field", "Reference letter without a signature or date.", "Request a corrected copy from the issuer."],
 ["Outside freshness window", "Previous-employer clearance issued 5 months ago (limit: 3 months).", "Request reissuance."],
 ["Wrong document type", "Photocopy of a diploma submitted where the TOR is required.", "Submit the correct document type."],
 ["Illegible scan", "Blurred or cropped photo of a physical-fitness certificate.", "Resubmit a clear scan or the original."],
 ["Name mismatch", "Name on the TOR does not match the government-issued ID (e.g., maiden vs. married name, uncorrected typo).", "Submit a supporting document (marriage certificate, affidavit of discrepancy) bridging the two names."],
 ], [2.2, 3.4, 3.4])),

 ("h2", "7. Cross-reference"),
 ("p",
 "For NBI Clearance, SSS, PhilHealth, Pag-IBIG, and BIR requirements -- "
 "none of which appear in the Faculty Manual -- see the companion "
 "document 'Philippine Statutory Pre-employment Requirements'. For the "
 "end-to-end submission timeline and roles, see 'DLSU Faculty "
 "Pre-boarding Process Guide'."),
]
