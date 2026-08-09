"""Content for dlsu-faculty-preemployment-requirements.pdf.

The single consolidated pre-employment requirements checklist for a DLSU
faculty hire -- faculty-specific documents and national statutory documents
in one list. It states *what* must be submitted and *how HRMO judges
acceptability* (format, copy type, freshness). It deliberately does not
describe what each document is or how to obtain it -- that is left to
external lookup.
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
 "This checklist lists every document a faculty candidate must submit to "
 "HRMO before their employment start date -- both the faculty-specific "
 "documents and the national statutory documents -- together with the "
 "format specifications HRMO uses to determine whether a submitted "
 "document is acceptable."),
 ("note",
 "If a document is not listed here, HRMO does not require it -- ask your "
 "Department Chair before submitting anything not on this list."),

 ("h2", "1. Faculty-Specific Documents -- All Faculty Classes"),
 ("p",
 "Required of every faculty candidate regardless of class or rank."),
 ("table", ([
 ["Document", "Format requirement"],
 ["Original Transcript of Records (TOR)", "Original or CHED-certified true copy, issued within the last 6 months, bearing the registrar's dry seal."],
 ["Diploma (highest relevant degree)", "Original or notarized true copy."],
 ["Biodata / Curriculum Vitae", "DLSU biodata form or equivalent CV, signed and dated by the candidate."],
 ["Three (3) character/professional references", "Signed reference letters or a completed reference-contact form; at least one referee must be a former direct supervisor."],
 ["Clearance from previous employer", "Original clearance certificate or equivalent letter, issued within the last 3 months."],
 ["Certification of physical fitness to teach", "Medical certificate from a licensed physician, issued within the last 30 days, explicitly stating fitness to teach or perform the assigned role."],
 ["Valid government-issued ID", "Any one of: passport, driver's license, UMID, PRC ID, or Philippine national ID (PhilSys)."],
 ["2x2 ID photo (recent, white background)", "Taken within the last 6 months; formal attire; no filters or digital alteration."],
 ], [3.4, 5.8])),

 ("h2", "2. National Statutory Documents"),
 ("p",
 "Required of every candidate in addition to the faculty-specific set in "
 "§1."),
 ("table", ([
 ["Document", "Format requirement"],
 ["NBI Clearance", "Issued within the last 6 months before the start date (employer freshness window); purpose stated as 'Employment' where possible; must show 'No Record', or any 'hit' fully resolved before contract signing."],
 ["SSS number", "Provide the SSS number (or proof of a pending first-time application) plus a valid ID matching the name on record."],
 ["PhilHealth Identification Number (PIN)", "Provide the PIN (or proof of a pending first-time registration) plus a valid ID."],
 ["Pag-IBIG Membership ID (MID)", "Provide the MID number (or proof of a pending first-time registration) plus a valid ID."],
 ["Tax Identification Number (TIN)", "Provide the TIN (or proof of a pending first-time registration); confirm RDO transfer if the TIN was registered under a previous employer's RDO."],
 ["BIR Form 2316", "From the previous employer; required only if the candidate was employed for any part of the current calendar year before joining."],
 ], [3.4, 5.8])),

 ("h2", "3. Rank-Specific Requirements"),
 ("p",
 "In addition to §1 and §2, each academic rank carries minimum "
 "qualifications the candidate must document. These are evaluated by the "
 "Department Chair at screening (see the companion process guide, §2.1) "
 "but the supporting documents below must still be on file before the "
 "start date."),
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

 ("h2", "4. Requirements by Faculty Class"),

 ("h3", "4.1 Full-time Academic Faculty"),
 ("p",
 "Full-time candidates submit the complete set in §1, §2, and the "
 "rank-specific documentation in §3. Because full-time appointment "
 "carries a probationary period, HRMO additionally requires a signed "
 "acknowledgment of the probationary terms, provided by HRMO at Stage 5 "
 "of pre-boarding (see the companion process guide)."),

 ("h3", "4.2 Part-time Academic Faculty"),
 ("p",
 "Part-time candidates submit the same documents as full-time candidates "
 "(§1, §2, §3). Two items are relaxed given the shorter, per-term nature "
 "of the appointment: the clearance from the previous employer (§1) may "
 "be substituted with a signed self-certification of no unresolved "
 "obligations if the previous employer cannot issue a clearance within "
 "the compressed part-time timeline, and the physical fitness "
 "certification's issuance window is extended from 30 to 60 days."),

 ("h3", "4.3 Academic Service Faculty (ASF)"),
 ("p",
 "ASF candidates submit §1 and §2. Because ASF roles are non-teaching, "
 "the rank-specific documentation in §3 does not apply; instead, ASF "
 "candidates submit documentation of the role-specific competency "
 "evaluated at their Stage 2 interview (see the companion process guide, "
 "§3.3) -- for example, a relevant certification or license for "
 "laboratory or technical-support roles."),

 ("h2", "5. Document Format Rules"),
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

 ("h2", "6. Submission Channels and Deadlines"),
 ("p",
 "Documents are submitted to HRMO either in person at the HRMO front "
 "desk or through the HRMO document portal, within the 15-working-day "
 "window from the conditional offer date described in the companion "
 "process guide (10 working days for part-time hires). Partial "
 "submissions are accepted and logged incrementally; HRMO does not "
 "require all documents to arrive in a single batch."),

 ("h2", "7. Common Reasons a Document Is Marked Incomplete"),
 ("table", ([
 ["Reason", "Example", "Resolution"],
 ["Missing required field", "Reference letter without a signature or date.", "Request a corrected copy from the issuer."],
 ["Outside freshness window", "Previous-employer clearance issued 5 months ago (limit: 3 months).", "Request reissuance."],
 ["Wrong document type", "Photocopy of a diploma submitted where the TOR is required.", "Submit the correct document type."],
 ["Illegible scan", "Blurred or cropped photo of a physical-fitness certificate.", "Resubmit a clear scan or the original."],
 ["Name mismatch", "Name on the TOR does not match the government-issued ID (e.g., maiden vs. married name, uncorrected typo).", "Submit a supporting document (marriage certificate, affidavit of discrepancy) bridging the two names."],
 ], [2.2, 3.4, 3.4])),

 ("h2", "8. Cross-reference"),
 ("p",
 "For the end-to-end submission timeline, offices involved, and roles, "
 "see the companion document 'DLSU Faculty Pre-boarding Process Guide'."),
]
